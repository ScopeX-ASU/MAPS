import math
from abc import ABC, abstractmethod
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import (
    TreeClass,
    autoinit,
    frozen_field,
    frozen_private_field,
)
from fdtdx.core.physics.metrics import compute_energy
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.fdtd.container import DetectorState, ObjectContainer, SimulationState
from fdtdx.objects.sources.profile import GaussianPulseProfile


@autoinit
class StoppingCondition(TreeClass, ABC):
    """Abstract base class for defining custom stopping conditions in simulations.

    This class provides the interface for implementing custom termination criteria
    that can depend on simulation state, detector readings, field values, etc.
    """

    @abstractmethod
    def setup(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> Self:
        """Preparing condition for its use in the simulation."""
        return self

    @abstractmethod
    def _validate(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> None:
        """Pre-run validation; override in subclasses."""
        pass

    @abstractmethod
    def __call__(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> jax.Array:
        """Evaluate the stopping condition.

        Args:
            state (SimulationState): Current simulation state
            config (SimulationConfig): Configuration of the simulation
            objects (ObjectContainer): Objects in simulation

        Returns:
            jax.Array: Boolean scalar - True if simulation should continue, False if it should stop

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        raise NotImplementedError()


@autoinit
class TimeStepCondition(StoppingCondition):
    """Default stopping condition based on maximum time steps.

    This recreates the original behavior where simulation continues until
    a specified end time is reached.
    """

    def setup(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> Self:
        self._validate(state, config, objects)
        return self

    def _validate(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> None:
        pass

    def __call__(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> jax.Array:
        """Check if simulation should continue based on time step.

        Args:
            state (SimulationState): Current simulation state.
            config (SimulationConfig): Configuration of the simulation.
            objects (ObjectContainer): Objects in simulation.

        Returns:
            jax.Array: Boolean scalar - True if current time < end_time, False otherwise
        """
        curr_time_step, _ = state
        return config.time_steps_total > curr_time_step


@autoinit
class EnergyThresholdCondition(StoppingCondition):
    """This condition stops a simulation when the total energy in the simulation volume
    falls below a specified threshold. A minimum number of steps can be enforced
    before convergence checks are applied, and a hard cutoff is imposed at
    ``SimulationConfig.time_steps_total``. This condition is intended to be used with
    pulsed sources, where the total amount of energy input in the volume is
    finite, therefore total energy in the system is expected to converge towards zero eventually.

    Attributes:
        threshold (float, optional): Lower energy threshold for determining
            simulation termination. Defaults to ``1e-6``.
        min_steps (int, optional): The minimum number of time steps that must be
            completed before convergence checking begins. Defaults to
            ``0.1 * SimulationConfig.time_steps_total``.
        max_steps (int, optional): The maximum number of time steps in a
            simulation, regardless of the condition being fulfilled. Defaults to
            ``SimulationConfig.time_steps_total``.
    """

    threshold: float = frozen_field(default=1e-6)
    min_steps: int | None = frozen_field(default=None)
    max_steps: int | None = frozen_field(default=None)

    def setup(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> Self:
        self = self.aset(
            "max_steps",
            config.time_steps_total if self.max_steps is None else self.max_steps,
            create_new_ok=True,
        )
        self = self.aset(
            "min_steps",
            (
                round(config.time_steps_total * 0.1)
                if self.min_steps is None
                else self.min_steps
            ),
            create_new_ok=True,
        )
        self._validate(state, config, objects)

        return self

    def _validate(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> None:
        if self.threshold <= 0:
            raise ValueError(
                f"Energy threshold must be positive, got {self.threshold}."
            )

        if self.min_steps is not None and self.min_steps < 0:
            raise ValueError(
                f"Minimum steps must be non-negative, got {self.min_steps}."
            )

    def __call__(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> jax.Array:
        """Check if simulation should continue based on energy reaching a lower threshold.
        Energy checks are only able to terminate the simulation after the time step is larger than ``min_steps``.

        Args:
            state (SimulationState): Current simulation state.
            config (SimulationConfig): Configuration of the simulation.
            objects (ObjectContainer): Objects in simulation.

        Returns:
            jax.Array: Boolean scalar - True if condition not met and curr_time_step < max_steps.
        """
        if self.max_steps is None or self.min_steps is None:
            raise RuntimeError(
                "EnergyThresholdCondition.setup() must be called before use. "
            )
        curr_time_step, arrays = state
        time_condition = curr_time_step < self.max_steps
        min_steps_condition = curr_time_step < self.min_steps
        total_energy = jnp.sum(
            compute_energy(
                arrays.fields.E,
                arrays.fields.H,
                arrays.inv_permittivities,
                arrays.inv_permeabilities,
            )
        )
        converged = total_energy < self.threshold

        return time_condition & (min_steps_condition | ~converged)


@autoinit
class DetectorDecayAfterSourceCondition(StoppingCondition):
    """Stop after sources are off and a detector signal decays by a ratio.

    This is the fdtdx analogue of Meep's ``stop_when_fields_decayed`` pattern:
    keep running until every source has turned off, then wait until the selected
    detector reading has fallen below ``decay_by * peak`` where ``peak`` is the
    maximum detector magnitude observed after the sources are off.

    The detector must record every time step and reduce to one scalar value, for
    example ``EnergyDetector(name=..., reduce_volume=True)`` with the default
    ``OnOffSwitch()``.
    """

    detector_name: str = frozen_field()
    decay_by: float = frozen_field(default=1e-6)
    source_decay_by: float = frozen_field(default=1e-6)
    min_steps_after_sources: int = frozen_field(default=0)
    max_steps: int | None = frozen_field(default=None)
    _source_off_step: int | None = frozen_private_field(default=None)
    detector_interval: int = frozen_field(default=1)

    def setup(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> Self:
        self = self.aset(
            "max_steps",
            config.time_steps_total if self.max_steps is None else self.max_steps,
            create_new_ok=True,
        )
        source_off_step = _source_off_step(objects, config, self.source_decay_by)
        self = self.aset("_source_off_step", source_off_step, create_new_ok=True)
        self._validate(state, config, objects)
        return self

    def _validate(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> None:
        del objects
        if self._source_off_step is None:
            raise RuntimeError(
                "DetectorDecayAfterSourceCondition.setup() must be called before use."
            )
        if self.max_steps is None:
            raise RuntimeError(
                "DetectorDecayAfterSourceCondition.setup() must initialize max_steps."
            )
        if self.detector_interval < 1:
            raise ValueError(
                f"detector_interval must be >= 1, got {self.detector_interval}."
            )
        if not (0 < self.decay_by < 1):
            raise ValueError(f"decay_by must be in (0, 1), got {self.decay_by}.")
        if not (0 < self.source_decay_by < 1):
            raise ValueError(
                f"source_decay_by must be in (0, 1), got {self.source_decay_by}."
            )
        if self.min_steps_after_sources < 0:
            raise ValueError(
                f"min_steps_after_sources must be non-negative, got {self.min_steps_after_sources}."
            )
        if self.max_steps > config.time_steps_total:
            raise ValueError(
                "DetectorDecayAfterSourceCondition.max_steps cannot exceed config.time_steps_total "
                f"because detector buffers are allocated for {config.time_steps_total} steps; got {self.max_steps}."
            )
        if self.detector_name not in state[1].detector_states:
            available = tuple(state[1].detector_states.keys())
            raise KeyError(
                f"Detector '{self.detector_name}' not found. Available detectors: {available}"
            )

        det_state = state[1].detector_states[self.detector_name]
        readings = next(iter(det_state.values()))
        if readings.ndim != 2 or readings.shape[1] != 1:
            raise ValueError(
                f"Detector '{self.detector_name}' must reduce to one scalar per time step; "
                f"got detector state shape {readings.shape}."
            )
        # if readings.shape[0] != config.time_steps_total:
        #     raise ValueError(
        #         f"Detector '{self.detector_name}' must record every time step. "
        #         f"Number of detector readings: {readings.shape[0]}, time steps: {config.time_steps_total}."
        #     )
        expected_readings = (
            config.time_steps_total + self.detector_interval - 1
        ) // self.detector_interval
        if readings.shape[0] != expected_readings:
            raise ValueError(
                f"Detector '{self.detector_name}' appears to use a different recording interval. "
                f"With detector_interval={self.detector_interval}, expected {expected_readings} readings, "
                f"got {readings.shape[0]}."
            )

    def __call__(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> jax.Array:
        """Return True while the simulation should continue."""
        del config, objects
        if self._source_off_step is None or self.max_steps is None:
            raise RuntimeError(
                "DetectorDecayAfterSourceCondition.setup() must be called before use."
            )

        curr_time_step, arrays = state
        readings = next(iter(arrays.detector_states[self.detector_name].values()))
        values = jnp.abs(jnp.squeeze(readings, axis=1))
        # idxs = jnp.arange(values.shape[0])

        source_off_step = jnp.asarray(self._source_off_step, dtype=curr_time_step.dtype)
        min_stop_step = source_off_step + jnp.asarray(
            self.min_steps_after_sources, dtype=curr_time_step.dtype
        )
        max_steps = jnp.asarray(self.max_steps, dtype=curr_time_step.dtype)

        # At condition evaluation for state time_step=t, detectors have recorded
        # samples through t - 1.
        last_recorded = jnp.maximum(curr_time_step - 1, 0)

        interval = jnp.asarray(self.detector_interval, dtype=curr_time_step.dtype)
        current_sample_idx = jnp.minimum(
            last_recorded // interval,
            values.shape[0] - 1,
        )
        current = values[current_sample_idx]

        # Convert detector sample index -> actual simulation timestep.
        idxs = jnp.arange(values.shape[0], dtype=curr_time_step.dtype) * interval
        # current = values[last_recorded]
        after_source_history = (idxs >= source_off_step) & (idxs <= last_recorded)
        peak = jnp.max(jnp.where(after_source_history, values, 0.0))

        before_sources_off = curr_time_step <= source_off_step
        before_min_after_source = curr_time_step < min_stop_step
        no_peak_yet = peak <= 0
        not_decayed = current > peak * self.decay_by
        below_max_steps = curr_time_step < max_steps

        return below_max_steps & (
            before_sources_off | before_min_after_source | no_peak_yet | not_decayed
        )


def _source_off_step(
    objects: ObjectContainer, config: SimulationConfig, source_decay_by: float
) -> int:
    """Return the first time step after all sources are effectively off."""
    if not objects.sources:
        return 0

    off_step = 0
    for source in objects.sources:
        on_list = source.switch.calculate_on_list(
            num_total_time_steps=config.time_steps_total,
            time_step_duration=config.time_step_duration,
        )
        source_on_steps = [idx for idx, is_on in enumerate(on_list) if is_on]
        if not source_on_steps:
            continue

        switch_off_step = source_on_steps[-1] + 1
        pulse_off_step = _gaussian_pulse_off_step(
            source, config, source_on_steps[0], source_decay_by
        )
        source_off_step = (
            min(switch_off_step, pulse_off_step)
            if pulse_off_step is not None
            else switch_off_step
        )
        off_step = max(off_step, source_off_step)
    return off_step


def _gaussian_pulse_off_step(
    source, config: SimulationConfig, first_on_step: int, source_decay_by: float
) -> int | None:
    active_until_step = getattr(source, "active_until_step", None)
    if callable(active_until_step):
        return min(
            config.time_steps_total,
            max(first_on_step, active_until_step(source_decay_by)),
        )
    mode_components = getattr(source, "_mode_components", None)
    if mode_components:
        last_active_step = 0
        found_signal = False
        for component in mode_components:
            profile = getattr(component, "temporal_profile", None)
            signal = getattr(profile, "signal", None)
            signal_dt = getattr(profile, "time_step_duration", None)
            if signal is None or signal_dt is None:
                continue
            samples = np.abs(np.asarray(signal))
            peak = float(np.max(samples)) if samples.size else 0.0
            if peak <= 0:
                continue
            active = np.flatnonzero(samples > peak * source_decay_by)
            if active.size == 0:
                continue
            found_signal = True
            off_time = (int(active[-1]) + 1) * float(signal_dt)
            last_active_step = max(
                last_active_step,
                math.ceil(off_time / config.time_step_duration),
            )
        if found_signal:
            return min(config.time_steps_total, first_on_step + last_active_step)
    if not isinstance(source.temporal_profile, GaussianPulseProfile):
        return None

    spectral_width_hz = source.temporal_profile.spectral_width.get_frequency()
    sigma_t = 1.0 / (2 * math.pi * spectral_width_hz)
    t0 = 6 * sigma_t
    off_time = t0 + sigma_t * math.sqrt(2 * math.log(1.0 / source_decay_by))
    active_steps_until_off = math.ceil(off_time / config.time_step_duration)
    return min(config.time_steps_total, first_on_step + active_steps_until_off)


@autoinit
class DetectorConvergenceCondition(StoppingCondition):
    """Stopping condition based on convergence of values read off by a
    Detector with ``reduce_volume=True`` and number of readings equal to
    number of time steps (with ``switch=OnOffSwitch()``).

    This stopping condition starts off estimating convergence of values read off by a
    user-specified Detector by considering two subsets of its readings:
      (i) a number of previous periods [t - (prev_periods + 1)*spp, t - spp), and
      (ii) the last full period [t - spp, t),
    where t is the current time step, spp is the number of samples of the period
    over which the reading changes (it's usually half the period of the CW
    source in the case of PoyntingFluxDetector and EnergyDetector).
    Once it has them, it computes the mean over all periods of (i) such that the
    result has the same shape as (ii). After this, their Fourier transform is
    computed, and then the L2 norm of these two transforms, and stops the
    simulation once this norm falls below a specified threshold.
    A minimum number of steps are enforced before convergence checks are
    applied, and a hard cutoff of the simulation is imposed when the time step
    is equal to ``max_steps``.

    Attributes:
        detector_name (str): The name of the Detector from which
            readings will be obtained in order to determine when to
            halt the simulation.
        wave_character (WaveCharacter): The WaveCharacter fed to the source in the simulation.
            This is used to calculate the period over which the detector reading changes.
            For PoyntingFluxDetector and EnergyDetector, this is typically half
            the period of the CW source specified in this WaveCharacter.
        prev_periods (int, optional): Number of previous full periods used as a reference (>=1).
        threshold (float, optional): Relative change threshold for determining
            convergence. Defaults to ``1e-6``.
        min_steps (int, optional): The minimum number of time steps that the
            simulation must go on for before convergence checking begins. Defaults
            to ``(prev_periods + 1) * spp``, where ``spp`` is samples per period of
            the source. This ensures that there are enough readings to compute the
            Fourier transforms.
        max_steps (int, optional): The maximum number of time steps before the
            simulation is stopped, regardless of convergence.
            Defaults to ``SimulationConfig.time_steps_total``.
    """

    detector_name: str = frozen_field()
    wave_character: WaveCharacter = frozen_field()
    prev_periods: int = frozen_field(default=4)
    threshold: float = frozen_field(default=1e-6)
    min_steps: int | None = frozen_field(default=None)
    max_steps: int | None = frozen_field(default=None)
    _spp: int | None = frozen_private_field(default=None)

    def setup(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> Self:
        """Setting up internal attributes and validating inputs."""
        spp = round(self.wave_character.get_period() / config.time_step_duration)
        self = self.aset("_spp", spp, create_new_ok=True)
        self = self.aset(
            "max_steps",
            config.time_steps_total if self.max_steps is None else self.max_steps,
            create_new_ok=True,
        )
        self = self.aset(
            "min_steps",
            (
                round((self.prev_periods + 1) * spp)
                if self.min_steps is None
                else self.min_steps
            ),
            create_new_ok=True,
        )
        self._validate(state, config, objects)

        return self

    def _validate(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> None:
        if self._spp is None:
            raise RuntimeError(
                "DetectorConvergenceCondition: _spp was not initialized. Run setup() first."
            )
        _, arrays = state

        if (self.prev_periods + 1) * self._spp > config.time_steps_total:
            raise ValueError(
                "Number of samples over which DetectorConvergenceCondition computes is "
                "greater than the number of time steps in the simulation. "
                "Increase the time over which the simulation runs in SimulationConfig, "
                "decrease prev_periods, or use a source with a shorter period."
            )

        if self.detector_name not in arrays.detector_states:
            available = tuple(arrays.detector_states.keys())
            raise KeyError(
                f"Detector '{self.detector_name}' not found. Available detectors: {available}"
            )
        det_state: DetectorState = arrays.detector_states[self.detector_name]

        if all(k not in det_state for k in ("energy", "poynting_flux", "fields")):
            available = tuple(det_state.keys())
            raise KeyError(
                f"Chosen detector does not seem to be an EnergyDetector, PoyntingFluxDetector, FieldDetector.\n "
                f"Available keys: {available}"
            )
        readings = next(iter(det_state.values()))

        if readings.ndim != 2:
            raise ValueError(
                f"The selected detector must have reduce_volume=True. Therefore, "
                f"the DetectorState('{self.detector_name}') array must have two "
                f"dimensions; got ndim={readings.ndim}.\n"
            )

        if readings.shape[0] != config.time_steps_total:
            raise ValueError(
                f"The number of detector readings must be exactly the same as the number of time steps in the simulation. "
                f"Number of detector readings: {readings.shape[0]}, time steps: {config.time_steps_total}.\n"
            )

        if self.prev_periods < 1:
            raise ValueError(f"prev_periods must be >= 1; got {self.prev_periods}.")

        if self.threshold < 0:
            raise ValueError(
                f"Detector convergence threshold must be non-negative, got {self.threshold}."
            )
        if self.min_steps is None:
            raise RuntimeError(
                "DetectorConvergenceCondition: min_steps was not initialized."
            )
        if (
            self.min_steps is not None
            and self.min_steps < (self.prev_periods + 1) * self._spp
        ):
            raise ValueError(
                "min_steps must be larger than the number of steps used to compute convergence, "
                f"got {self.min_steps}, need more than {(self.prev_periods + 1) * self._spp}. "
                "You can also decrease prev_periods to match min_steps, or you can leave min_steps unset, "
                "as a suitable default will be used."
            )

    def __call__(
        self, state: SimulationState, config: SimulationConfig, objects: ObjectContainer
    ) -> jax.Array:
        """Check if simulation should continue based on the L2 norm between Fourier
        transforms of the last period and an average over a number of previous
        periods ``prev_periods``.

        Args:
            state (SimulationState): Current simulation state.
            config (SimulationConfig): Configuration of the simulation.
            objects (ObjectContainer): Objects in simulation.

        Returns:
            jax.Array: Boolean scalar - True if condition not met and curr_time_step < max_steps.
        """
        if self._spp is None or self.min_steps is None or self.max_steps is None:
            raise RuntimeError(
                "DetectorConvergenceCondition.setup() must be called before use."
            )

        # Assigning to local Variables
        spp = self._spp
        min_steps = self.min_steps

        threshold = self.threshold
        prev_periods = self.prev_periods

        curr_time_step, arrays = state
        converged: jnp.ndarray = jnp.array(False, dtype=bool)
        readings: jax.Array = next(
            iter(arrays.detector_states[self.detector_name].values())
        )

        # Always continue if below minimum steps, always stop if at end_step
        time_condition = curr_time_step < config.time_steps_total
        min_steps_condition = curr_time_step >= min_steps

        # Wrapping this in a func so we don't compute it until min_steps_condition == True
        def _compute_converged(_):
            start_ref = curr_time_step - (prev_periods + 1) * spp
            start_last = curr_time_step - spp

            # Clamp to valid bounds to avoid OOB under JIT (it doesn't like that)
            start_ref = jnp.clip(
                start_ref, 0, config.time_steps_total - prev_periods * spp
            )
            start_last = jnp.clip(start_last, 0, config.time_steps_total - spp)

            ref_2d = jax.lax.dynamic_slice(
                readings, (start_ref, 0), (prev_periods * spp, 1)
            )
            last_2d = jax.lax.dynamic_slice(readings, (start_last, 0), (self._spp, 1))

            readings_ref = jnp.squeeze(ref_2d, axis=1)  # (k*spp,)
            readings_last = jnp.squeeze(last_2d, axis=1)  # (spp,)

            # Take the reference readings and average all prev_periods together (prev_periods*spp,) -> (spp,)
            ref_periods = readings_ref.reshape(
                self.prev_periods, self._spp
            )  # (prev_periods, spp)
            ref_mean = jnp.mean(ref_periods, axis=0)  # (spp,)

            fft_ref = jnp.fft.rfft(ref_mean, n=self._spp)  # (spp//2 + 1,)
            fft_last = jnp.fft.rfft(readings_last, n=self._spp)  # (spp//2 + 1,)

            spectra_distance = jnp.linalg.norm(jnp.abs(fft_ref) - jnp.abs(fft_last))
            return spectra_distance < threshold

        converged = jax.lax.cond(
            min_steps_condition,
            _compute_converged,
            lambda _: jnp.array(False, dtype=bool),
            operand=None,
        )

        return (~min_steps_condition) | (time_condition & (~converged))
