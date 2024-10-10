"""
Date: 2024-08-24 21:37:48
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-08-24 21:41:25
FilePath: /Metasurface-Opt/data/fdtd/device.py
"""

import os

from pyutils.config import Config
from pyutils.general import ensure_dir


class SimulationConfig(Config):
    def __init__(self):
        super().__init__()
        self.update(
            dict(
                device=dict(
                    type="",
                    cfg=dict(),
                ),
                sources=[],
                simulation=dict(),
            )
        )


class Device(object):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.config = SimulationConfig()
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.sources = []
        self.geometry = []
        self.sim = None

    def update_device_config(self, device_type, device_cfg):
        self.config.device.type = device_type
        self.config.device.update(dict(cfg=device_cfg))

    def reset_device_config(self):
        self.config.device.type = ""
        self.config.device.update(dict(cfg=dict()))

    def add_source_config(self, source_config):
        self.config.sources.append(source_config)

    def reset_source_config(self):
        self.config.sources = []

    def update_simulation_config(self, simulation_config):
        self.config.update(dict(simulation=simulation_config))

    def reset_simulation_config(self):
        self.config.update(dict(simulation=dict()))

    def dump_config(self, filepath, verbose=False):
        ensure_dir(os.path.dirname(filepath))
        self.config.dump_to_yml(filepath)
        if verbose:
            print(f"Dumped device config to {filepath}")

    def trim_pml(self, resolution, PML, x):
        PML = [int(round(i * resolution)) for i in PML]
        return x[..., PML[0] : -PML[0], PML[1] : -PML[1]]
