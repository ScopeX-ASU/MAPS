import h5py
import matplotlib.pyplot as plt
import meep as mp
import torch.nn.functional as F
from IPython.display import Video

if __name__ == "__main__":
    # Test the metalens_FL30_init.py
    initialization_file_path = (
        "core/invdes/initialization/Si_metalens1D_for_850nm_FL30um.mat"
    )
    # open the file as read-only
    with h5py.File(initialization_file_path, "r") as f:
        atom_width = f["Si_width"][:]
        print("this is the shape of atom_width:", atom_width.shape)
        atom_width = atom_width * 1e6  # convert to um
        atom_width = atom_width[: len(atom_width) // 2 + 1]
        atom_width = atom_width[::-1]
    geometry = []
    center = 0
    for i, width in enumerate(atom_width):
        geometry.append(
            mp.Block(
                mp.Vector3(0.75, width, mp.inf),
                center=mp.Vector3(0, center, 0),
                material=mp.Medium(epsilon=3.6482**2),
            )
        )
        geometry.append(
            mp.Block(
                mp.Vector3(0.75, width, mp.inf),
                center=mp.Vector3(0, -center, 0),
                material=mp.Medium(epsilon=3.6482**2),
            )
        )
        center = center + 0.3
    cell_size = (4.25, 22.2, 0)
    input_port_width = (cell_size[0] - 0.75) / 2
    input_port = mp.Block(
        mp.Vector3(input_port_width, mp.inf, mp.inf),
        center=mp.Vector3(-0.75 / 2 - input_port_width / 2, 0, 0),
        material=mp.Medium(epsilon=1.45193**2),
    )
    geometry.append(input_port)
    PML = [0.5, 0.5]
    boundary = [
        mp.PML(PML[0], direction=mp.X),
        mp.PML(PML[1], direction=mp.Y),
    ]
    nonpml_vol = mp.Volume(
        mp.Vector3(),
        size=mp.Vector3(
            cell_size[0] - PML[0] * 2,
            cell_size[1] - PML[1] * 2,
        ),
    )
    src_center = [-cell_size[0] / 2 + PML[0] + 0.2, 0, 0]
    src_size = (0, 22.2, 0)
    fcen = 1 / 0.85
    # fwidth = (
    #     3 * 0.1 * (1 / (0.85 - 0.1 / 2) - 1 / (0.85 + 0.1 / 2))
    # )  # pulse frequency width
    sources = []
    sources.append(
        mp.EigenModeSource(
            src=mp.ContinuousSource(fcen),
            center=mp.Vector3(*src_center),
            size=src_size,
            eig_match_freq=True,
            component=mp.Hz,
        )
    )
    sim = mp.Simulation(
        resolution=200,
        cell_size=mp.Vector3(*cell_size),
        boundary_layers=boundary,
        geometry=geometry,
        sources=sources,
    )
    dft_obj = sim.add_dft_fields([mp.Ey], fcen, 0, 1, where=nonpml_vol)
    output = dict(
        eps=None,
        Ez=[],
        Ey=[],
    )

    def record_fields(sim):
        data = sim.get_efield_y()
        output["Ey"].append(data)

    at_every = [record_fields]
    f = plt.figure(dpi=150)
    Animate = mp.Animate2D(fields=mp.Ey, f=f, realtime=False, normalize=True)
    at_every.append(Animate)

    sim.run(
        mp.at_every(0.3, *at_every),
        until=50,
    )

    filename = "./unitest/metalens_FL30_init.mp4"
    Animate.to_mp4(20, filename)
    Video(filename)

    ez_data = sim.get_dft_array(dft_obj, mp.Ey, 0)

    ez_real = ez_data.real
    ez_imag = ez_data.imag

    plt.figure()
    plt.imshow(ez_real.T, cmap="RdBu")
    plt.colorbar()
    plt.savefig("./unitest/metalens_FL30_init_Ey_real.png", dpi=300)
    plt.close()
