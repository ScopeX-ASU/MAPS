import argparse
import os

import matplotlib
import numpy as np
from PIL import Image

matplotlib.rcParams.update(matplotlib.rcParamsDefault)
matplotlib.rcParams["text.usetex"] = False
import matplotlib.pyplot as plt

plt.switch_backend("Agg")
import meep as mp


def make_bragg_mask(
    period_um=0.324,
    num_periods=10,
    duty=0.5,
    wg_width_um=0.40,
    corrug_depth_um=0.05,
    resolution=1000.0,
    mode="aligned",
    y_margin_um=1.00,
):
    length_um = num_periods * period_um  # grating length
    tooth_len_um = duty * period_um  # tooth length
    height_um = (
        wg_width_um + 2 * corrug_depth_um + y_margin_um
    )  # waveguide width + teeth width + margin
    W = int(np.round(length_um * resolution))  # resolution: px/um
    H = int(np.round(height_um * resolution))
    mask = np.ones((H, W), dtype=np.uint8)

    def um_to_px_x(x_um):
        return int(np.clip(np.round(x_um * resolution), 0, W - 1))

    def um_to_px_y(y_um):
        return int(np.clip(np.round(y_um * resolution), 0, H - 1))

    def y_to_row(y_um_centered):
        return um_to_px_y((height_um / 2) - y_um_centered)

    # base strip
    y0, y1 = -wg_width_um / 2, +wg_width_um / 2
    r0, r1 = sorted([y_to_row(y0), y_to_row(y1)])
    mask[r0 : r1 + 1, :] = 0
    # teeth
    for i in range(num_periods):
        x_start_um = i * period_um + 0.5 * (period_um - tooth_len_um)
        x_end_um = x_start_um + tooth_len_um
        c0, c1 = sorted([um_to_px_x(x_start_um), um_to_px_x(x_end_um)])
        # top
        y_top0, y_top1 = wg_width_um / 2, wg_width_um / 2 + corrug_depth_um
        rt0, rt1 = sorted([y_to_row(y_top0), y_to_row(y_top1)])
        mask[rt0 : rt1 + 1, c0 : c1 + 1] = 0
        # bottom
        shift_um = 0.0 if mode == "aligned" else 0.5 * period_um
        xb0_um = x_start_um + shift_um
        xb1_um = x_end_um + shift_um
        cb0, cb1 = sorted([um_to_px_x(xb0_um), um_to_px_x(xb1_um)])
        y_bot0, y_bot1 = -wg_width_um / 2 - corrug_depth_um, -wg_width_um / 2
        rb0, rb1 = sorted([y_to_row(y_bot0), y_to_row(y_bot1)])
        if cb0 <= W - 1 and cb1 >= 0:
            mask[rb0 : rb1 + 1, max(0, cb0) : min(W, cb1 + 1)] = 0
    if mode == "aligned":
        # enforce vertical symmetry in case of rounding mismatches
        mask = np.minimum(mask, np.flipud(mask))
    return mask, (length_um, height_um)


def load_mask_from_image(
    path: str, threshold: int = 128, invert: bool = False, cropped: int = 0
) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    mask01 = (arr >= threshold).astype(np.uint8)
    if invert:
        mask01 = 1 - mask01
    if cropped != 0:
        mask01 = mask01[:, cropped:-cropped]
    return mask01


def save_mask_png(mask01, path, padding=0, pad_value=0):
    # Image.fromarray((mask01 * 255).astype(np.uint8), mode="L").save(path)
    reversed_mask = 1 - mask01
    if padding != 0:
        reversed_mask = np.pad(
            reversed_mask,
            ((padding, padding), (0, 0)),
            mode="constant",
            constant_values=pad_value,
        )
    Image.fromarray((reversed_mask * 255).astype(np.uint8), mode="L").save(path)


def simulate_device(
    device_tensor,
    resolution,
    sim_resolution,
    fcen,
    df,
    freq_points=100,
    device_name=None,
):
    print("enter the simulate_device function", flush=True)
    weights = device_tensor.astype(np.float64)
    height, width = device_tensor.shape
    PML_thickness = 1.0
    io_wg_length = 1.0
    sx = width / resolution + 2 * PML_thickness + 2 * io_wg_length
    sy = height / resolution + 2 * PML_thickness
    wg_width = 0.4
    cell_size = mp.Vector3(sx, sy, 0)
    si = mp.Medium(epsilon=8.11)
    sio2 = mp.Medium(epsilon=1.45**2)
    material_grid = mp.MaterialGrid(
        grid_size=mp.Vector3(width, height, 0),
        medium1=si,
        medium2=sio2,
        weights=weights.T,
        do_averaging=True,  # Enable subpixel smoothing for better accuracy
        beta=256,  # Sharp transition between materials
        eta=0.5,
    )
    grating = mp.Block(
        center=mp.Vector3(0, 0, 0),
        size=mp.Vector3(width / resolution, height / resolution, 0),
        material=material_grid,
    )
    io_wg = mp.Block(
        center=mp.Vector3(0, 0, 0), size=mp.Vector3(sx, wg_width, 0), material=si
    )
    geometry = [io_wg]
    source_center = mp.Vector3(-0.5 * sx + PML_thickness + io_wg_length * 0.5, 0, 0)
    source_size = mp.Vector3(0, wg_width, 0)
    source = mp.Source(
        mp.GaussianSource(frequency=fcen, fwidth=df),
        component=mp.Ez,
        center=source_center,
        size=source_size,
    )
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=sim_resolution,
        default_material=sio2,
        boundary_layers=[mp.PML(PML_thickness)],
        sources=[source],
        geometry=geometry,
    )
    # reflected flux
    refl_fr = mp.FluxRegion(
        center=mp.Vector3(-0.5 * sx + PML_thickness + io_wg_length * 0.75, 0, 0),
        size=mp.Vector3(0, 2 * wg_width, 0),
    )
    # transmitted flux
    tran_fr = mp.FluxRegion(
        center=mp.Vector3(0.5 * sx - PML_thickness, 0, 0),
        size=mp.Vector3(0, 2 * wg_width, 0),
    )
    refl = sim.add_flux(fcen, df, freq_points, refl_fr)
    tran = sim.add_flux(fcen, df, freq_points, tran_fr)
    flux_center = mp.Vector3(0.5 * sx - PML_thickness - io_wg_length * 0.5, 0, 0)
    print("ready to norm run...", flush=True)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, flux_center, 1e-3)
    )
    print("norm run done", flush=True)
    straight_refl_data = sim.get_flux_data(refl)
    straight_tran_flux = mp.get_fluxes(tran)
    sim.reset_meep()
    geometry = [io_wg, grating]
    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=sim_resolution,
        default_material=sio2,
        boundary_layers=[mp.PML(PML_thickness)],
        sources=[source],
        geometry=geometry,
    )
    os.makedirs("./data/fdfd/grating/sim/", exist_ok=True)
    plt.figure()
    sim.plot2D()
    plt.savefig(f"./data/fdfd/grating/sim/{device_name or 'bragg'}_sim.png", dpi=300)
    plt.close()
    # refl = sim.add_flux(fcen, df, freq_points, refl_fr)
    # tran = sim.add_flux(fcen, df, freq_points, tran_fr)
    # sim.load_minus_flux_data(refl, straight_refl_data)
    # sim.run(
    #     until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, flux_center, 1e-3)
    # )
    # freqs = mp.get_flux_freqs(refl)
    # refl_flux = mp.get_fluxes(refl)
    # tran_flux = mp.get_fluxes(tran)
    # wl, Rs, Ts = [], [], []
    # for i in range(len(freqs)):
    #     wl = np.append(wl, 1 / freqs[i])
    #     Rs = np.append(Rs, -refl_flux[i] / straight_tran_flux[i])
    #     Ts = np.append(Ts, tran_flux[i] / straight_tran_flux[i])
    # ====== 用本征模分解计算“某个模式”的传输，而非总 flux ======
    refl = sim.add_flux(fcen, df, freq_points, refl_fr)
    tran = sim.add_flux(fcen, df, freq_points, tran_fr)
    sim.load_minus_flux_data(refl, straight_refl_data)
    sim.run(
        until_after_sources=mp.stop_when_fields_decayed(50, mp.Ez, flux_center, 1e-3)
    )
    freqs = mp.get_flux_freqs(refl)
    refl_flux = mp.get_fluxes(refl)
    tran_flux = mp.get_fluxes(tran)
    wl, Rs, Ts = [], [], []
    for i in range(len(freqs)):
        wl = np.append(wl, 1 / freqs[i])
        Rs = np.append(Rs, -refl_flux[i] / straight_tran_flux[i])
        Ts = np.append(Ts, tran_flux[i] / straight_tran_flux[i])
    return wl, Rs, Ts


def main():
    ap = argparse.ArgumentParser(
        description="Build Bragg grating mask (PNG) and simulate with MEEP."
    )
    # build brag grating
    ap.add_argument("--mode", choices=["aligned", "misaligned"], default="aligned")
    ap.add_argument("--period", type=float, default=0.324)
    ap.add_argument("--num_periods", type=int, default=10)  # 75
    ap.add_argument("--duty", type=float, default=0.5)
    ap.add_argument("--wg_width", type=float, default=0.40)
    ap.add_argument("--corrug_depth", type=float, default=0.05)
    ap.add_argument("--padding", type=int, default=0)
    ap.add_argument("--resolution", type=float, default=1000.0)
    # simulation setting
    ap.add_argument("--sim_resolution", type=float, default=50.0)
    ap.add_argument("--lam_center", type=float, default=1.60)
    ap.add_argument("--fwidth_factor", type=float, default=40.0)
    ap.add_argument("--nfreq", type=int, default=351)
    ap.add_argument("--outdir", type=str, default="data/ilt/grating")
    # read brag grating
    ap.add_argument("--design_mask_png", type=str, default=None, help="path to image")
    ap.add_argument("--predicted_mask_png", type=str, default=None)
    ap.add_argument("--prefab_correction", type=str, default=None)
    ap.add_argument("--unet_correction", type=str, default=None)
    ap.add_argument("--threshold", type=int, default=128, help="threshold")
    ap.add_argument("--invert", action="store_true", help="invert the pixel value")
    ap.add_argument("--cropped", type=int, default=500)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.design_mask_png is not None:
        mask = load_mask_from_image(
            args.design_mask_png,
            threshold=args.threshold,
            invert=args.invert,
            cropped=args.cropped,
        )
        H_um = mask.shape[0] / args.resolution
        L_um = mask.shape[1] / args.resolution
        # png_path = os.path.join(args.outdir, "bragg_from_image.png")
        # Image.fromarray((mask*255).astype(np.uint8), mode="L").save(png_path)
        print(f"[+] Loaded PNG mask: {args.design_mask_png}")
        # print(f"    Saved (normalized) copy to: {png_path}")
        print(
            f"    Mask size: {mask.shape[1]} x {mask.shape[0]} px (L x H)  ->  {L_um:.3f} x {H_um:.3f} µm"
        )
    else:
        mask, (L_um, H_um) = make_bragg_mask(
            period_um=args.period,
            num_periods=args.num_periods,
            duty=args.duty,
            wg_width_um=args.wg_width,
            corrug_depth_um=args.corrug_depth,
            resolution=args.resolution,
            mode=args.mode,
        )
        png_path = os.path.join(args.outdir, f"bragg_{args.mode}_10.png")
        save_mask_png(mask, png_path, padding=args.padding)
        print(f"[+] Saved PNG mask: {png_path}")
        print(
            f"    Mask size: {mask.shape[1]} x {mask.shape[0]} px (L x H)  ->  {L_um:.3f} x {H_um:.3f} µm"
        )
    mask_prediction = None
    if args.predicted_mask_png is not None:
        mask_prediction = load_mask_from_image(
            args.predicted_mask_png, threshold=args.threshold, invert=args.invert
        )
    prefab_correction_mask_prediction = None
    if args.prefab_correction is not None:
        prefab_correction_mask_prediction = load_mask_from_image(
            args.prefab_correction,
            threshold=args.threshold,
            invert=args.invert,
        )
    if args.unet_correction is not None:
        unet_corrected_mask_prediction = load_mask_from_image(
            args.unet_correction,
            threshold=args.threshold,
            invert=args.invert,
            cropped=400,
        )

    # freq band
    f0 = 1.0 / args.lam_center
    df = f0 / args.fwidth_factor
    lam_min = 1.0 / (f0 + df)
    lam_max = 1.0 / (f0 - df)
    fcen = 0.5 * (1 / lam_min + 1 / lam_max)
    df_meep = 1 / lam_min - 1 / lam_max

    T = None
    wl, R, T = simulate_device(
        mask,
        resolution=args.resolution,
        sim_resolution=args.sim_resolution,
        fcen=fcen,
        df=df_meep,
        freq_points=args.nfreq,
        device_name=f"bragg_{args.mode}_design",
    )
    order = np.argsort(wl)
    wl, R, T = wl[order], np.array(R)[order], np.array(T)[order]

    if mask_prediction is not None:
        wl_predicted, _, T_pred = simulate_device(
            mask_prediction,
            resolution=args.resolution,
            sim_resolution=args.sim_resolution,
            fcen=fcen,
            df=df_meep,
            freq_points=args.nfreq,
            device_name=f"bragg_{args.mode}_design_predicted",
        )
        order = np.argsort(wl_predicted)
        wl_predicted, T_pred = wl_predicted[order], np.array(T_pred)[order]

    if args.prefab_correction is not None:
        wl_prefab, _, T_prefab_corrected_pred = simulate_device(
            prefab_correction_mask_prediction,
            resolution=args.resolution,
            sim_resolution=args.sim_resolution,
            fcen=fcen,
            df=df_meep,
            freq_points=args.nfreq,
            device_name=f"bragg_{args.mode}_prefab_correction",
        )
        order = np.argsort(wl_prefab)
        wl_prefab, T_prefab_corrected_pred = (
            wl_prefab[order],
            np.array(T_prefab_corrected_pred)[order],
        )

    if args.unet_correction is not None:
        wl_unet, _, T_unet_corrected_mask_pred = simulate_device(
            unet_corrected_mask_prediction,
            resolution=args.resolution,
            sim_resolution=args.sim_resolution,
            fcen=fcen,
            df=df_meep,
            freq_points=args.nfreq,
            device_name=f"bragg_{args.mode}_unet_correction",
        )
        order = np.argsort(wl_unet)
        wl_unet, T_unet_corrected_mask_pred = (
            wl_unet[order],
            np.array(T_unet_corrected_mask_pred)[order],
        )

    fig = plt.figure(figsize=(10, 6))

    if T is not None:
        # ax1 = fig.add_subplot(131)
        plt.plot(wl, T, label="GDS Transmission")
    if args.predicted_mask_png is not None:
        # ax2 = fig.add_subplot(132)
        plt.plot(wl_predicted, T_pred, label="Wafer image Transmission")
    if args.prefab_correction is not None:
        plt.plot(
            wl_prefab,
            T_prefab_corrected_pred,
            label="Prefab corrected Wafer image Transmission",
        )
    if args.unet_correction is not None:
        plt.plot(
            wl_unet,
            T_unet_corrected_mask_pred,
            label="Unet corrected Wafer image Transmission",
        )
    plt.xlabel("wavelength (um)")
    plt.ylabel("transmission", fontsize=12)
    plt.legend(loc="upper right")
    plt.title("Device Frequency Response", fontsize=14)
    plt.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, f"spectrum_{args.mode}.png"), dpi=200)
    plt.close(fig)
    print("[+] Done.")


if __name__ == "__main__":
    main()
