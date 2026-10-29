"""Plot PanoRadar's processed output (run.py process_raw) for one rotation.

    python plot_processed.py /home/wildfire/PanoRadar/data/processed/spin_static/exp20261029-000 --frame 0

rf_npy/<frame>.npy is (range 256, elevation 64, azimuth 512), log scale.
Saves <traj>_frame<N>.png with: top-down view, range-azimuth panorama,
depth panorama (closest strong return per direction), elevation slice.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIN_M = 2.99792e8 * 5120e3 / (2 * 79.951e12 * 256)  # PanoRadar's assumed range bin, ~3.75 cm


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("traj", type=Path, help="processed trajectory folder (contains rf_npy/)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--max-range", type=float, default=10.0, help="metres to show")
    p.add_argument("--scale", type=float, default=1.0,
                   help="distance correction; 1.05 for our AWR1843AOP chirp (PanoRadar reads ~5%% short)")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()

    f = a.traj / "rf_npy" / f"{a.frame:05d}.npy"
    rf = np.load(f)  # (range, elev, azimuth)
    n_r, n_el, n_az = rf.shape
    print(f"{f}: shape {rf.shape} {rf.dtype}, min {rf.min():.3f} max {rf.max():.3f}, "
          f"nan {np.isnan(rf).sum()}")

    r = np.arange(n_r) * BIN_M * a.scale
    keep = r <= a.max_range
    r, rf = r[keep], rf[keep]
    az = np.linspace(0, 360, n_az, endpoint=False)
    el = np.linspace(45, -45, n_el)  # PanoRadar's elevation grid
    first = int(0.3 / (BIN_M * a.scale))  # skip the radar's own leakage near 0 m

    horiz = rf[:, n_el // 2 - 2:n_el // 2 + 2].mean(axis=1)  # (range, azimuth) near 0 deg elevation
    lo, hi = np.percentile(horiz[first:], [50, 99.9])

    # depth panorama: distance of the strongest return per (elevation, azimuth)
    depth = r[first + np.argmax(rf[first:], axis=0)]  # (elev, azimuth)

    fig = plt.figure(figsize=(14, 10))

    ax = fig.add_subplot(2, 2, 1)
    th = np.deg2rad(np.r_[az, 360])
    R, TH = np.meshgrid(np.r_[r, r[-1] + BIN_M * a.scale], th, indexing="ij")
    ax.pcolormesh(R * np.cos(TH), R * np.sin(TH), horiz, cmap="jet", vmin=lo, vmax=hi, shading="flat")
    ax.set_aspect("equal")
    ax.set_xlim(-a.max_range, a.max_range)
    ax.set_ylim(-a.max_range, a.max_range)
    ax.plot(0, 0, "w+", ms=10)
    ax.set_title("Top-down view (0 deg elevation)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(horiz, aspect="auto", origin="lower", cmap="jet", vmin=lo, vmax=hi,
              extent=[0, 360, r[0], r[-1]])
    ax.set_title("Range vs azimuth (0 deg elevation)")
    ax.set_xlabel("azimuth (deg, order of rotation)")
    ax.set_ylabel("distance (m)")

    ax = fig.add_subplot(2, 2, 3)
    im = ax.imshow(depth, aspect="auto", cmap="turbo", vmin=0, vmax=a.max_range,
                   extent=[0, 360, el[-1], el[0]])
    fig.colorbar(im, ax=ax, label="distance (m)")
    ax.set_title("Depth panorama (strongest return per direction)")
    ax.set_xlabel("azimuth (deg)")
    ax.set_ylabel("elevation (deg)")

    ax = fig.add_subplot(2, 2, 4)
    col = int(np.argmax(horiz[first:].max(axis=0)))  # azimuth with the strongest reflection
    sl = rf[:, :, col].T  # (elev, range)
    ax.imshow(sl, aspect="auto", cmap="jet", vmin=np.percentile(sl, 50), vmax=sl.max(),
              extent=[r[0], r[-1], el[-1], el[0]])
    ax.set_title(f"Elevation vs range at azimuth {az[col]:.0f} deg (strongest direction)")
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("elevation (deg)")

    fig.suptitle(f"{a.traj.name} rotation {a.frame}  (distance scale x{a.scale})")
    fig.tight_layout()
    out = a.out or Path(f"{a.traj.name}_frame{a.frame}.png")
    fig.savefig(out, dpi=110)
    print(f"saved {out.resolve()}")


if __name__ == "__main__":
    main()