"""Print shapes, timestamps and rotations of PanoRadar trajectory folders,
to compare yours with the sample dataset.

    python inspect_traj.py <sample traj folder> <your traj folder>
"""

import json
import sys
from pathlib import Path

import numpy as np

FILES = ["radar_data.npy", "radar_ts.txt", "lidar_data.npy", "lidar_ts.txt",
         "imu_data.npy", "imu_ts.txt", "data_params.npz"]


def show(folder: Path) -> None:
    print(f"\n=== {folder}")
    missing = [f for f in FILES if not (folder / f).exists()]
    if missing:
        print(f"  MISSING: {', '.join(missing)}")
    for f in ("radar_data.npy", "lidar_data.npy", "imu_data.npy"):
        if (folder / f).exists():
            a = np.load(folder / f, mmap_mode="r")
            print(f"  {f:16s} shape {tuple(a.shape)}  dtype {a.dtype}")
    for f in ("radar_ts.txt", "lidar_ts.txt", "imu_ts.txt"):
        if (folder / f).exists():
            t = np.loadtxt(folder / f, ndmin=1)
            dt = np.median(np.diff(t)) if len(t) > 1 else float("nan")
            print(f"  {f:16s} {len(t)} stamps, {t[-1] - t[0]:.2f} s, median step {dt * 1e3:.3f} ms")
    if (folder / "data_params.npz").exists():
        d = np.load(folder / "data_params.npz", allow_pickle=True)
        for k in d.files:
            v = d[k]
            print(f"  data_params[{k!r}]: shape {v.shape} dtype {v.dtype} "
                  f"value {v.item() if v.ndim == 0 else (v if v.size <= 6 else str(v[:6]) + ' ...')}")
        sep = d["radar_sep"]
        per = np.diff(sep)
        n_syn = int(np.deg2rad(90) / (2 * np.pi) * per[0])
        n_syn += n_syn % 2 == 0
        print(f"  -> {len(per)} rotation frames, beams/rotation median {np.median(per):.0f} "
              f"(min {per.min()}, max {per.max()}), synthetic aperture {n_syn} beams")
    rp = folder / "radar_params.json"
    if rp.exists():
        print(f"  radar_params.json: {json.loads(rp.read_text())}")
    else:
        print("  no radar_params.json -> PanoRadar uses its defaults "
              "(5120 ksps, 79.951 MHz/us, 77-81 GHz)")
    kind = "static" if "static" in folder.parent.name else "moving"
    print(f"  PanoRadar will treat this as: {kind} (from folder name '{folder.parent.name}')")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        show(Path(arg).expanduser())