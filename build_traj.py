"""Convert a recorded session into PanoRadar trajectory folders (its 7-file format).

    python build_traj.py sessions/lab1_20261028-101500 --motor-csv motor.csv \
        --building ~/PanoRadar/data/raw/lab1_static --r-radar 0.08

Each rotation (motor angle passing --zero-angle) becomes one PanoRadar frame.
Keep "static" in the building folder name. LiDAR/IMU files are zero placeholders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

C = 2.99792e8


# ---------------------------------------------------------------- radar data
def iq_from_iiqq(iiqq: np.ndarray) -> np.ndarray:
    """xwr's default MSB_LSB_IQ layout -> complex64 (same as xwr.rsp.iq_from_iiqq)."""
    out = np.empty((*iiqq.shape[:-1], iiqq.shape[-1] // 2), dtype=np.complex64)
    out[..., 0::2] = iiqq[..., 2::4] + 1j * iiqq[..., 0::4]
    out[..., 1::2] = iiqq[..., 3::4] + 1j * iiqq[..., 1::4]
    return out


def fit_timestamps(ts: np.ndarray, period_s: float) -> tuple[np.ndarray, dict]:
    """Beams are hardware-timed, so fit ts = a + b*i to remove network jitter."""
    i = np.arange(len(ts))
    a, b = np.polyfit(i, ts, 1)[::-1]
    for _ in range(3):  # refit ignoring outliers (e.g. late packets)
        res = ts - (a + b * i)
        mad = np.median(np.abs(res - np.median(res))) + 1e-9
        keep = np.abs(res) < 5 * 1.4826 * mad + 1e-4
        b, a = np.polyfit(i[keep], ts[keep], 1)
    res = ts - (a + b * i)
    info = {"fitted_period_ms": b * 1e3, "config_period_ms": period_s * 1e3,
            "jitter_ms_p99": float(np.percentile(np.abs(res), 99) * 1e3)}
    # Arrival time lags capture time a little; the minimum residual is the
    # least-delayed beam, so shift the fit down to it.
    return a + b * i + res.min(), info


# ---------------------------------------------------------------- motor data
def load_motor(path: Path, ts_col: str, angle_col: str, units: str,
               steps_per_rev: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Load (timestamps, unwrapped angle in degrees) from a CSV file."""
    with open(path) as f:
        first = f.readline()
    has_header = any(c.isalpha() for c in first)
    if has_header:
        data = np.genfromtxt(path, delimiter=",", names=True, dtype=float,
                             invalid_raise=False)
        names = data.dtype.names

        def col(c):
            return data[names[int(c)]] if c.isdigit() else data[c]
        t, ang = col(ts_col), col(angle_col)
    else:
        data = np.genfromtxt(path, delimiter=",", dtype=float, invalid_raise=False)
        t, ang = data[:, int(ts_col)], data[:, int(angle_col)]

    ok = np.isfinite(t) & np.isfinite(ang)
    t, ang = t[ok], ang[ok]
    order = np.argsort(t, kind="stable")
    t, ang = t[order], ang[order]

    if units == "rad":
        ang = np.rad2deg(ang)
    elif units == "steps":
        if not steps_per_rev:
            raise SystemExit("--angle-units steps needs --steps-per-rev")
        ang = ang * 360.0 / steps_per_rev
    # handles both wrapped (0..360) and cumulative angle readings
    ang = np.rad2deg(np.unwrap(np.deg2rad(ang)))
    if t.min() < 1e9:
        raise SystemExit(
            f"{path}: timestamps don't look like Unix epoch seconds (min {t.min():.1f}). "
            "Motor and radar must share the PC clock (time.time()).")
    return t, ang


def rotation_starts(angle_at_beam: np.ndarray, zero_deg: float) -> np.ndarray:
    """Indices of the first beam of each full rotation (angle passes zero_deg)."""
    direction = np.sign(angle_at_beam[-1] - angle_at_beam[0]) or 1.0
    turns = np.floor(direction * (angle_at_beam - zero_deg) / 360.0)
    return np.flatnonzero(np.diff(turns) != 0) + 1


# ---------------------------------------------------------------- main
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--building", type=Path, required=True,
                   help="PanoRadar building folder, e.g. ~/PanoRadar/data/raw/lab1_static")
    p.add_argument("--config", type=Path, default=Path("config.yaml"),
                   help="config with the platform section (r_radar_m, motor_ticks_per_rev)")
    p.add_argument("--r-radar", type=float, default=None,
                   help="override platform.r_radar_m: rotation axis to antennas, metres")
    p.add_argument("--motor-csv", type=Path, default=None,
                   help="motor log (default: <session>/motor.csv)")
    p.add_argument("--motor-time-offset", type=float, default=0.0,
                   help="seconds to ADD to motor timestamps, if the motor log came from "
                        "another computer whose clock is off (e.g. -0.012)")
    p.add_argument("--assume-rev-per-s", type=float, default=None,
                   help="no motor log yet: assume this constant speed (rev/s) for testing. "
                        "Rotations are then evenly cut, but frame 0's facing direction is arbitrary")
    p.add_argument("--ts-col", default="0", help="timestamp column name or index")
    p.add_argument("--angle-col", default="1", help="angle column name or index")
    p.add_argument("--angle-units", choices=["deg", "rad", "steps"], default=None,
                   help="default: steps if platform.motor_ticks_per_rev is set, else deg")
    p.add_argument("--steps-per-rev", type=float, default=None,
                   help="override platform.motor_ticks_per_rev")
    p.add_argument("--zero-angle", type=float, default=0.0,
                   help="motor angle (deg) where each rotation/frame starts")
    p.add_argument("--direction", choices=["cw", "ccw"], default="cw",
                   help="physical spin direction seen from ABOVE. PanoRadar assumes cw; "
                        "ccw data is time-reversed so the image isn't mirrored")
    p.add_argument("--rotations-per-traj", type=int, default=20,
                   help="split into trajectories of this many rotations "
                        "(PanoRadar loads a whole trajectory into RAM)")
    p.add_argument("--conjugate", action="store_true",
                   help="conjugate I/Q; try this if targets appear at the wrong range")
    p.add_argument("--raw-ts", action="store_true", help="don't smooth radar timestamps")
    p.add_argument("--traj-prefix", default=None,
                   help="trajectory folder prefix (default exp<session date>)")
    args = p.parse_args()

    plat = {}
    if args.config.exists():
        plat = (yaml.safe_load(args.config.read_text()) or {}).get("platform") or {}
    r_radar = args.r_radar if args.r_radar is not None else plat.get("r_radar_m")
    if r_radar is None:
        raise SystemExit(f"Rotation radius not set: measure it and put it in {args.config} "
                         "under platform: r_radar_m (see README), or pass --r-radar.")
    steps_per_rev = args.steps_per_rev or plat.get("motor_ticks_per_rev")
    angle_units = args.angle_units or ("steps" if steps_per_rev else "deg")
    print(f"Platform: r_radar = {r_radar} m, motor units = {angle_units}"
          + (f" ({steps_per_rev:g} ticks/rev)" if angle_units == "steps" else ""))

    sess = args.session
    meta = json.loads((sess / "session.json").read_text())
    rad = meta["radar"]
    chirps, n_tx, n_rx, raw_last = meta["raw_shape"]
    n_samp = meta["shape"][-1]
    frame_bytes = meta["frame_bytes"]

    # PanoRadar decides static vs moving from the building folder's name
    moving = "static" not in args.building.name
    print(f"Mode: {'MOVING' if moving else 'static'} (building folder '{args.building.name}')")

    # --- radar frames + timestamps
    raw = np.memmap(sess / "radar_raw.bin", dtype=np.int16, mode="r")
    n_bin = raw.size * 2 // frame_bytes
    ts_tab = np.loadtxt(sess / "radar_ts.csv", delimiter=",", skiprows=1, ndmin=2)
    n = min(n_bin, len(ts_tab))
    raw = raw[: n * frame_bytes // 2].reshape(n, chirps, n_tx, n_rx, raw_last)
    ts_raw, complete = ts_tab[:n, 0], ts_tab[:n, 1].astype(bool)
    print(f"Radar: {n} beams, {np.count_nonzero(~complete)} incomplete, "
          f"{(ts_raw[-1] - ts_raw[0]):.1f} s")

    if args.raw_ts:
        ts = ts_raw
    else:
        ts, info = fit_timestamps(ts_raw, rad["frame_period"] / 1e3)
        print(f"  frame period {info['fitted_period_ms']:.4f} ms "
              f"(config {info['config_period_ms']:.4f}), arrival jitter p99 "
              f"{info['jitter_ms_p99']:.2f} ms")

    if chirps > 1:
        # Stock TI firmware needs >= 16 chirps per frame. Each chirp loop (all TX
        # once) is one PanoRadar beam, so split every frame into its loops and
        # give each loop its own time: frame start + k * loop period.
        loop_s = n_tx * (rad["idle_time"] + rad["ramp_end_time"]) * 1e-6
        raw = raw.reshape(n * chirps, 1, n_tx, n_rx, raw_last)
        ts = (ts[:, None] + np.arange(chirps)[None] * loop_s).ravel()
        complete = np.repeat(complete, chirps)
        print(f"  split {n} frames x {chirps} chirps -> {n * chirps} beams, "
              f"{loop_s * 1e3:.3f} ms apart within a frame")
        n, chirps = n * chirps, 1

    # --- motor angle at every beam
    motor_csv = args.motor_csv or sess / "motor.csv"
    if args.assume_rev_per_s:
        print(f"Motor: none, assuming a constant {args.assume_rev_per_s} rev/s (testing only)")
        mt = np.array([ts[0], ts[-1]])
        mang = 360.0 * args.assume_rev_per_s * (mt - ts[0]) + 1e-6
    elif not motor_csv.exists():
        raise SystemExit(f"No motor log found at {motor_csv}; pass --motor-csv, "
                         "or --assume-rev-per-s to test without one")
    else:
        mt, mang = load_motor(motor_csv, args.ts_col, args.angle_col,
                              angle_units, steps_per_rev)
        mt = mt + args.motor_time_offset
    rate = np.diff(mang) / np.diff(mt)
    if not args.assume_rev_per_s:
        print(f"Motor: {len(mt)} readings at {len(mt) / (mt[-1] - mt[0]):.0f} Hz, "
              f"speed {np.median(np.abs(rate)) / 360:.3f} rev/s")

    inside = (ts >= mt[0]) & (ts <= mt[-1])
    if not inside.any():
        raise SystemExit("Motor log and radar recording don't overlap in time.")
    lo, hi = np.flatnonzero(inside)[[0, -1]]
    raw, ts, complete = raw[lo:hi + 1], ts[lo:hi + 1], complete[lo:hi + 1]
    ang = np.interp(ts, mt, mang)

    order = np.arange(len(ts))
    if args.direction == "ccw":
        order = order[::-1]
        ts = ts[-1] + ts[0] - ts[::-1]  # keep timestamps increasing
        ang = ang[::-1]
    starts = rotation_starts(ang, args.zero_angle)
    if len(starts) < 2:
        raise SystemExit("Fewer than one full rotation found. Is the motor spinning, "
                         "and are --angle-units right?")
    if moving:
        # Moving frames borrow half a rotation of beams on each side (PanoRadar pads
        # each frame with neighbouring beams), so keep only rotations that have them.
        margin = int(np.median(np.diff(starts))) // 2
        starts = starts[(starts >= margin) & (starts + margin < len(ts))]
        if len(starts) < 2:
            raise SystemExit("Recording too short for moving data: record a few more rotations.")
    else:
        margin = 0
    per_rot = np.diff(starts)
    spread = (per_rot.max() - per_rot.min()) / np.median(per_rot)
    print(f"Rotations: {len(per_rot)} full, {np.median(per_rot):.0f} beams each "
          f"(min {per_rot.min()}, max {per_rot.max()})")
    beam_s = rad["frame_period"] / rad.get("frame_length", 1) / 1e3  # average time per beam
    med = float(np.median(per_rot))
    if abs(med - 1200) / 1200 > 0.05:
        print(f"NOTE: {med:.0f} beams/rotation; the PanoRadar dataset has 1200. Your motor ran "
              f"at {1 / (med * beam_s):.2f} rev/s; set it to {1 / (1200 * beam_s):.2f} rev/s "
              f"({60 / (1200 * beam_s):.0f} rpm) to match.")
    if spread > 0.03:
        print("WARNING: beams per rotation vary by >3%. PanoRadar assumes evenly "
              "spaced beams, so keep the motor at constant speed.")

    # --- chirp parameters actually used
    fs = rad["sample_rate"] * 1e3
    slope = rad["freq_slope"] * 1e12
    f_st = rad["frequency"] * 1e9 + slope * rad["adc_start_time"] * 1e-6
    params = {"fs": fs, "freq_slope": slope, "freq_st": f_st,
              "freq_end": f_st + slope * n_samp / fs, "device": meta["device"],
              "n_virtual_antennas": chirps * n_tx * n_rx}
    # Same chirp as PanoRadar's authors -> let PanoRadar use its own built-in
    # constants, exactly as for the sample dataset. Otherwise tell it ours.
    same_chirp = (rad["frequency"] == 77.0 and rad["sample_rate"] == 5120
                  and abs(rad["freq_slope"] - 79.951) < 1e-6 and n_samp == 256)
    print("Chirp: same as PanoRadar's, so it uses its built-in constants" if same_chirp
          else "Chirp: differs from PanoRadar's; writing radar_params.json for the patched loader")
    if params["n_virtual_antennas"] != 8:
        print(f"WARNING: {params['n_virtual_antennas']} virtual antennas; PanoRadar's "
              "vertical beamforming assumes 8 (use device AWR1843L).")

    # --- write trajectories
    prefix = args.traj_prefix or "exp" + sess.name.split("_")[-1].split("-")[0]
    args.building.mkdir(parents=True, exist_ok=True)
    R = args.rotations_per_traj
    n_traj = 0
    for k, i0 in enumerate(range(0, len(starts) - 1, R)):
        seps = starts[i0: min(i0 + R, len(starts) - 1) + 1]
        if len(seps) < 2:
            break
        # end one beam past the last rotation start (PanoRadar indexes radar_ts[radar_sep]),
        # plus the padding margin on both sides for moving data
        b0, b1 = seps[0] - margin, seps[-1] + 1 + margin
        out = args.building / f"{prefix}-{k:03d}"
        out.mkdir(parents=True, exist_ok=True)

        data = np.lib.format.open_memmap(
            out / "radar_data.npy", mode="w+", dtype=np.complex64,
            shape=(int(b1 - b0), int(chirps * n_tx), int(n_rx), int(n_samp)))
        idx = order[b0:b1]
        for c in range(0, len(idx), 4096):  # decode in chunks to save RAM
            sl = idx[c:c + 4096]
            fwd = np.sort(sl)  # read the raw file forwards
            block = iq_from_iiqq(np.asarray(raw[fwd])).reshape(
                len(sl), chirps * n_tx, n_rx, n_samp)
            if sl[0] != fwd[0]:  # ccw: put back in time-reversed order
                block = block[::-1]
            data[c:c + len(sl)] = np.conj(block) if args.conjugate else block
        data.flush()
        del data
        np.savetxt(out / "radar_ts.txt", ts[b0:b1], fmt="%.6f")

        radar_sep = (seps - b0).astype(np.int64)
        n_fr = len(radar_sep)
        # placeholders: one LiDAR "scan" and one IMU reading per rotation
        np.save(out / "lidar_data.npy", np.zeros((n_fr, 64, 16, 3), np.float32))
        rot_ts = ts[seps]
        np.savetxt(out / "lidar_ts.txt", rot_ts, fmt="%.6f")
        np.save(out / "imu_data.npy", np.zeros(n_fr, np.float32))
        np.savetxt(out / "imu_ts.txt", rot_ts, fmt="%.6f")
        np.savez(out / "data_params.npz", radar_sep=radar_sep,
                 lidar_sep=np.arange(n_fr), imu_sep=np.arange(n_fr),
                 r_radar=np.array(r_radar), transform=np.eye(2))
        if not same_chirp:
            (out / "radar_params.json").write_text(json.dumps(
                {**params, "incomplete_beams": int(np.count_nonzero(~complete[idx])),
                 "source_session": str(sess)}, indent=2))
        print(f"  wrote {out}  ({n_fr - 1} frames, {b1 - b0} beams)")
        n_traj += 1

    print(f"Done: {n_traj} trajectories in {args.building}")
    print("Image one frame with:\n  python run.py imaging --in_traj_folder "
          f"{args.building}/{prefix}-000 --frame_num 0 --out_plot_folder .")


if __name__ == "__main__":
    main()