"""Make a fake session with point targets at TARGETS, to test without hardware.

    python simulate_session.py --out sessions/sim_20261028-000000
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

C = 2.99792e8
TARGETS = [(3.0, 0.0), (0.0, -4.0), (-2.0, 2.0), (5.0, 5.0)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--out", type=Path, default=Path("sessions/sim_20261028-000000"))
    p.add_argument("--seconds", type=float, default=4.5)
    p.add_argument("--rev-per-s", type=float, default=1.7)
    p.add_argument("--r-radar", type=float, default=None, help="default: config, else 0.077")
    p.add_argument("--ccw", action="store_true", help="spin counter-clockwise (seen from above)")
    p.add_argument("--walk-speed", type=float, default=0.0,
                   help="carry the platform along +x at this speed (m/s), for a moving test")
    a = p.parse_args()

    cfg = yaml.safe_load(open(a.config))
    rad, plat = cfg["radar"], cfg.get("platform") or {}
    if a.r_radar is None:
        a.r_radar = plat.get("r_radar_m") or 0.077
    ticks = plat.get("motor_ticks_per_rev")  # log like the real Arduino would
    n_samp, fs = rad["adc_samples"], rad["sample_rate"] * 1e3
    slope = rad["freq_slope"] * 1e12
    f_st = rad["frequency"] * 1e9 + slope * rad["adc_start_time"] * 1e-6
    f_n = f_st + slope * np.arange(n_samp) / fs
    period = rad["frame_period"] / 1e3
    chirps, n_tx, n_rx = rad["frame_length"], 2, 4

    # one beam per chirp loop; with frame_length > 1 (stock firmware) a frame
    # holds frame_length beams, loop_s apart, then the next frame starts
    loop_s = n_tx * (rad["idle_time"] + rad["ramp_end_time"]) * 1e-6
    n_frames = int(a.seconds / period)
    t0 = time.time()
    frame_t = t0 + np.arange(n_frames) * period
    t = (frame_t[:, None] + np.arange(chirps)[None] * loop_s).ravel()
    n = len(t)
    theta = 360.0 * a.rev_per_s * (t - t0) + 37.0     # motor angle, deg, increasing
    psi = (1 if a.ccw else -1) * np.deg2rad(theta)    # cw: heading decreases
    centre = np.stack([a.walk_speed * (t - t0), np.zeros(n)], 1)   # platform carried along +x
    pos = centre + a.r_radar * np.stack([np.cos(psi), np.sin(psi)], 1)

    sig = np.zeros((n, n_samp), np.complex128)
    for tx, ty in TARGETS:
        v = np.array([tx, ty]) - pos
        d = np.linalg.norm(v, axis=1)
        off = np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - psi)))
        gain = np.where(np.abs(off) < np.pi / 2, np.cos(off) ** 4, 0) / d
        sig += gain[:, None] * np.exp(1j * 4 * np.pi * d[:, None] * f_n[None] / C)
    rng = np.random.default_rng(0)
    sig = 3000 * sig + 20 * (rng.standard_normal(sig.shape) + 1j * rng.standard_normal(sig.shape))

    # every virtual antenna sees the same signal at elevation 0
    iq = np.broadcast_to(sig.reshape(n_frames, chirps, 1, 1, n_samp),
                         (n_frames, chirps, n_tx, n_rx, n_samp))
    iiqq = np.zeros((n_frames, chirps, n_tx, n_rx, 2 * n_samp), np.int16)
    iiqq[..., 0::4] = np.round(iq.imag[..., 0::2])
    iiqq[..., 1::4] = np.round(iq.imag[..., 1::2])
    iiqq[..., 2::4] = np.round(iq.real[..., 0::2])
    iiqq[..., 3::4] = np.round(iq.real[..., 1::2])

    a.out.mkdir(parents=True, exist_ok=True)
    iiqq.tofile(a.out / "radar_raw.bin")
    arrival = frame_t + 0.002 + rng.exponential(0.0005, n_frames)  # network delay + jitter
    arrival = np.maximum.accumulate(arrival)          # frames arrive in order
    np.savetxt(a.out / "radar_ts.csv", np.c_[arrival, np.ones(n_frames)], delimiter=",",
               header="timestamp,complete", comments="", fmt=["%.6f", "%d"])
    mt = np.arange(t0, t[-1], 0.01)                    # 100 Hz motor log
    turns = a.rev_per_s * (mt - t0) + 37.0 / 360
    mval = np.round(turns * ticks) if ticks else (360.0 * turns) % 360  # ticks, or wrapped degrees
    np.savetxt(a.out / "motor.csv", np.c_[mt, mval], delimiter=",",
               header="timestamp,value", comments="", fmt="%.6f")
    meta = {"device": "AWR1843L", "raw_shape": [chirps, n_tx, n_rx, 2 * n_samp],
            "shape": [chirps, n_tx, n_rx, n_samp],
            "frame_bytes": chirps * n_tx * n_rx * n_samp * 4,
            "frame_period_ms": rad["frame_period"], "radar": rad,
            "n_frames": n_frames, "incomplete_frames": 0, "simulated_targets": TARGETS}
    (a.out / "session.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {n} simulated beams ({n_frames} frames) to {a.out}")


if __name__ == "__main__":
    main()