"""Check a recording from a radar that is NOT spinning: print the distances of the
strongest reflections, and save a range plot.

    python check_range.py sessions/still1_<date-time>

Put a strong reflector (a metal plate, a laptop lid, a wall) straight in front
of the radar at a known distance, e.g. 2.00 m. The top peak should be within a
few cm of it. If it matches the "if I/Q is flipped" column instead, use
build_traj.py --conjugate for this radar.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from build_traj import iq_from_iiqq

C = 2.99792e8


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path)
    p.add_argument("--beams", type=int, default=2000, help="how many radar frames to average")
    p.add_argument("--peaks", type=int, default=5)
    a = p.parse_args()

    meta = json.loads((a.session / "session.json").read_text())
    rad = meta["radar"]
    chirps, n_tx, n_rx, raw_last = meta["raw_shape"]
    n_samp = meta["shape"][-1]
    frame_int16 = meta["frame_bytes"] // 2

    raw = np.memmap(a.session / "radar_raw.bin", dtype=np.int16, mode="r")
    n = raw.size // frame_int16
    if n == 0:
        raise SystemExit("No radar frames in this session.")
    take = min(a.beams, n)
    mid = (n - take) // 2  # skip start-up beams
    block = np.asarray(raw[mid * frame_int16:(mid + take) * frame_int16])
    iq = iq_from_iiqq(block.reshape(take, chirps, n_tx, n_rx, raw_last)).reshape(take, -1, n_samp)

    # range FFT of every beam and antenna, averaged: one range profile for the scene
    spec = np.abs(np.fft.fft(iq * np.hanning(n_samp), axis=-1))   # (beams, antennas, range)
    profile = spec.mean(axis=(0, 1))
    db = 20 * np.log10(profile + 1e-9)

    fs = rad["sample_rate"] * 1e3
    slope = rad["freq_slope"] * 1e12
    bin_m = C * fs / (2 * slope * n_samp)
    ranges = np.arange(n_samp) * bin_m

    # peaks: local maxima, ignoring the first few bins (radar's own leakage)
    first = max(3, int(0.15 / bin_m))
    is_peak = (db[1:-1] > db[:-2]) & (db[1:-1] >= db[2:])
    idx = np.flatnonzero(is_peak) + 1
    idx = idx[idx >= first]
    idx = idx[np.argsort(db[idx])[::-1]][:a.peaks]

    print(f"{take} frames ({take * chirps} beams) from {a.session.name}; {bin_m * 100:.2f} cm per bin, "
          f"scale 0 to {ranges[-1]:.2f} m")
    print("strongest reflections (compare with your tape measure):")
    print("   distance    power   distance if I/Q is flipped")
    for i in idx:
        print(f"  {ranges[i]:6.2f} m  {db[i] - db.max():6.1f} dB   {ranges[(n_samp - i) % n_samp]:6.2f} m")
    print("If your reflector matches the LAST column instead of the first, the I/Q sign is "
          "flipped: add --conjugate to every build_traj.py run for this radar.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 3.5))
        plt.plot(ranges, db - db.max())
        for i in idx:
            plt.annotate(f"{ranges[i]:.2f} m", (ranges[i], db[i] - db.max()),
                         textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)
        plt.xlabel("distance (m)")
        plt.ylabel("power (dB)")
        plt.title(f"Range profile: {a.session.name}")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out = a.session / "range_profile.png"
        plt.savefig(out, dpi=120)
        print(f"plot saved: {out}")
    except ImportError:
        print("(pip install matplotlib for a plot)")


if __name__ == "__main__":
    main()