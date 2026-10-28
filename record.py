"""Record raw radar frames (and optionally motor angles) into sessions/<name>_<time>/.

    python record.py --name lab1 --duration 30
    python record.py --name lab1 --motor-cmd "python3 motor.py"   # motor on this PC
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import signal
import threading
import time
from pathlib import Path

import yaml

import xwr


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--name", default="session", help="label for the session folder")
    p.add_argument("--out", default="sessions", help="parent folder for sessions")
    p.add_argument("--motor-cmd", default=None,
                   help="command that runs your motor-angle script and prints angles")
    p.add_argument("--duration", type=float, default=None, help="seconds to record")
    p.add_argument("--warmup", type=float, default=1.0,
                   help="seconds to let the motor reach speed before recording radar")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(name)-14s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("record")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sess = Path(args.out) / f"{args.name}_{time.strftime('%Y%m%d-%H%M%S')}"
    sess.mkdir(parents=True, exist_ok=False)
    shutil.copy(args.config, sess / "config.yaml")
    log.info(f"Session folder: {sess}")

    # strict=True refuses configs that violate xwr's known hardware constraints
    awr = xwr.XWRSystem(radar=cfg["radar"], capture=cfg["capture"], strict=True)
    rc = awr.config
    meta = {
        "device": rc.device_name,
        "raw_shape": list(rc.raw_shape),          # (chirps, tx, rx, 2*samples) int16
        "shape": list(rc.shape),                  # (chirps, tx, rx, samples) complex
        "frame_bytes": rc.frame_size,
        "frame_period_ms": rc.frame_period,
        "radar": {k: getattr(rc, k) for k in (
            "frequency", "freq_slope", "sample_rate", "adc_samples",
            "adc_start_time", "ramp_end_time", "idle_time", "tx_start_time",
            "frame_length", "frame_period")},
        "start_time": time.time(),
        "motor_cmd": args.motor_cmd,
    }
    (sess / "session.json").write_text(json.dumps(meta, indent=2))

    motor = None
    if args.motor_cmd:
        from motor_source import SubprocessMotorSource  # only needed with --motor-cmd
        motor = SubprocessMotorSource(args.motor_cmd, sess / "motor.csv")
        motor.start()
        log.info(f"Started motor script: {args.motor_cmd}")
        time.sleep(args.warmup)
        if not motor.alive():
            log.error("Motor script exited right away; check the command. Output is in motor.csv")
    else:
        log.info("Radar only (no --motor-cmd). Merge the motor log later with build_traj.py --motor-csv")

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    if args.duration:
        threading.Timer(args.duration, stop.set).start()

    q = awr.qstream(numpy=False)  # worker thread fills the queue; we only write bytes
    n = incomplete = 0
    t_report = time.time()
    n_report = 0
    with open(sess / "radar_raw.bin", "wb") as fbin, open(sess / "radar_ts.csv", "w") as fts:
        fts.write("timestamp,complete\n")
        try:
            while not stop.is_set():
                try:
                    frame = q.get(timeout=0.5)
                except Exception:
                    continue
                if frame is None:  # capture card timed out
                    log.error("Radar stream ended (capture card timeout).")
                    break
                fbin.write(frame.data)
                fts.write(f"{frame.timestamp:.6f},{int(frame.complete)}\n")
                n += 1
                incomplete += not frame.complete

                now = time.time()
                if now - t_report >= 2.0:
                    fps = (n - n_report) / (now - t_report)
                    msg = f"{n} beams  {fps:7.1f} beams/s  incomplete={incomplete}"
                    if motor and motor.last:
                        msg += f"  motor angle={motor.last[1]:8.2f}  rows={motor.count}"
                    log.info(msg)
                    t_report, n_report = now, n
        finally:
            awr.stop()
            if motor:
                motor.stop()
            # drain anything still queued so the last beams are kept
            while True:
                try:
                    frame = q.get_nowait()
                except Exception:
                    break
                if frame is None:
                    break
                fbin.write(frame.data)
                fts.write(f"{frame.timestamp:.6f},{int(frame.complete)}\n")
                n += 1
                incomplete += not frame.complete

    meta.update(n_frames=n, incomplete_frames=incomplete, end_time=time.time())
    (sess / "session.json").write_text(json.dumps(meta, indent=2))
    log.info(f"Saved {n} beams ({incomplete} incomplete) to {sess}")
    if incomplete > 0.01 * max(n, 1):
        log.warning("More than 1% incomplete frames: check rmem_max and use a dedicated NIC.")


if __name__ == "__main__":
    main()