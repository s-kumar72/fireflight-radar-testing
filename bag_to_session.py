"""Convert a ROS 2 bag of /xwr/iq into a session folder for build_traj.py.

    python bag_to_session.py ~/bags/lab1 --config panoradar.yaml --motor-csv motor.csv
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import yaml
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

IQ_MSG = """std_msgs/Header header
std_msgs/Int16MultiArray iq
bool complete
"""
INFO_MSG = """std_msgs/Header header
float32 frequency
float32 idle_time
float32 adc_start_time
float32 ramp_end_time
float32 tx_start_time
float32 freq_slope
uint16 adc_samples
uint16 sample_rate
uint16 frame_length
"""
RADAR_KEYS = ("frequency", "idle_time", "adc_start_time", "ramp_end_time",
              "tx_start_time", "freq_slope", "adc_samples", "sample_rate",
              "frame_length")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bag", type=Path, help="bag folder from ros2 bag record")
    p.add_argument("--config", type=Path, default=None,
                   help="radar config yaml used by the ROS node (needed for "
                        "device name and frame_period)")
    p.add_argument("--motor-csv", type=Path, default=None,
                   help="motor log from log_motor.py; copied into the session")
    p.add_argument("--topic", default="/xwr/iq")
    p.add_argument("--out", type=Path, default=None,
                   help="session folder (default: sessions/<bag name>_<date>)")
    args = p.parse_args()

    ts_store = get_typestore(Stores.ROS2_HUMBLE)
    ts_store.register(get_types_from_msg(IQ_MSG, "xwr_msgs/msg/IQ"))
    ts_store.register(get_types_from_msg(INFO_MSG, "xwr_msgs/msg/ChirpInfo"))

    cfg = yaml.safe_load(args.config.read_text()) if args.config else None

    with Reader(args.bag) as reader:
        iq_conns = [c for c in reader.connections if c.topic == args.topic]
        if not iq_conns:
            raise SystemExit(f"No {args.topic} in bag; topics: "
                             f"{sorted({c.topic for c in reader.connections})}")
        info_conns = [c for c in reader.connections if c.topic == "/xwr/info"]

        radar = {}
        # careful: messages(connections=[]) would return EVERY message
        for conn, _, raw in (reader.messages(connections=info_conns) if info_conns else []):
            info = ts_store.deserialize_cdr(raw, conn.msgtype)
            radar = {k: getattr(info, k) for k in RADAR_KEYS}
            radar = {k: (round(float(v), 4) if isinstance(v, float) else int(v))
                     for k, v in radar.items()}
            break
        if cfg:  # the yaml is exact (floats in the msg are float32)
            radar.update({k: cfg["radar"][k] for k in RADAR_KEYS})
            radar["frame_period"] = cfg["radar"]["frame_period"]
            device = cfg["radar"]["device"]
        else:
            raise SystemExit("Pass --config (the yaml the ROS node used): the bag "
                             "does not record the device name or frame_period.")
        radar["frame_period"] = float(radar["frame_period"])

        first = next(iter(reader.messages(connections=iq_conns)))
        stamp0 = first[1]
        name = args.bag.name
        stamp_str = time.strftime("%Y%m%d-%H%M%S", time.localtime(stamp0 / 1e9))
        out = args.out or Path("sessions") / f"{name}_{stamp_str}"
        out.mkdir(parents=True, exist_ok=False)

        n = incomplete = 0
        raw_shape = None
        with open(out / "radar_raw.bin", "wb") as fbin, \
                open(out / "radar_ts.csv", "w") as fts:
            fts.write("timestamp,complete\n")
            for conn, _, raw in reader.messages(connections=iq_conns):
                msg = ts_store.deserialize_cdr(raw, conn.msgtype)
                if raw_shape is None:
                    raw_shape = [int(d.size) for d in msg.iq.layout.dim]
                data = np.asarray(msg.iq.data, dtype=np.int16)
                fbin.write(data.tobytes())
                t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                fts.write(f"{t:.6f},{int(msg.complete)}\n")
                n += 1
                incomplete += not msg.complete

    shape = raw_shape[:-1] + [raw_shape[-1] // 2]
    meta = {"device": device, "raw_shape": raw_shape, "shape": shape,
            "frame_bytes": int(np.prod(raw_shape)) * 2,
            "frame_period_ms": radar["frame_period"], "radar": radar,
            "n_frames": n, "incomplete_frames": incomplete,
            "source_bag": str(args.bag)}
    (out / "session.json").write_text(json.dumps(meta, indent=2))
    if args.motor_csv:
        shutil.copy(args.motor_csv, out / "motor.csv")
    print(f"Wrote {n} frames ({incomplete} incomplete), shape {raw_shape} to {out}")
    if not args.motor_csv:
        print("No --motor-csv given: pass the motor log to build_traj.py --motor-csv")


if __name__ == "__main__":
    main()