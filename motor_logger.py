"""Read motor readings (angle or ticks) from an Arduino over USB serial and stamp
each line with this computer's clock (time.time()).

The Arduino sketch just prints one reading per line, e.g. Serial.println(ticks);
Close the Arduino IDE's Serial Monitor first (only one program can use the port).

    python motor_logger.py --port /dev/ttyACM2 --out motor.csv

Every 2 s it prints the motor speed and the beams per rotation it gives
(target: 1200), using motor_ticks_per_rev and frame_period from config.yaml.
"""

import argparse
import signal
import sys
import time

import serial  # pip install pyserial
import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", required=True, help="Arduino port, e.g. /dev/ttyACM2 or /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200, help="must match Serial.begin() in the sketch")
    p.add_argument("--out", default=None, help="CSV file to write; omit to print to stdout")
    p.add_argument("--duration", type=float, default=None, help="seconds to log")
    p.add_argument("--config", default="config.yaml",
                   help="read motor_ticks_per_rev and frame_period from here for the speed readout")
    a = p.parse_args()

    ticks_per_rev, beam_s = None, 0.000491
    try:
        cfg = yaml.safe_load(open(a.config))
        ticks_per_rev = (cfg.get("platform") or {}).get("motor_ticks_per_rev")
        beam_s = cfg["radar"]["frame_period"] / cfg["radar"].get("frame_length", 1) / 1e3  # average time per beam
    except (OSError, KeyError, TypeError):
        pass
    units_per_rev = ticks_per_rev or 360.0  # degrees if no tick count is set

    running = True

    def stop(*_):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    ser = serial.serial_for_url(a.port, a.baud, timeout=0.5)
    ser.reset_input_buffer()  # drop anything buffered before we started
    out = open(a.out, "w") if a.out else sys.stdout
    out.write("timestamp,value\n")
    t_end = time.time() + a.duration if a.duration else None
    n = 0
    last_report = None  # (time, total turned) at the previous speed readout
    prev, total = None, 0.0  # running total, so wrapped 0..360 degrees still count up
    try:
        while running and not (t_end and time.time() > t_end):
            line = ser.readline()
            t = time.time()  # stamp on arrival, before parsing
            if not line:
                continue
            text = line.decode(errors="ignore").strip()
            try:
                value = float(text.split(",")[-1])  # last number on the line
            except ValueError:
                continue  # skip debug text such as "motor ready"
            out.write(f"{t:.6f},{value}\n")
            out.flush()
            n += 1
            if prev is not None:
                step = value - prev
                if ticks_per_rev is None:  # degrees may wrap at 360
                    step = (step + 180) % 360 - 180
                total += step
            prev = value
            if last_report is None:
                last_report = (t, total)
            elif t - last_report[0] >= 2.0:
                rev_s = abs(total - last_report[1]) / units_per_rev / (t - last_report[0])
                msg = f"{n} readings | {rev_s:.2f} rev/s ({rev_s * 60:.0f} rpm)"
                if rev_s > 0:
                    msg += f" -> {1 / (rev_s * beam_s):.0f} beams/rotation (target 1200)"
                if ticks_per_rev is None:
                    msg += " [assumes degrees; set motor_ticks_per_rev in config.yaml]"
                print(msg, file=sys.stderr, flush=True)
                last_report = (t, total)
    finally:
        ser.close()
        if a.out:
            out.close()
            print(f"Saved {n} readings to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()