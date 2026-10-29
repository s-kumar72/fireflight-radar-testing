# Testing the radar without spinning it

This checks everything except the rotation: the radar, capture card, network,
`record.py`, that distances come out right, whether the I/Q sign needs
`--conjugate`, and that the data is in a format PanoRadar accepts.

The motor and Arduino are **not** needed.

> [!NOTE]
> **Before you start**, finish the README's one-time setup: **A4** (the
> `.venv`), **A5** (PanoRadar's conda environment), **A6** (the sample data
> link), and **C1** steps 1–8 (switches, cables, network, buffer, USB
> permissions, xwr). The xwr demo in C1 step 9 should show a live plot.

---

## Part 1: Does it measure distance correctly?

### 1. Set up a target

- Point the radar at **one strong reflector**: a metal plate, a laptop lid, or
  a flat wall.
- Put it **2.00 m** straight ahead, measured from the radar's **antenna face**
  (the front of the board where the antenna patches are).
- Keep other objects farther away. The radar's scale only goes to **9.6 m**.

### 2. Record 10 seconds

```bash
cd ~/fireflight-radar-testing
source .venv/bin/activate
pip install matplotlib        # first time only, for the plot
python record.py --name still1 --duration 10
```

✅ Every 2 s it prints about **2040 beams/s**, with **incomplete** beams near 0.

At the end it prints the session folder it saved, e.g.
`sessions/still1_20261029-101500`. Use that name below.

### 3. Check the distances

```bash
python check_range.py sessions/still1_20261029-101500
```

It prints the strongest reflections and saves `range_profile.png` in the
session folder:

```
strongest reflections (compare with your tape measure):
   distance    power   distance if I/Q is flipped
    2.01 m     0.0 dB     7.55 m
    ...
```

✅ The **top line's first column** matches your tape measure within a few cm.

> [!IMPORTANT]
> If your tape measure matches the **last column** instead, the radar's I/Q
> sign is flipped. Add `--conjugate` to **every** `build_traj.py` command for
> this radar from now on.

### 4. Move the target and repeat

Move the reflector to **3.00 m**, then repeat steps 2–3 with
`--name still2`.

✅ The top peak moves by about **1 m**.

---

## Part 2: Will PanoRadar accept the data?

This pretends the still recording was spinning at 1.7 rev/s (102 rpm), so it
gets cut into "rotations" of 1200 beams, like the real thing.

### 5. Record 30 seconds

```bash
python record.py --name still3 --duration 30
```

### 6. Convert it to PanoRadar's format

```bash
python build_traj.py sessions/still3_<date-time> --assume-rev-per-s 1.7 --r-radar 0.077 \
    --building ~/PanoRadar/data/raw/still_static
```

Add `--conjugate` if step 3 said the sign is flipped.

✅ The printout shows:
- `beam period 0.4910 ms` and `0 incomplete`
- `Rotations: ... 1198 beams each` (about 1200)
- `Done: 1 trajectories` (or more)

### 7. Compare with the authors' sample

```bash
python inspect_traj.py ~/PanoRadar/data/raw/design_static/exp20230524-000 \
    ~/PanoRadar/data/raw/still_static/exp<date>-000
deactivate
```

✅ Both folders show:
- `radar_data.npy shape (…, 2, 4, 256) dtype complex64`
- `beams/rotation median` about **1200**
- `synthetic aperture 299 beams`

### 8. Run PanoRadar's imaging on it

```bash
cd ~/PanoRadar
conda activate panoradar
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/still_static/exp<date>-000', 1, '.')"
```

✅ It finishes **without errors** and saves `imaging_result.png`.

The image will be **nearly empty, and that's expected**. PanoRadar subtracts
whatever stays the same across all beams; for a radar that doesn't turn,
that's everything. This step only proves PanoRadar loads and processes your
files.

---

## If something's off

| Symptom | Fix |
|---|---|
| `Device or resource busy: /dev/ttyACM0` | Radar still booting; wait 10–30 s after power-on |
| Hangs after `sensorStart` | Set `frequency: 76.0` in `config.yaml` |
| Capture times out | Check `ip -br addr` shows `192.168.33.30`; reseat the LVDS ribbon |
| Many incomplete beams | Check `sysctl net.core.rmem_max` is 6291456; use a dedicated Ethernet port |
| Top peak is far from your tape measure in both columns | Something closer or stronger is in view; clear the area or use a bigger reflector |
| `No module named matplotlib` | `pip install matplotlib` in the `.venv` (the distances still print without it) |
| `No module named cupy` in step 8 | `deactivate`, then `conda activate panoradar`; see README A5 |

## What this doesn't test

The **rotation**: the motor, the Arduino's tick counts, and how they line up
with the radar. That was tested with simulated data, and the first spinning
recording is the final check. Look for about 1200 beams per rotation and no
`NOTE` or `WARNING` from `build_traj.py`.