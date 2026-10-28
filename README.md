# fireflight-radar-testing

# FireFlight radar testing

Record raw data from a TI **AWR1843Boost + DCA1000EVM** on a rotating platform,
convert it to the [PanoRadar](https://github.com/penn-waves-lab/PanoRadar)
dataset format, and image it with PanoRadar's own, unmodified code.

```
radar   ──► record.py ───────┐
                              ├──► build_traj.py ──► PanoRadar data folder ──► PanoRadar image
Arduino ──► motor_logger.py ─┘
```

| File | What it does |
|---|---|
| `config.yaml` | Radar settings + your platform's measurements (**you fill in the `platform:` part**) |
| `record.py` | Records the radar into `sessions/<name>_<date-time>/` |
| `motor_logger.py` | Reads the Arduino's motor readings and stamps each with the PC clock |
| `build_traj.py` | Combines radar + motor log into PanoRadar's 7-file format |
| `simulate_session.py` | Makes a fake recording, for testing without the radar |
| `inspect_traj.py` | Prints a data folder's format, to compare with PanoRadar's sample |

Paths below assume this repo is at `~/fireflight-radar-testing`, PanoRadar at
`~/PanoRadar`, and the PanoRadar sample dataset at `~/Downloads/design`.

> [!IMPORTANT]
> **Two things to get right for real recordings:**
> 1. **Motor speed: 1200 beams per rotation.** The radar takes a beam every
>    0.491 ms, so the platform must turn at **1.70 rev/s (102 rpm)** to match
>    the PanoRadar dataset. `motor_logger.py` shows this live.
> 2. **Rotation radius in `config.yaml`.** Measure it (see
>    [Measuring the rotation radius](#measuring-the-rotation-radius)) and set
>    `platform: r_radar_m`. `build_traj.py` won't convert real data without it.

---

## A. One-time setup

### A1. System packages

```bash
sudo add-apt-repository -y universe
sudo apt update
sudo apt install -y git curl wget build-essential python3-dev python3-pip python3-venv
```

### A2. Check the NVIDIA GPU (PanoRadar needs one)

```bash
nvidia-smi
```

It should show your GPU. If it says "command not found", install the driver
with `sudo ubuntu-drivers autoinstall`, then `sudo reboot`.

### A3. Install Miniconda (once)

```bash
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh     # press Enter / type "yes" at each prompt
source ~/.bashrc
```

### A4. Create the capture environment (for this repo)

1. Go to the repo:
   ```bash
   cd ~/fireflight-radar-testing
   ```
2. Create the environment (a folder called `.venv`; git ignores it):
   ```bash
   python3 -m venv .venv
   ```
3. Turn it on. Your prompt now starts with `(.venv)`:
   ```bash
   source .venv/bin/activate
   ```
4. Install its packages:
   ```bash
   pip install numpy pyyaml pyserial
   ```
5. Turn it off when done:
   ```bash
   deactivate
   ```

### A5. Create the PanoRadar environment

1. Download PanoRadar:
   ```bash
   cd ~ && git clone https://github.com/penn-waves-lab/PanoRadar.git
   cd ~/PanoRadar
   ```
2. Create and turn on the environment. Your prompt now starts with `(panoradar)`:
   ```bash
   conda create -y --name panoradar python=3.10
   conda activate panoradar
   ```
3. Install PanoRadar's packages. CuPy is skipped here and added in the next
   step, because their pinned version doesn't install on new CUDA:
   ```bash
   grep -v '^cupy' requirements.txt > /tmp/req.txt
   python -m pip install -r /tmp/req.txt
   ```
4. Install CuPy and the imaging extras:
   ```bash
   python -m pip install "cupy-cuda12x[ctk]" opencv-python scikit-image scipy matplotlib tqdm
   ```
5. Check it works. It should print your GPU's name:
   ```bash
   python -c "import cupy as cp; print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())"
   ```

> [!NOTE]
> Always use `python -m pip`, not `pip`, inside the PanoRadar environment, and
> run `deactivate` first if `(.venv)` is showing. Mixing environments is what
> causes `No module named cupy`. detectron2 isn't needed; it's only for
> PanoRadar's ML models.

### A6. Link the PanoRadar sample data

```bash
mkdir -p ~/PanoRadar/data/raw
ln -s ~/Downloads/design/* ~/PanoRadar/data/raw/
ls ~/PanoRadar/data/raw          # should list design_static (and design_moving)
```

### How to make a PanoRadar image

PanoRadar's `run.py imaging` has a typo (`out_plot_path`) that makes it crash,
so call the same function directly:

```bash
cd ~/PanoRadar && conda activate panoradar
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('<DATA FOLDER>', <FRAME>, '.')"
```

This saves `imaging_result.png` in `~/PanoRadar`. One frame = one full
rotation. Each run overwrites the last image, so rename any you want to keep.

---

## B. Test everything with fake data (no radar)

**B1. Image the authors' sample** (proves PanoRadar works):

```bash
deactivate 2>/dev/null; cd ~/PanoRadar && conda activate panoradar
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/design_static/exp20230524-000', 5, '.')"
mv imaging_result.png imaging_sample.png
```

✅ `imaging_sample.png` shows a map of a room.

**B2. Make a fake recording and convert it** (proves your converter works):

```bash
conda deactivate; cd ~/fireflight-radar-testing && source .venv/bin/activate
python simulate_session.py --out sessions/sim_20261028-000000
python build_traj.py sessions/sim_20261028-000000 --building ~/PanoRadar/data/raw/sim_static --r-radar 0.077
deactivate
```

✅ Prints `Rotations: 6 full, 1198 beams each`. (`--r-radar 0.077` is the fake
platform's radius, so you don't need to edit `config.yaml` for this test.)

**B3. Image the fake recording:**

```bash
cd ~/PanoRadar && conda activate panoradar
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/sim_static/exp20261028-000', 1, '.')"
```

✅ `imaging_result.png` shows 4 dots at (3, 0), (0, −4), (−2, 2) and
(5, 5) m, plus faint rings.

**B4. Compare formats:**

```bash
python ~/fireflight-radar-testing/inspect_traj.py data/raw/design_static/exp20230524-000 data/raw/sim_static/exp20261028-000
```

✅ Both show `radar_data.npy shape (…, 2, 4, 256) complex64` and about 1200
beams/rotation.

---

## C. Real radar (already flashed)

### C1. One-time computer setup

1. **Board switches.** On the AWR1843Boost, set `SOP2:0` = `001` (functional
   mode). On the DCA1000, set `SW2.5` = `SW_CONFIG`, `SW2.6` = `USER_SW1`, and
   the power switch to `RADAR_5V_IN`.
2. **Cables.** Connect the LVDS ribbon between the boards, the radar's
   micro-USB to the PC, and the DCA1000's Ethernet port to the PC.
3. **Find the Ethernet connection's name** (the one the DCA1000 is plugged into):
   ```bash
   nmcli connection show
   ```
4. **Give it a fixed address**, `192.168.33.30` (the PC's address; the DCA1000
   is `192.168.33.180`). Replace `Wired connection 1` with the name from step 3:
   ```bash
   sudo nmcli connection modify "Wired connection 1" ipv4.method manual ipv4.addresses 192.168.33.30/24 ipv4.gateway ""
   sudo nmcli connection up "Wired connection 1"
   ```
5. **Bigger network buffer**, so radar packets aren't dropped:
   ```bash
   echo "net.core.rmem_max=6291456" | sudo tee /etc/sysctl.d/99-xwr.conf
   sudo sysctl --system
   ```
6. **USB permissions** for the radar and Arduino. Log out and back in afterwards:
   ```bash
   printf 'KERNEL=="ttyACM[0-9]*", MODE="0666"\nKERNEL=="ttyUSB[0-9]*", MODE="0666"\n' | sudo tee /etc/udev/rules.d/99-xwr.rules
   sudo udevadm control --reload-rules
   sudo usermod -aG dialout $USER
   ```
7. **Install xwr** (the radar library) into the capture environment:
   ```bash
   cd ~ && git clone https://github.com/RadarML/xwr.git
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   pip install -e ~/xwr
   ```
8. **Check the radar settings are valid.** Every value printed should be `True`:
   ```bash
   python -c "import yaml, xwr; c = yaml.safe_load(open('config.yaml')); print([r.passed for r in xwr.check_config(xwr.XWRConfig(**c['radar']), xwr.DCAConfig(**c['capture']), log=False)])"
   deactivate
   ```
9. **Live radar check.** Power the radar on and wait about 30 s, then run xwr's
   demo. A plot that changes when you wave your hand means everything works:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
   cd ~/xwr/demo && uv run demo.py --device AWR1843L --rsp AWR1642Boost
   ```

### C2. Fill in `config.yaml` → `platform:`

```yaml
platform:
  r_radar_m: 0.077           # your measured radius, in metres (77 mm = 0.077)
  motor_ticks_per_rev: 3200  # Arduino counts per full turn; null if it prints degrees
```

> [!IMPORTANT]
> #### Measuring the rotation radius
> `r_radar_m` is the **horizontal distance from the rotation axis to the
> radar's antennas**. PanoRadar uses it to focus the image, so a wrong value
> makes it blurry.
>
> 1. Stop the motor and power off the radar.
> 2. Find the **rotation axis**: the centre of the motor shaft.
> 3. Find the **antenna face**: the front surface of the AWR1843Boost where
>    the small rectangular antenna patches are printed.
> 4. With calipers or a steel ruler, measure **straight out from the shaft
>    centre to the antenna face**, level with the antennas, in millimetres.
>    If you can't reach the shaft centre, measure from the **edge** of the
>    shaft to the antenna face, then **add half the shaft's diameter**.
> 5. Convert to metres (77 mm → `0.077`) and put it in `config.yaml`.
>
> Aim for ±1–2 mm; the radar's wavelength is only 3.9 mm. PanoRadar's own rig
> was 0.077 m. If images look slightly smeared, re-measure, or try ±3 mm.

To find `motor_ticks_per_rev`, ask the motor-code author, or mark the platform,
turn it exactly one full turn by hand while `motor_logger.py` runs (C3), and
see how much the count changed.

### C3. Arduino motor logger

1. The Arduino sketch must print **one reading per line**, e.g.
   `Serial.println(ticks);`, with `Serial.begin(115200)`.
2. Upload the sketch, then **close the Arduino IDE's Serial Monitor**. Only one
   program can use the USB port.
3. Find the Arduino's port. The radar uses `ttyACM0` and `ttyACM1`, so on the
   radar PC the Arduino is usually `ttyACM2`:
   ```bash
   ls /dev/ttyACM* /dev/ttyUSB*
   ```
4. Start the motor, then the logger:
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python motor_logger.py --port /dev/ttyACM2 --out motor_test.csv
   ```
5. Every 2 s it prints the speed and the beams per rotation it gives. **Adjust
   the motor until it reads about 1200 beams/rotation (1.70 rev/s, 102 rpm).**
   Stop with Ctrl-C.

> [!WARNING]
> **Arduino on a different computer from the radar?** The two clocks must
> agree. On **both** computers run `sudo apt install chrony`, then check that
> `chronyc tracking` shows a "System time" offset of a few milliseconds or
> less. Copy the motor CSV to the radar computer before converting.

### C4. Every recording

1. Mount the radar with its antenna row **vertical**, facing outward from the
   axis.
2. Start the motor and let it reach **1200 beams/rotation** (C3 step 5).
3. **Terminal 1, motor log** (on the computer the Arduino is plugged into):
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python motor_logger.py --port /dev/ttyACM2 --out motor_lab1.csv
   ```
4. **Terminal 2, radar** (on the radar computer):
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python record.py --name lab1 --duration 30
   ```
   ✅ It prints about **2040 beams/s** every 2 s, with incomplete beams near 0.
5. When the radar finishes, stop the motor logger with **Ctrl-C**.
6. **Convert** (radar computer). Use the session folder name `record.py` printed:
   ```bash
   python build_traj.py sessions/lab1_<date-time> --motor-csv motor_lab1.csv \
       --building ~/PanoRadar/data/raw/lab1_static
   deactivate
   ```
   ✅ It prints your radius, about **1200 beams each**, and no `NOTE` or
   `WARNING`. A `NOTE` means the motor speed was off; a `WARNING` means it
   wasn't steady. Keep `static` in the `--building` name: it tells PanoRadar the
   platform isn't driving around.
7. **Image it:**
   ```bash
   cd ~/PanoRadar && conda activate panoradar
   python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/lab1_static/exp<date>-000', 0, '.')"
   ```

Recordings are about 17 MB/s (≈500 MB per 30 s). `sessions/` is git-ignored.

### If something's off

| Symptom | Fix |
|---|---|
| `Device or resource busy: /dev/ttyACM0` | Radar still booting; wait 10–30 s |
| Radar hangs after `sensorStart` | Set `frequency: 76.0` in `config.yaml` |
| Capture times out | Check `ip -br addr` shows `192.168.33.30`; reseat the LVDS ribbon |
| Many incomplete beams | Check `sysctl net.core.rmem_max` is 6291456; use a dedicated Ethernet port |
| `motor_logger.py` can't open the port | Close the Arduino Serial Monitor; check `--port` |
| `Rotation radius not set` | Fill in `platform: r_radar_m` in `config.yaml` (C2) |
| `Fewer than one full rotation` | Motor not spinning, or wrong `motor_ticks_per_rev` |
| Image mirrored | Add `--direction ccw` to `build_traj.py` |
| Targets smeared or at the wrong distance | Re-measure the radius; or add `--conjugate` to `build_traj.py` |
| No motor log yet | Add `--assume-rev-per-s 1.7` (sharp targets, but the image is rotated by an unknown angle) |