# FireFlight radar testing

Records raw data from a TI **AWR1843Boost + DCA1000EVM** on a rotating platform,
converts it to the [PanoRadar](https://github.com/penn-waves-lab/PanoRadar)
dataset format, and images it with PanoRadar's code (does not implement full ML pipeline).

```
radar   ──► record.py ───────┐
                              ├──► build_traj.py ──► PanoRadar data folder ──► PanoRadar image
Arduino ──► motor_logger.py ─┘
```

| File | What it does |
|---|---|
| `config.yaml` | Radar settings & platform measurements, mostly the same as PanoRadar but need to fill in `platform:` with our radar measurements |
| `record.py` | Records the radar data into `sessions/<name>_<date-time>/` |
| `motor_logger.py` | Reads the Arduino's motor readings and stamps each with the PC clock |
| `build_traj.py` | Combines radar & motor log into PanoRadar's 7-file format |
| `simulate_session.py` | Makes a fake recording, for testing without the radar |
| `inspect_traj.py` | Prints a data folder's format, to compare with PanoRadar's sample |
| `bag_to_session.py` | Turns a ROS 2 bag recording into a session folder (C6) |

Paths below assume this repo is at `~/fireflight-radar-testing`, PanoRadar at
`~/PanoRadar`, and the PanoRadar sample dataset at `~/Downloads/[name]`.

> [!IMPORTANT]
> **Edit using real radar:**
> 1. **Motor speed: 1200 beams per rotation.** PR takes a beam every 0.491ms, so platform needs to turn at 1.70 rev/s (102rpm) to match the dataset (live rpm visible in motor_logger.py).
> 2. **Rotation radius in `config.yaml`.** Measure it (see
>    [Measuring the rotation radius](#measuring-the-rotation-radius)) and set
>    `platform: r_radar_m`.

---

## A. One-time setup

### A1. System packages

```bash
sudo add-apt-repository -y universe
sudo apt update
sudo apt install -y git curl wget build-essential python3-dev python3-pip python3-venv
```

### A2. Check the NVIDIA GPU 

```bash
nvidia-smi
```
If cmd not found, install the driver with `sudo ubuntu-drivers autoinstall` then reboot.

### A3. Install Miniconda

```bash
cd ~/Downloads
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh    
source ~/.bashrc
```

### A4. Create the capture environment (for this repo)

1. Go to the repo:
   ```bash
   cd ~/fireflight-radar-testing
   ```
2. Create the environment (git ignored):
   ```bash
   python3 -m venv .venv
   ```
3. Turn it on: 
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
2. Create and turn on the environment:
   ```bash
   conda create -y --name panoradar python=3.10
   conda activate panoradar
   ```
3. Install PanoRadar's packages (CuPy skipped in this step because the PanoRadar version wouldn't install on my CUDA (13)):
   ```bash
   grep -v '^cupy' requirements.txt > /tmp/req.txt
   python -m pip install -r /tmp/req.txt
   ```
4. Install CuPy and the imaging extras. OpenCV must be **below version 5** to work w/ PanoRadar:
   ```bash
   python -m pip install "cupy-cuda12x[ctk]" "opencv-python<5" scikit-image scipy matplotlib tqdm
   ```
5. Check it works (should print GPU name):
   ```bash
   python -c "import cupy as cp; print('GPU:', cp.cuda.runtime.getDeviceProperties(0)['name'].decode())"
   ```

> [!NOTE]
> Use `python -m pip`, not `pip`, inside the PanoRadar environment, and
> run `deactivate` first if `(.venv)` is showing. 
> Detectron2 not needed for this part (only for PanoRadar ML models) 

### A6. Link the PanoRadar sample (instead of copying GB)
This is for my downloaded data, but path should be changed accordingly.

```bash
mkdir -p ~/PanoRadar/data/raw
ln -s ~/Downloads/design/* ~/PanoRadar/data/raw/
ls ~/PanoRadar/data/raw          
```

### Make PanoRadar image

PanoRadar's `run.py imaging` has a typo (`out_plot_path`) that makes it crash,
so need to call the same function directly:

```bash
cd ~/PanoRadar && conda activate panoradar
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('<DATA FOLDER>', <FRAME>, '.')"
```

This saves `imaging_result.png` in `~/PanoRadar`. One frame = one full
rotation. 

---

## B. Test everything with fake data (no radar)

**B1. Image the authors' sample** (proves PanoRadar works):

```bash
deactivate 2>/dev/null; cd ~/PanoRadar && conda activate panoradar
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/design_static/exp20230524-000', 5, '.')"
mv imaging_result.png imaging_sample.png
```

✅ `imaging_sample.png` shows a map of a room.

**B2. Make a fake recording and convert it** (proves converter works):

```bash
conda deactivate; cd ~/fireflight-radar-testing && source .venv/bin/activate
python simulate_session.py --out sessions/sim_20261028-000000
python build_traj.py sessions/sim_20261028-000000 --building ~/PanoRadar/data/raw/sim_static --r-radar 0.077
deactivate
```

✅ Prints `Rotations: 6 full, 1198 beams each`. (`--r-radar 0.077` is the fake
platform's radius, so no need to edit `config.yaml` for this test.)

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


**B5. Moving data (fake walk):**

```bash
conda deactivate; cd ~/fireflight-radar-testing && source .venv/bin/activate
python simulate_session.py --out sessions/walk_20261028-000000 --seconds 6 --walk-speed 0.5
python build_traj.py sessions/walk_20261028-000000 --building ~/PanoRadar/data/raw/sim_moving --r-radar 0.077
deactivate
cd ~/PanoRadar && conda activate panoradar
python run.py motion_estimation --in_building_folder data/raw/sim_moving
python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/sim_moving/exp20261028-000', 3, '.')"
```

✅ `build_traj.py` prints `Mode: MOVING`. Motion estimation finishes with a
progress bar (a few `Open3D WARNING ... no data` lines are normal: there's no
LiDAR). The image shows the 4 dots, shifted left by how far the fake platform
walked.

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
   demo:
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
> radar's antennas** that PanoRadar uses to focus the image. 
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

### C3. Arduino motor logger

1. The Arduino sketch must print **one reading per line**, e.g.
   `Serial.println(ticks);`, with `Serial.begin(115200)`.
2. Upload the sketch, then **close the Arduino IDE's Serial Monitor** since only one program can use the USB port. 
3. Find the Arduino's port. The radar uses `ttyACM0` and `ttyACM1`, so on the
   radar PC the Arduino is usually `ttyACM2`:
   ```bash
   ls /dev/ttyACM* /dev/ttyUSB*
   ```
4. Start the motor, then the logger in separate terminals:
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python motor_logger.py --port /dev/ttyACM2 --out motor_test.csv
   ```
5. Every 2 s it prints the speed and the beams per rotation it gives. **If needed, adjust
   the motor until it reads about 1200 beams/rotation (1.70 rev/s, 102 rpm).**

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

### C5. Moving recordings (carrying the radar)

Moving data is recorded the same way. The difference is the **folder name**
and one extra PanoRadar step.

> [!IMPORTANT]
> - **Folder name must NOT contain `static`**, e.g. `lab1_moving`. That's how
>   PanoRadar knows the platform was moving.
> - **Walk slowly and straight** (about 0.5 m/s), keep the radar level, and
>   **avoid turning**. PanoRadar corrects turns using an IMU, which this setup
>   doesn't have, so turns blur the image.
> - Keep the motor at **1200 beams/rotation** the whole time.

1. Record: same as C4 steps 1–5 (motor logger + `record.py`), while walking.
2. **Convert into a `_moving` folder:**
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python build_traj.py sessions/walk1_<date-time> --motor-csv motor_walk1.csv \
       --building ~/PanoRadar/data/raw/lab1_moving
   deactivate
   ```
   ✅ It prints `Mode: MOVING`. The first and last half-rotation are dropped:
   PanoRadar pads each moving frame with beams from its neighbours.
3. **Estimate the motion** (PanoRadar's step; writes `motion_output.npz` into
   each trajectory folder):
   ```bash
   cd ~/PanoRadar && conda activate panoradar
   python run.py motion_estimation --in_building_folder data/raw/lab1_moving
   ```
   `Open3D WARNING ... no data` lines are normal. The `*_gt` values it saves
   are zeros, because they come from the LiDAR; the speed estimate itself
   (`v_esti`) comes from the radar.
4. **Image a frame:**
   ```bash
   python -c "from panoradar_SP import imaging_result; imaging_result.imaging_frame_from_traj('data/raw/lab1_moving/exp<date>-000', 0, '.')"
   ```

### C6. Recording with ROS 2 instead (same as record.py)

- xwr ROS node publishes each radar frame on `/xwr/iq`
- `bag_to_session.py` turns it into a normal session folder, then everything from C4 step 6 onwards is the same

*The stuff below is for Sadhana (since I'm on 24.04):*

> [!NOTE]
> Ubuntu 24.04 uses **ROS 2 Jazzy** (Humble is only for 22.04). The xwr ROS
> node hasn't been tested on Jazzy; `bag_to_session.py` reads Jazzy bags.

**One-time setup**

1. Install ROS 2 Jazzy:
   ```bash
   sudo apt install -y software-properties-common && sudo add-apt-repository -y universe
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
   sudo apt update && sudo apt install -y ros-jazzy-desktop ros-dev-tools
   ```
2. Install the build plugin, and the bag reader into this repo's environment:
   ```bash
   pip install --break-system-packages uv colcon-uv
   cd ~/fireflight-radar-testing && source .venv/bin/activate && pip install rosbags && deactivate
   ```
3. Let git download the node's xwr submodule without an SSH key:
   ```bash
   git config --global url."https://github.com/".insteadOf git@github.com:
   ```
4. Create the workspace and build the node:
   ```bash
   mkdir -p ~/ros2_ws/src ~/ros2_ws/config && cd ~/ros2_ws/src
   git clone --recurse-submodules https://github.com/RadarML/xwr_ros.git
   cd ~/ros2_ws && source /opt/ros/jazzy/setup.bash
   UV_VENV_CLEAR=1 UV_LINK_MODE=symlink colcon build --packages-select xwr_ros xwr_msgs
   ```
5. Make the node's config: a copy of `config.yaml` **without** the
   `platform:` section (the node passes the whole file to xwr, which rejects
   it). Re-run this whenever you change the radar settings:
   ```bash
   cd ~/fireflight-radar-testing
   python3 -c "import yaml; c = yaml.safe_load(open('config.yaml')); c.pop('platform', None); yaml.safe_dump(c, open('$HOME/ros2_ws/config/panoradar.yaml', 'w'))"
   ```

**Every recording.** Use three terminals; in each one first run
`source /opt/ros/jazzy/setup.bash && source ~/ros2_ws/install/setup.bash`.

1. **Terminal 1, radar node** (give the path without `.yaml`):
   ```bash
   ros2 launch xwr_ros radar.launch.py config:=$HOME/ros2_ws/config/panoradar
   ```
2. **Check it's streaming** in another terminal. It should show about 2040 Hz:
   ```bash
   ros2 topic hz /xwr/iq
   ```
3. **Terminal 2, motor log** (same as C4; no ROS needed):
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python motor_logger.py --port /dev/ttyACM2 --out ~/bags/lab1_motor.csv
   ```
4. **Terminal 3, record the bag:**
   ```bash
   mkdir -p ~/bags && ros2 bag record -o ~/bags/lab1 /xwr/iq /xwr/info
   ```
5. Stop with **Ctrl-C** in the order: bag (terminal 3), motor logger, radar node.
6. **Turn the bag into a session folder** (ROS doesn't need to be sourced):
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python bag_to_session.py ~/bags/lab1 --config config.yaml --motor-csv ~/bags/lab1_motor.csv
   ```
   It prints the session folder it made (`sessions/lab1_<date-time>`).
7. **Convert and image** exactly as in C4 step 6 onwards (static) or C5
   (moving). The motor log was already copied into the session folder, so
   `--motor-csv` can be left out.

### C7. ROS 2 Humble in Docker (optional, instead of C6)

Use this if you need ROS 2 **Humble** (e.g. to match lab machines) on Ubuntu
24.04. The container runs Ubuntu 22.04 with Humble inside; your system is
unchanged. It uses the PC's network port (`--net=host`), USB ports
(`--privileged -v /dev:/dev`) and clock, so timestamps still line up with
`motor_logger.py` running outside it. Workspace and bags live in folders on the
PC, so they survive the container.

> [!NOTE]
> These Docker steps haven't been tested on hardware yet. Do C1 (network,
> buffer, USB permissions) on the PC first; the container uses those settings.

**One-time setup** (on the PC unless it says "inside the container")

1. Install Docker, then **log out and back in**:
   ```bash
   sudo apt install -y docker.io
   sudo usermod -aG docker $USER
   ```
2. Create the shared folders and the node's config (the node needs
   `config.yaml` **without** its `platform:` section; re-run this whenever you
   change radar settings):
   ```bash
   mkdir -p ~/ros2_ws/src ~/ros2_ws/config ~/bags
   cd ~/fireflight-radar-testing
   python3 -c "import yaml; c = yaml.safe_load(open('config.yaml')); c.pop('platform', None); yaml.safe_dump(c, open('$HOME/ros2_ws/config/panoradar.yaml', 'w'))"
   ```
3. Install the bag reader into this repo's environment:
   ```bash
   source .venv/bin/activate && pip install rosbags && deactivate
   ```
4. Create the container. Plug the radar in first. `xhost` lets it open
   windows such as rviz2:
   ```bash
   xhost +local:
   docker run -it --name humble --net=host --privileged -v /dev:/dev \
     -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
     -v ~/ros2_ws:/root/ros2_ws -v ~/bags:/root/bags \
     osrf/ros:humble-desktop bash
   ```
5. **Inside the container**, build the radar node:
   ```bash
   apt update && apt install -y python3-pip git
   pip install uv colcon-uv
   git config --global url."https://github.com/".insteadOf git@github.com:
   cd /root/ros2_ws/src && git clone --recurse-submodules https://github.com/RadarML/xwr_ros.git
   cd /root/ros2_ws && source /opt/ros/humble/setup.bash
   UV_VENV_CLEAR=1 UV_LINK_MODE=symlink colcon build --packages-select xwr_ros xwr_msgs
   exit
   ```

**Every recording**

1. Start the container and open terminals in it. Plug the radar in first:
   ```bash
   xhost +local:
   docker start -ai humble        # container terminal 1
   docker exec -it humble bash    # container terminal 2 (in a new PC terminal)
   ```
   In **every** container terminal, first run:
   ```bash
   source /opt/ros/humble/setup.bash && source /root/ros2_ws/install/setup.bash
   ```
2. **Container terminal 1, radar node:**
   ```bash
   ros2 launch xwr_ros radar.launch.py config:=/root/ros2_ws/config/panoradar
   ```
3. **Container terminal 2, check it's streaming.** It should show about
   2040 Hz; stop it with Ctrl-C:
   ```bash
   ros2 topic hz /xwr/iq
   ```
4. **PC terminal, motor log** (outside the container, as in C4):
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   python motor_logger.py --port /dev/ttyACM2 --out ~/bags/lab1_motor.csv
   ```
5. **Container terminal 2, record the bag:**
   ```bash
   ros2 bag record -o /root/bags/lab1 /xwr/iq /xwr/info
   ```
6. Stop with **Ctrl-C** in the order: bag, motor logger, radar node.
7. **Convert on the PC** (the bag is in `~/bags/lab1`):
   ```bash
   cd ~/fireflight-radar-testing && source .venv/bin/activate
   sudo chown -R $USER ~/bags     # the container saves files as root
   python bag_to_session.py ~/bags/lab1 --config config.yaml --motor-csv ~/bags/lab1_motor.csv
   ```
   Then continue with `build_traj.py` as in C4 step 6 (static) or C5 (moving).
