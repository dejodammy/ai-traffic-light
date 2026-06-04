# Demo Implementation Guide

This document covers the full hardware and software setup for running the live demo with a Raspberry Pi, USB webcam, and physical traffic light LEDs.

---

## Hardware Required

- Raspberry Pi (any model with 40-pin GPIO header)
- USB webcam
- 4 mini traffic light LED units (one per approach: North, East, South, West)
- Jumper wires (female-to-male)
- 4 x 330Ω resistors (one per unit, or one per LED if wiring individually)
- Laptop or monitor to display the camera feed (optional)

---

## GPIO Wiring

Each traffic light unit has three LEDs: Red, Yellow, Green. Wire each LED through a 330Ω resistor to its GPIO pin. Connect the other leg of each LED to any GND pin on the Pi.

Use **BCM pin numbering** throughout.

| Approach | Red | Yellow | Green |
|---|---|---|---|
| North | GPIO 4 | GPIO 17 | GPIO 27 |
| East | GPIO 22 | GPIO 23 | GPIO 24 |
| South | GPIO 25 | GPIO 8 | GPIO 7 |
| West | GPIO 12 | GPIO 16 | GPIO 20 |

If you wire differently, update `DEFAULT_PIN_MAP` in `src/ai_traffic_light/gpio_bridge.py`.

**Wiring steps:**
1. Arrange the 4 traffic light units in a cross shape on the table labelled N, E, S, W
2. Run jumper wires from each LED to the corresponding GPIO pin via a 330Ω resistor
3. Connect all GND legs to GND pins on the Pi (there are 8 GND pins available)
4. Connect the USB webcam to any USB port on the Pi

---

## Software Setup on the Raspberry Pi

### 1. Clone or copy the project

```bash
git clone <your-repo-url>
cd ai-traffic-light
```

Or copy the project folder to the Pi via USB or `scp`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install RPi.GPIO
```

### 3. Copy the trained checkpoint to the Pi

Copy `results/lagos_peak_rl/best_dqn_model.pt` from your laptop to the Pi:

```bash
scp results/lagos_peak_rl/best_dqn_model.pt pi@<pi-ip>:~/ai-traffic-light/results/lagos_peak_rl/
```

---

## Running the Demo

### Full demo command

```bash
python src/main.py live \
  --checkpoint results/lagos_peak_rl/best_dqn_model.pt \
  --source 0 \
  --decision-interval 2.0 \
  --gpio \
  --yellow-duration 3.0 \
  --log-db results/demo_session.db
```

| Flag | Purpose |
|---|---|
| `--source 0` | USB webcam (index 0). Use `--source video.mp4` for a file instead |
| `--gpio` | Activates GPIO output to physical traffic lights |
| `--yellow-duration 3.0` | Holds yellow for 3 seconds before every phase switch |
| `--log-db` | Records every decision to a SQLite database |
| `--no-display` | Headless mode if no monitor is connected to the Pi |

### Headless (no monitor on the Pi)

```bash
python src/main.py live \
  --checkpoint results/lagos_peak_rl/best_dqn_model.pt \
  --source 0 \
  --decision-interval 2.0 \
  --gpio \
  --yellow-duration 3.0 \
  --log-db results/demo_session.db \
  --no-display
```

---

## What Happens at Runtime

1. All 4 traffic lights go **red** on startup (safe default)
2. The webcam starts reading frames every 2 seconds
3. YOLOv8 detects vehicles in each zone and counts them per approach
4. The DQN reads the 12-feature state vector and outputs a phase decision:
   - **Action 0** → East and West go **green**, North and South stay **red**
   - **Action 1** → North and South go **green**, East and West stay **red**
5. Before every phase switch:
   - The currently green pair goes **yellow** for 3 seconds
   - Then the new pair goes **green**
6. Every decision is printed to the terminal:
   ```
   [12] north=0, east=8, south=1, west=7  ->  phase 0
   [14] north=0, east=9, south=0, west=6  ->  phase 0
   [16] north=3, east=2, south=4, west=1  ->  phase 1
   ```
7. All decisions are saved to `results/demo_session.db`

---

## Demo Presentation Flow

### Step 1 — Show the comparison GIF (2 minutes)

Open `results/comparison.gif` on a laptop and walk through it. It shows the DQN and fixed-time controller running on the same traffic simultaneously. The queue divergence during the rush-hour surge is visible within the first 60 seconds.

### Step 2 — Run the live controller (3 minutes)

Point the USB webcam at a laptop screen playing traffic footage. A good source is dashcam or junction footage with heavy traffic in one direction.

Start the demo command. The examiner will see:
- The camera detecting vehicles on screen with bounding boxes per zone
- The physical traffic lights switching in response to detected traffic
- Yellow transitions before every phase change
- Terminal output showing live vehicle counts and phase decisions

**Tip:** Play `results/comparison.gif` on the laptop screen and point the webcam at it. The GIF already has heavy E-W traffic, so the DQN will hold E-W green during the surge — the physical lights will reflect this in real time.

### Step 3 — Show the decision log (1 minute)

After stopping the demo, open the SQLite database to show recorded decisions:

```bash
sqlite3 results/demo_session.db "SELECT timestamp, action, observations FROM decisions LIMIT 20;"
```

This demonstrates the logging pipeline and shows the examiner that the system was genuinely reading traffic state at each step.

### Step 4 — Show the results table (1 minute)

Pull up `docs/model_improvement_report.md` and point to the robustness table. Six scenarios tested, honest about where the model works and where it does not. This shows depth and maturity.

---

## What to Say During the Demo

> "The webcam is feeding frames into YOLOv8, which counts vehicles per approach. Those counts become the 12-feature state vector the DQN reads. Every 2 seconds the DQN outputs a phase decision — you can see it physically switching the lights. The yellow transition is automatic before every phase change, exactly like a real junction. On a production deployment the camera would be mounted above the intersection and the GPIO would connect to full-size signal heads, but the control logic is identical."

---

## Limitations to Mention

- The detection zones divide the frame into equal vertical bands. A production deployment would need camera calibration and per-lane zones mapped to the physical intersection geometry.
- The model was trained on simulated traffic. Real-world performance would require calibration against measured vehicle counts.
- The GPIO bridge blocks the capture loop during the yellow transition (3 seconds). This is acceptable for a demo but a production system would handle the transition asynchronously.

---

---

## 4-iPad Camera Setup

Each iPad acts as an IP camera streaming over WiFi. The laptop or Raspberry Pi reads all 4 streams simultaneously and treats each one as a separate approach lane.

### Step 1 — Install an IP camera app on each iPad

The recommended free app is **"IP Camera"** (search the App Store). Other options that work: **EpocCam**, **Camo**, **DotCamera**.

Once the app is running on an iPad it shows a stream URL on screen, for example:
```
http://192.168.1.101:8080/video
```

### Step 2 — Make sure everything is on the same WiFi network

The laptop/Pi and all 4 iPads must be connected to the same WiFi router. The IP address shown in the app is the iPad's local network address.

### Step 3 — Test each stream

```powershell
python src/main.py stream-test `
  --urls "http://192.168.1.101:8080/video,http://192.168.1.102:8080/video,http://192.168.1.103:8080/video,http://192.168.1.104:8080/video" `
  --labels north,east,south,west `
  --preview
```

Expected output:
```
  OK    north      http://192.168.1.101:8080/video
  OK    east       http://192.168.1.102:8080/video
  OK    south      http://192.168.1.103:8080/video
  OK    west       http://192.168.1.104:8080/video
```

### Step 4 — Run the full controller with 4 iPads

```powershell
python src/main.py live-multi `
  --checkpoint results/lagos_peak_rl/best_dqn_model.pt `
  --sources "http://192.168.1.101:8080/video,http://192.168.1.102:8080/video,http://192.168.1.103:8080/video,http://192.168.1.104:8080/video" `
  --approaches north,east,south,west `
  --decision-interval 2.0 `
  --gpio `
  --log-db results/demo_session.db
```

### Step 5 — Play traffic videos on each iPad

On each iPad, play a video of traffic approaching from that direction. Good sources:
- YouTube dashcam footage — search "Lagos traffic" or "busy intersection"
- Any road footage where vehicles are clearly visible approaching the camera

The system will detect vehicles on each iPad's screen and count them per approach. The 2×2 display grid on the laptop shows all 4 feeds simultaneously with bounding boxes.

### Physical arrangement

Place the 4 iPads in a cross shape on the table, each labelled with its direction:

```
        [ NORTH iPad ]
              |
[ WEST ]  ---+---  [ EAST ]
              |
        [ SOUTH iPad ]
```

The traffic light units sit in the middle of the cross. The laptop/Pi sits to one side running the controller.

### Troubleshooting

| Problem | Fix |
|---|---|
| Stream URL not connecting | Check all devices are on the same WiFi; try opening the URL in a browser first |
| Video is laggy | Reduce `--decision-interval` to 3.0 or use 5GHz WiFi |
| Wrong approach getting green | Check `--approaches` order matches `--sources` order |
| App shows different URL format | Try appending `/mjpeg` or `/stream` to the URL; check the app's documentation |

---

## Emergency Vehicle Detection

The system supports two modes of emergency vehicle detection:

### Mode 1 — Colour Flash Detector (works immediately, no training needed)

Detects red and blue flashing light regions in the camera frame using HSV colour analysis. No model required.

```powershell
python src/main.py live `
  --checkpoint results/lagos_peak_rl/best_dqn_model.pt `
  --source 0 `
  --emergency-color
```

### Mode 2 — Fine-tuned YOLO Model (accurate, labels by vehicle type)

Detects ambulances, fire trucks, and police cars by name. Requires a one-time training step.

#### Step 1 — Install Roboflow

```powershell
pip install roboflow
```

#### Step 2 — Get a free API key

1. Go to [roboflow.com](https://roboflow.com) and create a free account
2. Go to Settings → Roboflow API
3. Copy your API key

#### Step 3 — Train the model

```powershell
python src/train_emergency_detector.py --api-key YOUR_KEY_HERE --epochs 50
```

This downloads ~1,500 labelled images of ambulances, fire trucks, and police cars from Roboflow Universe, then fine-tunes YOLOv8n on them. Takes 10–20 minutes on a laptop CPU. The trained model is saved to `results/emergency_detector/best.pt`.

#### Step 4 — Test live detection

```powershell
python src/main.py detect-live --source 0 --model-name results/emergency_detector/best.pt
```

Point the camera at images of emergency vehicles — each one will now be labelled `ambulance`, `fire truck`, or `police` instead of `truck`.

#### Step 5 — Wire into the traffic controller

```powershell
python src/main.py live `
  --checkpoint results/lagos_peak_rl/best_dqn_model.pt `
  --source 0 `
  --emergency-model results/emergency_detector/best.pt `
  --gpio `
  --log-db results/demo_session.db
```

### What happens during preemption

When an emergency vehicle is detected on any approach:

1. The DQN decision is overridden immediately
2. The approach the emergency vehicle is on gets green
3. Green is held for **30 seconds**
4. The display shows a red bounding box and "EMERGENCY OVERRIDE" text
5. After 30 seconds, the DQN resumes normal control

### Live detection test (images or camera)

To test what the model labels on static images:

```powershell
python src/main.py detect-test firetruck.jpg ambulance.jpg car.jpg --show
```

To test on live camera feed:

```powershell
python src/main.py detect-live --source 0
python src/main.py detect-live --source 0 --vehicles-only
```

---

## GPIO Pin Reference (BCM)

```
Pi 40-pin header (odd pins on left, even on right):

 3V3  [ 1] [ 2] 5V
 SDA  [ 3] [ 4] 5V
 SCL  [ 5] [ 6] GND  <-- use for LED ground legs
GPIO4 [ 7] [ 8] TX
 GND  [ 9] [10] RX
GPIO17[11] [12] GPIO18
GPIO27[13] [14] GND  <-- use for LED ground legs
GPIO22[15] [16] GPIO23
 3V3  [17] [18] GPIO24
GPIO10[19] [20] GND
GPIO9 [21] [22] GPIO25
GPIO11[23] [24] GPIO8
 GND  [25] [26] GPIO7
...
GPIO12[32] ...
GPIO16[36] ...
GPIO20[38] ...
```

Pins used: 4, 7, 8, 12, 16, 17, 20, 22, 23, 24, 25, 27
