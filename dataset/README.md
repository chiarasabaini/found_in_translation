# Sign Language Dataset Acquisition (`generate.py`), Post-Processor (`postprocess.py`) & 3D Player (`preview.py`)

Complete, high-framerate dataset acquisition, post-processing, and 3D playback pipeline for **Italian Sign Language (LIS)** designed for training sequence-to-sequence neural networks (**Italian Sentence $\rightarrow$ 4 Synchronized 3D CSVs**).

---

## 1. Installation & Environment Setup

```bash
# Activate virtual environment
source .venv/bin/activate

# Install required dependencies (numpy, opencv-python, mediapipe)
pip install -r requirements.txt
```

### Files not in the repository

`config/`, `examples/`, `models/` and the output folders are in [`.gitignore`](./.gitignore). Before the first run:

| Path | What to put there |
| :--- | :--- |
| `examples/solver_core/` | The Rokoko hand solver package from the Rokoko Device SDK: copy or link `<sdk>/examples/python/solver_core` (the installer puts the SDK in `~/.local/share/rokoko-device-sdk`). [`core/gloves_tracker.py`](./core/gloves_tracker.py) imports it. |
| `config/config.json` | Optional. Mannequin proportions; without it the built-in defaults (or `--height`) are used. |
| `config/prompts.txt` | Optional. One Italian sentence per line for the Teleprompter (or pass `--prompts`). |
| `models/` | Nothing: the MediaPipe `face_landmarker.task` and `pose_landmarker_lite.task` models are downloaded there on the first run. |

---

## 2. Hardware Architecture (Parallel Multi-Threaded Pipeline)

The pipeline merges **3 synchronized hardware sources** in real time (~`45–48 FPS` capture rate):

| Source | Device & Worker | Orientation | Extracted Data |
| :--- | :--- | :--- | :--- |
| **1. Face Camera** (close-up) | Fixed OpenCV index `0` (`FaceTrackerWorker` @ `640x480 60FPS`) | **Fixed `90°` CW** (`9:16` portrait) | MediaPipe `FaceLandmarker` on dedicated background thread: **478 3D facial landmarks** (`x, y, z`) + **52 ARKit facial expression blendshapes** (`[0..1]`). |
| **2. Body Camera** (distant) | Fixed OpenCV index `1` (`BodyTrackerWorker` @ `640x480 60FPS`) | **Fixed `0°`** (`16:9` landscape) | MediaPipe `PoseLandmarker` on dedicated background thread: **5 upper-body joints** (`neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`) in shoulder-centered metric 3D coordinates (`pose_world_landmarks`, `x, y, z, visibility`). |
| **3. Rokoko Smartgloves** | RGMP v2 (`127.0.0.1:12276`, `rgmp_glove_worker`) | Auto-calibrated (3s countdown) | `rokoko-sdk` + `solver_core`: **26 3D joints per hand** (`ForeArm`, `Hand`, and 5 fingers $\times$ 5 joints) with elbow-relative 3D positions (`x, y, z` in meters) and 3D rotation quaternions (`x, y, z, w`). |

---

## 3. Kinematic Fusion, Mannequin Proportions & Mandatory Calibration

1. **Anthropometric Mannequin Configuration ([`config/config.json`](./config/config.json) or `--height`)**:
   - All anatomical dimensions (`actor_height_m`, `neck_to_chin_y_m`, `face_height_m`, `face_width_m`, `hand_rig_scale`, shoulder/pelvis/elbow offsets) are configurable via [`config/config.json`](./config/config.json) (`--config`).
   - Passing `--height <meters>` (e.g. `--height 1.65` or `--height 1.80`) automatically deduces all segment lengths and hand rig scaling proportionally from total stature, just like Rokoko Studio.
2. **Upper-Body & Chin-Anchored Face Mesh**:
   - The distant **Body Camera** drives the metric 3D positions of `neck`, `left_shoulder`, `right_shoulder`, `left_elbow`, and `right_elbow`.
   - The **478-point Face Mesh** is projected via vectorized NumPy math and anchored by its chin landmark (`#152`) onto `head_base` (`neck_to_chin_y_m = 0.115m` above the neck), so hands touching the mouth or chin in sign language align anatomically in 3D.
3. **Smartgloves Elbow Anchoring**:
   - Each Smartglove solves the 26-joint kinematic chain starting from the elbow (`LeftForeArm` / `RightForeArm` $\rightarrow$ `LeftHand` / `RightHand` $\rightarrow$ 5 fingers), scaled by `hand_rig_scale` (`0.82` at `1.75m`) and anchored onto `left_elbow` and `right_elbow`.
4. **Mandatory 3-Second Calibration on Every Take**:
   - Pressing **`SPACE`** to start a recording triggers a **mandatory 3-second countdown** (`CALIBRATING IN 3.0s`).
   - Hold arms along the sides with forearms bent straight forward: right before the countdown reaches `0.0s`, horizontal forearm yaw is zeroed so **Frame #1** starts already calibrated.

---

## 4. Continuous Teleprompter & Dataset Labeling Workflow

To record dozens of Italian sentences without restarting the program:

1. **Prepare [`config/prompts.txt`](./config/prompts.txt)**:
   - Write one Italian sentence per line (lines starting with `#` are ignored).
2. **Auto-Resume & Manifest Self-Cleaning**:
   - On startup, [`generate.py`](./generate.py) reads `manifest.csv` inside the target output folder, automatically prunes any entries whose session folder was manually deleted from disk, and positions the **Teleprompter** on the first sentence with `0` recorded takes.
3. **RAM-Buffered Recording & Auto-Advance (`SPACE`)**:
   - During recording, frames are buffered in RAM (with `frame_seq` deduplication so duplicate camera frames are never written) and flushed to disk in a single burst into `<slug>_<YYYYMMDD_HHMMSS>` when you press **`SPACE`** to stop, automatically advancing to the next sentence.
4. **Instant Discard & Redo (`x` / `Backspace`)**:
   - Pressing **`x`** (or **`Backspace`**) during or right after a take deletes that take folder, removes its entry from `manifest.csv`, and keeps the Teleprompter on that sentence to redo it immediately.

---

## 5. Neural-Network Post-Processing Pipeline (`postprocess.py`)

[`postprocess.py`](./postprocess.py) (and [`core/postprocessor.py`](./core/postprocessor.py)) prepares raw recordings for neural network training through **5 deterministic stages**:

1. **Missing-Modality Gap Interpolation ([`_fill_missing_modalities`](./core/postprocessor.py))**:
   - Reconstructs any frame where `face_detected == 0`, `body_detected == 0`, or a hand packet was dropped using **LERP** for 3D positions/blendshapes and **SLERP** for hand quaternions.
2. **Static Calibration-Pose Head & Tail Trimming ([`_trim_head_and_tail`](./core/postprocessor.py))**:
   - No fixed timer is used. Since every take starts and ends in the static calibration pose (arms along the sides, forearms bent forward), the algorithm computes for every frame $t$:
     - The 3D RMS distance $d_{\text{start}}(t)$ and $d_{\text{end}}(t)$ of wrists, middle fingertips, and elbows from the initial and final calibration poses.
     - The 5-frame smoothed 3D motion velocity $v(t)$ (in $\text{m/s}$).
   - Automatically trims the initial static hold (until $d_{\text{start}}(t) > 2.5\text{ cm}$ and $v(t) > 0.08\text{ m/s}$) and the final static hold (when the arms return to the calibration pose and settle below $0.08\text{ m/s}$), keeping a `4`-frame natural margin on both ends.
3. **Canonical Chin-Centered Face Normalization ([`_canonicalize_face_landmarks`](./core/postprocessor.py))**:
   - Centers all 478 facial landmarks on the **Chin (`#152`)** and normalizes horizontal/vertical face span into a canonical `[0..1]` box (`chin = (0.50, 0.85, 0.0)`, `span = (0.50, 0.70)`), making `face.csv` **100% invariant to camera translation and actor distance** while remaining directly playable in `preview.py`.
4. **Constant-FPS Temporal Resampling ([`_resample_constant_fps`](./core/postprocessor.py))**:
   - Resamples variable-dt raw capture (`~45–48 FPS`) onto a strict uniform time grid (default **`30.0 FPS`**, $\Delta t = 33.33\text{ ms}$) using **LERP** (positions, blendshapes, visibilities) and **SLERP** (4D hand quaternions).
5. **Quaternion Hemisphere Continuity ([`_enforce_quaternion_continuity`](./core/postprocessor.py))**:
   - Enforces $\text{dot}(q_t, q_{t-1}) \ge 0$ across all 26 joints of each hand so $q$ and $-q$ never flip sign between consecutive frames (preventing L1/MSE loss spikes during training).

---

## 6. Dataset & CSV Structure

```text
dataset/
├── row/                                             # Raw recorded sessions grouped by speaker
│   ├── output-chiara/
│   │   ├── manifest.csv                             # Global index for PyTorch/TensorFlow Dataset
│   │   ├── ciao_mi_chiamo_chiara_20261114_185050/
│   │   │   ├── metadata.json                        # Sentence, frame count, duration, FPS, mannequin config
│   │   │   ├── left_hand.csv                        # 26 left-hand 3D joints (pos + quat, 186 cols)
│   │   │   ├── right_hand.csv                       # 26 right-hand 3D joints (pos + quat, 186 cols)
│   │   │   ├── face.csv                             # 478 3D face landmarks + 52 blendshapes (1489 cols)
│   │   │   └── body.csv                             # 5 upper-body 3D joints: neck, shoulders, elbows (23 cols)
│   │   └── ...
│   ├── output-davide/
│   ├── output-gabriele/
│   └── output-matteo/
└── processed/                                       # Post-processed sessions (mirrored folder structure, 30 FPS)
    ├── output-chiara/
    ├── output-davide/
    ├── output-gabriele/
    └── output-matteo/
```

---

## 7. Codebase Structure

Only the **3 executable scripts (`generate.py`, `postprocess.py`, and `preview.py`)** and [`requirements.txt`](./requirements.txt) sit at the root; configuration files live in [`config/`](./config), library modules live in [`core/`](./core), and recorded datasets live in `dataset/` (or `output/` for new raw captures):

| File / Folder | Responsibility |
| :--- | :--- |
| [`generate.py`](./generate.py) | **Executable — Dataset Recorder**: launches parallel `FaceTrackerWorker` and `BodyTrackerWorker` threads, Rokoko Smartgloves worker, live 3D Studio Teleprompter viewport, and RAM-buffered CSV export. |
| [`postprocess.py`](./postprocess.py) | **Executable — Neural-Network Post-Processor**: batch-cleans recorded sessions preserving speaker subfolder hierarchy (`--input-dir` $\rightarrow$ `--output-dir`). |
| [`preview.py`](./preview.py) | **Executable — 3D Playback Viewer**: replays any raw or post-processed session folder with a live **`[t]`** toggle to switch between **RAW** and **POST-PROCESSED** frames in real time. |
| [`config/`](./config) | Configuration & prompt lists (`config.json`, `prompts.txt`, `frasi.txt`, `frasi2.txt`, `frasi3.txt`, `frasi4.txt`). |
| [`core/config.py`](./core/config.py) | Hardware constants + [`MannequinConfig`](./core/config.py) loader supporting JSON files (`--config`) and anthropometric height scaling (`--height`). |
| [`core/state.py`](./core/state.py) | Thread-safe [`SharedState`](./core/state.py) with unified `handle_key()` dispatcher, mandatory 3s calibration countdown timer, and Teleprompter queue state. |
| [`core/camera_tracker.py`](./core/camera_tracker.py) | Parallel [`FaceTrackerWorker`](./core/camera_tracker.py) and [`BodyTrackerWorker`](./core/camera_tracker.py) threads (`640x480 @ 60 FPS`) with pre-rendered PiP thumbnails and `frame_seq` tracking. |
| [`core/gloves_tracker.py`](./core/gloves_tracker.py) | RGMP v2 client (`127.0.0.1:12276`), 26-joint 3D kinematic hand solver ([`examples/solver_core`](./examples/solver_core)), and forearm yaw calibration. |
| [`core/renderer_3d.py`](./core/renderer_3d.py) | Shared 3D perspective renderer (`Camera3D`, vectorized NumPy face projection, [`render_live_studio_frame`](./core/renderer_3d.py), and [`render_playback_studio_frame`](./core/renderer_3d.py)). |
| [`core/csv_exporter.py`](./core/csv_exporter.py) | [`SessionCsvExporter`](./core/csv_exporter.py) (RAM buffering, 4 CSVs + `metadata.json` + self-cleaning `manifest.csv`, take rollback) and [`load_session`](./core/csv_exporter.py). |
| [`core/postprocessor.py`](./core/postprocessor.py) | [`postprocess_frames`](./core/postprocessor.py) and [`postprocess_session_dir`](./core/postprocessor.py) implementing the 5-stage neural-network post-processing pipeline. |
| `dataset/` | Collected dataset split into `dataset/row/` (raw captures) and `dataset/processed/` (cleaned 30 FPS sequences). |
| `models/` | Local MediaPipe task models (`face_landmarker.task`, `pose_landmarker_lite.task`). |

---

## 8. Usage & Controls

### A. Record Dataset (`generate.py`)
```bash
# Start with default ./config/config.json and ./config/prompts.txt
python generate.py

# Deduce all mannequin proportions automatically from actor height (like Rokoko Studio)
python generate.py --height 1.65 --prompts config/frasi.txt --output-dir output/output-chiara

# Load a custom mannequin configuration JSON file
python generate.py --config config/config.json
```

| Key (`generate.py`) | Action |
| :--- | :--- |
| **`SPACE`** | **Start** (3s straight-pose calibration countdown) / **Stop** recording & auto-advance to next sentence |
| **`x`** / **`Backspace`** | **Discard & Redo**: delete the active or just-recorded take and stay on that sentence |
| **`p`** / **`n`** (or **`[`** / **`]`**) | Step to **Previous / Next** sentence in prompts file |
| **`c`** | Re-calibrate Smartgloves neutral forward orientation |
| **`a` / `d` / `w` / `s` / `r`** | Orbit / Tilt / Reset 3D Studio camera |
| **`q`** / **`ESC`** | Quit |

---

### B. Post-Process Dataset for Neural Networks (`postprocess.py`)
```bash
# Process all sessions from dataset/row -> dataset/processed at 30 FPS
python postprocess.py --input-dir dataset/row --output-dir dataset/processed

# Process at 60 FPS
python postprocess.py --input-dir dataset/row --output-dir dataset/processed --fps 60

# Process a single speaker folder or a single session folder
python postprocess.py dataset/row/output-chiara --output-dir dataset/processed/output-chiara
python postprocess.py dataset/row/output-chiara/ciao_mi_chiamo_chiara_20261114_185050
```

---

### C. Replay Recorded or Post-Processed 3D Animation (`preview.py`)
```bash
# Replay a specific raw session (press [t] at any time to toggle RAW vs POST-PROCESSED!)
python preview.py dataset/row/output-chiara/piacere_di_conoscerti_20261114_185239

# Replay a specific post-processed session
python preview.py dataset/processed/output-chiara/piacere_di_conoscerti_20261114_185239

# Replay the latest recorded session in a folder
python preview.py --output-dir dataset/row/output-chiara
python preview.py --output-dir dataset/processed/output-chiara --processed
```

| Key (`preview.py`) | Action |
| :--- | :--- |
| **`SPACE`** | **Play / Pause** animation |
| **`t`** | **Toggle live between `RAW` and `POST-PROCESSED` playback** |
| **`n`** / **`m`** (or **`←`** / **`→`**) | **Step 1 frame** backward / forward |
| **`+`** / **`-`** | Increase / decrease playback speed (`0.25x` – `4.0x`) |
| **`0`** | Restart from Frame #1 |
| **`a` / `d` / `w` / `s` / `r`** | Orbit / Tilt / Reset 3D Studio camera |
| **`q`** / **`ESC`** | Quit |
