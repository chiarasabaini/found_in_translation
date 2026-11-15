# LIS Motion Studio — Junction 2026 Demo

End-to-end system (offline Python pipeline + static React/Three.js web app) for the retrieval, retargeting and 3D playback of Italian Sign Language (LIS) signs from **Smartgloves II** gloves and **Face Mesh**.

## Languages

- The UI is in English.
- **Source language** (top bar, left select): the language of the phrase you type. English and Italian are available. Each dictionary class has its text per language: the base fields (`phrase`, `category`, `description`, `synonyms`) use `metadata.base_language` (`it`), and `translations.<code>` holds the other languages.
- **Sign language** (top bar, right select): the target sign language. It selects the dictionary and the animations. LIS is available.
- The choice is remembered per browser (localStorage).

To add a source language: add it to `SOURCE_LANGUAGES` in `src/i18n/languages.ts` and add `translations.<code>` to every class in `config/dictionary.yaml` (then re-run the pipeline) or directly in `public/data/dictionary.json`.
To add a sign language: add an entry to `SIGN_LANGUAGES` with its own `dictionary.json`, `qa_summary.json` and animations.

## Project structure

- `config/dictionary.yaml` — Closed dictionary (10 LIS classes), phrases and synonyms per source language, body-relative spatial anchors and manual prototype override.
- `config/sensors_and_retargeting.yaml` — Header-driven mapping of the raw Smartgloves II CSV, quaternion order (`xyzw`/`wxyz`), cleaning/EMF thresholds, phalanx weights, IK arm lengths, 52 ARKit blendshape mapping.
- `pipeline/` — Offline pipeline, Python 3.10+:
  - `synthetic_generator.py`: generates a synthetic Smartgloves II CSV dataset (right/left hand, initial reset frame, ~83 Hz local microsecond clock, Group 1 + Hub + EM-corrected Group 2, dropouts and EMF spikes) + Face Mesh JSON (~30 fps, 52 blendshapes + head pose).
  - `glove_loader.py`: header-driven parser, reset frame removal, Group 1 vs Group 2 comparison, flag/EMF masking.
  - `signal_processing.py`: quaternion sign continuity, dropout SLERP, 1-Euro filter, resampling to a fixed frame rate (30 fps).
  - `prototype_selector.py`: per-class prototype selection with DTW medoid (manual override supported).
  - `retargeting.py`: neutral pose calibration, finger curl relative to the back of the hand spread over the phalanges, analytical 2-bone IK toward the class anchor, fade from/to the rest pose, face blendshape mapping.
  - `qa_compare.py`: QA metrics and `public/data/qa_summary.json`.
  - `run_pipeline.py`: full orchestrator.
- `src/` — Static client-side web app (React + Three.js + `@pixiv/three-vrm`).
  - `src/recording/`: in-browser import of real recordings (CSV parser, retargeting, segment detection, IndexedDB storage). It does not depend on three.js, so it runs under `node --test`.
  - `src/i18n/languages.ts`: source and sign language settings.
- `public/data/` — Precomputed JSON tracks (`dictionary.json`, `animations/*.json`, `qa_summary.json`).

## Real recordings ("Recordings" view)

The web app reads a recording folder directly, without the Python pipeline:

| File | Content |
|---|---|
| `left_hand.csv`, `right_hand.csv` | Hands solved by the Rokoko hand solver: `frame, timestamp_ms, glove_timestamp_us, hand_detected` + 26 joints × (`_pos_x/y/z`, `_rot_x/y/z/w`) |
| `face.csv` (optional) | `face_detected`, 478 MediaPipe landmarks `face_<i>_x/y/z`, 52 blendshapes `bs_<ARKit name>` |
| `body.csv` (optional) | `body_detected`, `neck/left_shoulder/right_shoulder/left_elbow/right_elbow` with `_x/y/z/_vis` |

All files share `timestamp_ms`. Workflow:

1. Open **Recordings** and drop the folder (or use **Choose folder**).
2. Pick a segment (suggested automatically from the forearm elevation) and check the avatar.
3. If needed, correct the glove yaw (the gloves are referenced to magnetic north), globally or per hand.
4. **Use for "…"** replaces the animation of that sign in the interpreter (saved in the browser, per sign language). **Download JSON** produces a file in the format of `public/data/animations/<id>.json`: copy it there to make it permanent.

Retargeting method (`src/recording/recordingRetarget.ts`): bone directions from joint positions (3 phalanges per finger, back of the hand, forearm), upper arm from `body.csv`, head from the face landmarks relative to the median pose, ARKit blendshapes → avatar morphs. The axis signs of `body.csv` and the camera mirroring are detected automatically.

## Motion model (text → motion)

In the **Interpreter**, the default mode is **Motion model**: the typed phrase goes to a text-to-motion model, the model returns recording-format files, and the site animates the avatar with the same import and retargeting used for real recordings. **Dictionary clips** mode keeps the old behavior (closed dictionary → stored animation).

The real model does not exist yet. Until then, `src/generation/mockMotionModel.ts` plays its role in the browser: it splits the phrase into dictionary signs and synthesizes the CSV files at 30 Hz from placeholder handshapes, the dictionary anchors and motion arcs, and arm IK. **The mock handshapes are not validated LIS signs.** The interpreter shows the gloss sequence, the words it could not sign, and buttons to download the generated CSV files.

To use the real model, set `VITE_MOTION_MODEL_URL` at build time (see `.env.example`).

### Motion model API

```
POST <VITE_MOTION_MODEL_URL>
{"text": "Hello, how are you?", "sourceLanguage": "en", "signLanguage": "LIS"}

200 OK
{
  "files": [
    {"name": "right_hand.csv", "text": "frame,timestamp_ms,glove_timestamp_us,hand_detected,RightForeArm_pos_x,..."},
    {"name": "left_hand.csv",  "text": "..."},
    {"name": "body.csv",       "text": "..."},
    {"name": "face.csv",       "text": "..."}
  ],
  "segments": [{"text": "hello", "gloss": "CIAO", "classId": "ciao", "startSec": 0.35, "endSec": 1.85}],
  "unmatched": [],
  "model": "my-model v1"
}
```

File contract (same as the "Real recordings" table above):

- `left_hand.csv`, `right_hand.csv` are required. `timestamp_ms` is required and shared by all files. Joint positions (`<Side><Joint>_pos_x/y/z`, meters, Y-up, right-handed, signer facing +Z, `ForeArm` at the origin) are required; `_rot_*` columns are optional and not used.
- `body.csv` is strongly recommended: without shoulders and elbows the hand position relative to the body is only estimated, and in LIS location changes meaning.
- `face.csv` is optional: ARKit blendshapes `bs_<name>` and the landmarks `face_1`, `face_10`, `face_33`, `face_152`, `face_263` for head rotation are enough.
- `segments`, `unmatched` and `model` are optional; they only feed the UI.

Training data notes: normalize the glove yaw of each recording first (the gloves are referenced to magnetic north; `estimateGloveYawDeg` in `src/recording/recordingRetarget.ts` gives the estimate the site uses), and record the gloves at full rate (83 Hz) when possible.

The model normally runs on a GPU server, not in the browser. AI Studio apps can have a Node backend (`express` is already a dependency) that forwards the request and keeps API keys secret; the browser then calls that backend route.

## Quick commands

### 1. Python pipeline (tests + synthetic generation + JSON export)

```bash
# Optional, local environment with pip:
pip install -r pipeline/requirements.txt

# Unit tests (also work with the Python 3.10 standard library only):
PYTHONPATH=. python3 -m unittest discover -s pipeline/tests -v

# Generate synthetic data and export the tracks for the web app:
python3 -m pipeline.run_pipeline --clips-per-class 4 --signers 4 --regenerate
```

To use a real raw-sensor dataset instead of the synthetic one, create a `dataset_manifest.json` in the dataset folder (with `class_id`, `clip_id`, `right_csv`, `left_csv`, `face_json`) and run:

```bash
python3 -m pipeline.run_pipeline --dataset data/real --out public/data
```

### 2. Web app (development, TypeScript tests and static build)

```bash
npm install
npm test        # recording import, matcher, clip builder, languages
npm run lint
npm run build
npm run dev
```

Test the recording import on a real recording:

```bash
LIS_RECORDING_DIR=/path/to/test_1_20261114_162048 node --test src/recording/recording.test.ts
```

## Avatars

The **avatar** select in the top bar picks the avatar shown in every viewport:

- **Woman (default)**: `assets/woman.glb` in the repository root (next to `website/`). The build copies it into `dist/assets/` with a content hash; the dev server serves it from the source folder (`server.fs.allow` in `vite.config.ts`). Any humanoid GLB/VRM works: the rig is analyzed when it loads. If the file cannot be loaded, the viewport shows the procedural avatar and a notice.
- **Procedural**: the avatar generated in code, always available.
- **Load VRM / GLB**: a local file. It stays in the list for the session.

The built-in choice is remembered per browser (localStorage). To add a built-in avatar, put the file in the repository `assets/` folder and add an entry to `BUILTIN_AVATARS` in `src/avatars/avatarCatalog.ts` (with a `new URL('../../../assets/<file>', import.meta.url)` literal); `DEFAULT_AVATAR_ID` sets the default.

## Using an external VRM / GLB avatar

1. Export a free humanoid avatar from **VRoid Studio / VRoid Hub** (`.vrm`) or **Ready Player Me** (`.glb` with ARKit blendshapes).
2. In the web app, click **Load VRM / GLB** at the top right to test the local file with `@pixiv/three-vrm`.
3. If the morph target names differ from the ARKit standard, update `retargeting.face_blendshape_map` in `config/sensors_and_retargeting.yaml` and re-run `python3 -m pipeline.run_pipeline` (for recordings: `FACE_BLENDSHAPE_MAP` in `src/recording/recordingRetarget.ts`).

### Face rig: three levels, chosen per channel

The avatar loader drives the 9 face channels (`mouthOpen`, `mouthSmile`, `mouthFrown`, `mouthPucker`, `browUp`, `browDown`, `eyeBlinkLeft`, `eyeBlinkRight`, `cheekPuff`) with the best source the model has, channel by channel:

1. **Authored blendshapes** (ARKit, VRM/RPM visemes, …) when the model has them.
2. **Facial bones** (Rigify `DEF-lid`, `DEF-brow`, `jaw_master`, generic jaw) for the channels still missing.
3. **Automatic facial rig** (`src/three/autoFaceRig.ts`) for whatever is left. Models such as Avaturn exports have neither blendshapes nor face bones, only a `Head` bone. The loader then:
   - finds the face mesh among the meshes skinned to the head;
   - detects the landmarks from its geometry: nose tip, lip line (the shallow hollow between the lips, at about 1/3 from subnasale to chin), eye sockets (profile concavity), brows, face width;
   - defines a displacement field per channel fitted to that face (jaw rotation around a hinge near the ears, lid closure, brow raise, mouth corners, lips, cheeks);
   - builds a **mixed rig**:
     - **Facial bones**, children of `Head`, placed on the landmarks: `AutoFace_Jaw`, `AutoFace_EyelidUpper_L/R`, `AutoFace_EyelidLower_L/R`, `AutoFace_Brow_L/R`, `AutoFace_MouthCorner_L/R`, `AutoFace_Cheek_L/R`. The face vertices get skin weights toward them, fitted so that the bone motion reproduces each field. The bones carry the main motion and appear in the **Skeleton** view.
     - **Residual blendshapes**: whatever the bones cannot reproduce, with the canonical channel names. Bones plus blendshapes give exactly the target shape.

   Other head meshes (eyebrows, lashes, beard) follow only near the skin; hair strands do not. Each new avatar gets its own fit when it is loaded. The viewport label shows `auto face rig (N channels)`.

The default (procedural) avatar uses the same mixed-rig engine, with its hand-made deformations as the target: 10 facial bones (no lower lids) plus residual blendshapes. Left/right follow the avatar's side, as in ARKit: `eyeBlinkLeft` closes the avatar's left eye (+X).

Limits of level 3: meshes with closed lips have no mouth interior, so `mouthOpen` stretches the lips instead of showing teeth; painted eyes close by folding the lid texture. Test it on a real model with `LIS_AVATAR_GLB=/path/avatar.glb npx tsx --test src/three/autoFaceRig.test.ts`.
