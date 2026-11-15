# Found in Translation — website

A one-page translator from text to Italian Sign Language (LIS). You type a phrase, as in a text translator, and a 3D avatar signs it.

## The page

- **Title** at the top, the **avatar** in the middle, a **chat-style input** at the bottom.
- **Input:** pick the languages (from English or Italian, to LIS), type the phrase, press Enter (Shift+Enter for a new line).
- **Result:** the signs (glosses, e.g. `CIAO · COME-STAI`) appear above the input, with the words that could not be signed. A compact player (play, replay, timeline, speed) sits above them.
- **Camera** (right side): Front, Hands, Face, and the skeleton view. Narrow screens get a wider framing so the arms stay in view.
- **Settings** (gear, top right): avatar choice and local VRM/GLB upload, avatar details (bones, face rig), transition curve between signs, motion model name, download of the last model output.

## Languages

- The UI is in English.
- **From** (left language picker): the language of the phrase you type. English and Italian are available. Each dictionary class has its text per language: the base fields (`phrase`, `category`, `description`, `synonyms`) use `metadata.base_language` (`it`), and `translations.<code>` holds the other languages.
- **To** (right language picker): the target sign language. It selects the dictionary (the vocabulary of the mock model). LIS is available.
- The choice is remembered per browser (localStorage).

To add a source language: add it to `SOURCE_LANGUAGES` in `src/i18n/languages.ts` and add `translations.<code>` to every class in `config/dictionary.yaml` (then re-run the pipeline) or directly in `public/data/dictionary.json`.
To add a sign language: add an entry to `SIGN_LANGUAGES` with its own `dictionary.json`.

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
  - `src/recording/`: import of recording-format CSV files (parser, retargeting, segment detection). The model output goes through it. It does not depend on three.js, so it runs under `node --test`.
  - `src/generation/`: motion model client, mock model, phrase-to-signs planner, clip joining (`stitchClips.ts`).
  - `src/components/`: the page (`TranslateComposer`, `SettingsMenu`, `AvatarViewport`).
  - `src/i18n/languages.ts`: source and sign language settings.
- `public/data/` — `dictionary.json` (vocabulary of the mock model) and the pipeline outputs (`animations/*.json`, `qa_summary.json`), which the page no longer plays.

## Recording format

The motion model returns the same files a glove recording session produces (Rokoko Smartgloves + camera tracking):

| File | Content |
|---|---|
| `left_hand.csv`, `right_hand.csv` | Hands solved by the Rokoko hand solver: `frame, timestamp_ms, glove_timestamp_us, hand_detected` + 26 joints × (`_pos_x/y/z`, `_rot_x/y/z/w`) |
| `face.csv` (optional) | `face_detected`, 478 MediaPipe landmarks `face_<i>_x/y/z`, 52 blendshapes `bs_<ARKit name>` |
| `body.csv` (optional) | `body_detected`, `neck/left_shoulder/right_shoulder/left_elbow/right_elbow` with `_x/y/z/_vis` |

All files share `timestamp_ms`.

Retargeting method (`src/recording/recordingRetarget.ts`): bone directions from joint positions (3 phalanges per finger, back of the hand, forearm), upper arm from `body.csv`, head from the face landmarks relative to the median pose, ARKit blendshapes → avatar morphs. The axis signs of `body.csv` and the camera mirroring are detected automatically.

## Motion model (text → motion)

The typed phrase goes to a text-to-motion model. The model returns recording-format files (one clip per sign, or one recording of the whole phrase), and the site animates the avatar with the recording import and retargeting, joining the clips with smooth transitions. The user does not choose how the signs are produced.

The real model does not exist yet. Until then, `src/generation/mockMotionModel.ts` plays its role in the browser: it splits the phrase into dictionary signs (`src/generation/signPlanner.ts`) and synthesizes one rest-to-rest clip per sign (CSV files at 30 Hz) from placeholder handshapes, the dictionary anchors and motion arcs, and arm IK, as the real model will. **The mock handshapes are not validated LIS signs.** The page shows the gloss sequence and the words it could not sign; the settings menu downloads the generated CSV files.

To use the real model, set `VITE_MOTION_MODEL_URL` at build time (see `.env.example`).

The repository router (`model/lis_router.py`) returns sample folders of the dataset: `mode="sentence"` (one folder for the whole phrase) or `mode="words"` (one folder per word, in LIS order). A backend endpoint can read those folders and answer with `clips` (one clip per folder, the CSV files as text); the site then joins them.

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

Instead of `files` (one continuous recording of the phrase), the model can return one clip per sign. Each clip starts and ends at rest; the site joins them (see "Joining per-sign clips" below):

```
{
  "clips": [
    {"text": "hello", "gloss": "CIAO", "classId": "ciao", "strokeStartSec": 0.35, "strokeEndSec": 1.85,
     "files": [{"name": "right_hand.csv", "text": "..."}, {"name": "left_hand.csv", "text": "..."}, ...]},
    {"text": "how are you", "gloss": "COME-STAI", "classId": "come_stai", "files": [...]}
  ],
  "unmatched": [],
  "model": "my-model v1"
}
```

`strokeStartSec` / `strokeEndSec` mark the sign inside the clip (after leaving rest, before returning). They are optional: when missing, the site detects them from the arm motion (`detectStroke`: the part where the arms hold the sign level, without the raise from rest and the drop back to rest).

File contract (same as the "Recording format" table above):

- `left_hand.csv`, `right_hand.csv` are required. `timestamp_ms` is required and shared by all files. Joint positions (`<Side><Joint>_pos_x/y/z`, meters, Y-up, right-handed, signer facing +Z, `ForeArm` at the origin) are required; `_rot_*` columns are optional and not used.
- `body.csv` is strongly recommended: without shoulders and elbows the hand position relative to the body is only estimated, and in LIS location changes meaning.
- `face.csv` is optional: ARKit blendshapes `bs_<name>` and the landmarks `face_1`, `face_10`, `face_33`, `face_152`, `face_263` for head rotation are enough.
- `segments`, `unmatched` and `model` are optional; they only feed the UI.

### Joining per-sign clips

`src/generation/stitchClips.ts` turns the per-sign clips into one animation:

1. Each clip is retargeted on its own (one glove yaw for all clips). Dictionary clips are already retargeted.
2. The stroke of every sign is kept unchanged, plus the lead-in of the first sign and the lead-out of the last one. The rest pose between signs is removed.
3. Each gap gets a transition from the end pose of sign A to the start pose of sign B. It matches pose and velocity (and acceleration, with the quintic curve) at both ends:
   - bone rotations: Hermite curve on the rotation vector, `q(t) = q0 · exp(h(t))`; the end velocity goes through the inverse right Jacobian of SO(3), so the angular velocity matches at both joins;
   - blendshapes: scalar Hermite curve, clamped to [0, 1].
4. The transition time comes from the largest arm rotation (0.15–0.6 s), and it is long enough that no bone goes faster than 7 rad/s (arms) or 12 rad/s (fingers, head) at its peak.
5. A hand that is idle in a sign between two signs that use it does not drop to rest: one long transition bridges it.

Curves: **Quintic (minimum jerk)**, the default, is C2 and gives the bell-shaped speed profile of human reaching movements; **Cubic Hermite** is C1. Pick one in the settings menu to compare them on the same output, without a new model request.

Limit: the transitions are smooth but neutral. Real coarticulation (the handshape of B forming during A) needs a model trained on continuous phrases or a motion in-betweening network.

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
npm test        # recording import, matcher, clip builder, languages, mock model, face rig, clip joining
npm run lint
npm run build
npm run dev
```

Test the recording import on a real recording:

```bash
LIS_RECORDING_DIR=/path/to/test_1_20261114_162048 node --test src/recording/recording.test.ts
```

## Avatars

The **avatar** select in the settings menu picks the avatar:

- **Woman (default)**: `assets/woman.glb` in the repository root (next to `website/`). The build copies it into `dist/assets/` with a content hash; the dev server serves it from the source folder (`server.fs.allow` in `vite.config.ts`). Any humanoid GLB/VRM works: the rig is analyzed when it loads. If the file cannot be loaded, the viewport shows the procedural avatar and a notice.
- **Procedural**: the avatar generated in code, always available.
- **Load VRM / GLB**: a local file. It stays in the list for the session.

The built-in choice is remembered per browser (localStorage). To add a built-in avatar, put the file in the repository `assets/` folder and add an entry to `BUILTIN_AVATARS` in `src/avatars/avatarCatalog.ts` (with a `new URL('../../../assets/<file>', import.meta.url)` literal); `DEFAULT_AVATAR_ID` sets the default.

## Using an external VRM / GLB avatar

1. Export a free humanoid avatar from **VRoid Studio / VRoid Hub** (`.vrm`) or **Ready Player Me** (`.glb` with ARKit blendshapes).
2. In the web app, open the settings menu (gear, top right) and click **Load VRM / GLB** to test the local file with `@pixiv/three-vrm`.
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

   Other head meshes (eyebrows, lashes, beard) follow only near the skin; hair strands do not. Each new avatar gets its own fit when it is loaded. The settings menu shows `auto face rig (N channels)` in the avatar details.

The default (procedural) avatar uses the same mixed-rig engine, with its hand-made deformations as the target: 10 facial bones (no lower lids) plus residual blendshapes. Left/right follow the avatar's side, as in ARKit: `eyeBlinkLeft` closes the avatar's left eye (+X).

Limits of level 3: meshes with closed lips have no mouth interior, so `mouthOpen` stretches the lips instead of showing teeth; painted eyes close by folding the lid texture. Test it on a real model with `LIS_AVATAR_GLB=/path/avatar.glb npx tsx --test src/three/autoFaceRig.test.ts`.
