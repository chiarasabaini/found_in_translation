# Found in Translation — website

A one-page translator from English or Italian to Italian Sign Language (LIS). You type a phrase, as in a text translator, and a 3D avatar signs it with the recordings of the LIS dataset ([`dataset/`](../dataset/)). Everything runs in the browser: the site is static and deploys to GitHub Pages.

## The page

- **Left column:** the title, the translator (languages, the text, **Translate**), the result, and the **Settings** at the bottom. Enter translates, Shift+Enter adds a new line.
- **Result:** the signs (glosses, e.g. `IO · CASA · 19 · MANGIARE`), the Italian text the router used when the input is English, and the words that could not be signed.
- **Avatar** on the right, in the rest pose (arms along the sides) until the first phrase, with a compact player (play, replay, timeline, speed) at the bottom center. Drag to orbit the camera. On phones the avatar is on top and the column below.
- **Settings:** avatar choice and local VRM/GLB upload, avatar details (bones, face rig), **Show skeleton**, transition curve between signs, the semantic sentence match, how the last phrase was routed, download of the last model output (model server only).

## From text to signs

```
English text ─► translator to Italian (browser, opus-mt-en-it) ─┐
Italian text ───────────────────────────────────────────────────┴─► LisRouter
   1. whole sentence: exact, semantic (multilingual encoder, also with the English text) or
      lexical match, with the negation / number guards and the margin of model/lis_router.py
   2. else words in LIS order: time → subject → object → verb → negation → question word;
      dataset phrases found inside the text are one clip ("mi chiamo")
─► dataset clips (public/lis/, already retargeted) ─► joined with smooth transitions
─► face markers for questions and negation ─► avatar
```

- **Router** (`src/router/lisRouter.ts`): TypeScript port of [`model/lis_router.py`](../model/README.md). No spaCy in the browser: function-word lists (a function word is signed when the dataset has a clip for it, e.g. "con"), a table of irregular verbs, the Snowball Italian stemmer (`src/router/italianStemmer.ts`, checked against the reference implementation), number words to digits ("sette" → the clip "7"), accent folding ("giù" → "giu"), the POS of the dataset words.
- **Translator** (`src/router/browserTranslator.ts`): `Xenova/opus-mt-en-it`, loaded when English is the source language (~130 MB once, then cached). Proper names the model changes are put back ("Calara" → "Chiara"); "19:00" becomes "19".
- **Semantic match** (`src/router/browserEncoder.ts`): `Xenova/multilingual-e5-small` (~120 MB once, then cached), on by default, can be turned off in the settings. It is multilingual, so the English text also matches the Italian dataset sentences ("Where do you live?" → "dove abiti?"). Without it the whole-sentence step uses exact and character similarity.
- Both models run with transformers.js, loaded from the CDN only when needed (`src/router/transformersCdn.ts`).
- **Face markers** from the router modifiers: raised brows for yes/no questions, furrowed brows for wh-questions, furrowed brows and a frown for negation.
- **Speed:** a phrase takes ~150–400 ms in the browser (translation included); the router itself < 1 ms. First load of the two models: ~10 s on a fast connection, ~1–5 s from the cache.

## The dataset in the site

The build exports the post-processed recordings of [`dataset/data/processed`](../dataset/data/processed) for the browser: `npm run build` and `npm run dev` run `scripts/export-lis-dataset.ts` first (`prebuild` / `predev`), which writes `public/lis/index.json` (sentences, word lemmas and POS, stroke times) and `public/lis/clips/<id>.json` (bone quaternions and face morphs at 30 fps, retargeted with the site code). `public/lis/` is generated, not committed. 139 samples take ~2.5 s and ~20 MB; the browser fetches only the clips it plays.

```bash
npm run export:lis -- <dataset-root> --out public/lis --source "LIS dataset"   # by hand, other datasets
```

`metadata.json` of each sample: `"sentence"` (required), optional `"type"` (`word` / `sentence`, default: one word = word clip), `"lemma"`, `"pos"` (`VERB`, `NOUN`, …; guessed from the ending when missing). Without a dataset the page says so and does not translate.

## Recording format

The motion model returns the same files a glove recording session produces (Rokoko Smartgloves + camera tracking):

| File | Content |
|---|---|
| `left_hand.csv`, `right_hand.csv` | Hands solved by the Rokoko hand solver: `frame, timestamp_ms, glove_timestamp_us, hand_detected` + 26 joints × (`_pos_x/y/z`, `_rot_x/y/z/w`) |
| `face.csv` (optional) | `face_detected`, 478 MediaPipe landmarks `face_<i>_x/y/z`, 52 blendshapes `bs_<ARKit name>` |
| `body.csv` (optional) | `body_detected`, `neck/left_shoulder/right_shoulder/left_elbow/right_elbow` with `_x/y/z/_vis` |

All files share `timestamp_ms`.

Retargeting method (`src/recording/recordingRetarget.ts`): bone directions from joint positions (3 phalanges per finger, back of the hand, forearm), upper arm from `body.csv`, head from the face landmarks relative to the median pose, ARKit blendshapes → avatar morphs. The axis signs of `body.csv` and the camera mirroring are detected automatically.

Retargeting method (`src/recording/recordingRetarget.ts`): bone directions from joint positions (3 phalanges per finger, back of the hand, forearm), upper arm from `body.csv`, head from the face landmarks relative to the median pose, ARKit blendshapes → avatar morphs. The axis signs of `body.csv` and the camera mirroring are detected automatically.

## Joining the clips

`src/generation/stitchClips.ts` turns the per-sign clips into one animation:

1. Each clip is retargeted on its own: at export time for the dataset clips, in the browser for model-server output (one glove yaw for all clips).
2. The stroke of every sign is kept unchanged, plus the lead-in of the first sign and the lead-out of the last one. The rest pose between signs is removed.
3. Each gap gets a transition from the end pose of sign A to the start pose of sign B. It matches pose and velocity (and acceleration, with the quintic curve) at both ends:
   - bone rotations: Hermite curve on the rotation vector, `q(t) = q0 · exp(h(t))`; the end velocity goes through the inverse right Jacobian of SO(3), so the angular velocity matches at both joins;
   - blendshapes: scalar Hermite curve, clamped to [0, 1].
4. The transition time comes from the largest arm rotation (0.15–0.6 s), and it is long enough that no bone goes faster than 7 rad/s (arms) or 12 rad/s (fingers, head) at its peak.
5. A hand that is idle in a sign between two signs that use it does not drop to rest: one long transition bridges it.

Curves: **Quintic (minimum jerk)**, the default, is C2 and gives the bell-shaped speed profile of human reaching movements; **Cubic Hermite** is C1. Pick one in the settings to compare them on the same output, without a new model request.

Limit: the transitions are smooth but neutral. Real coarticulation (the handshape of B forming during A) needs a model trained on continuous phrases or a motion in-betweening network.

## Model server (optional)

With `VITE_MOTION_MODEL_URL` set at build time (see `.env.example`), the page sends the phrase to that server instead of the browser router, for example `model/lis_router.py` behind an HTTP endpoint that reads the sample folders and answers with `clips`.

### API

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

Training data notes: normalize the glove yaw of each recording first (the gloves are referenced to magnetic north; `estimateGloveYawDeg` in `src/recording/recordingRetarget.ts` gives the estimate the site uses), and record the gloves at full rate (83 Hz) when possible.

## Project structure

- `src/router/`: router, stemmer, translator, encoder, dataset export (`exportDataset.ts`, Node only).
- `src/generation/`: clip joining (`stitchClips.ts`), model output to animation, model-server client.
- `src/recording/`: recording-format CSV parser and retargeting (no three.js: runs under `node --test`).
- `src/three/`: avatars, universal skeleton retargeting, automatic face rig.
- `src/components/`: the page (`TranslatePanel`, `SettingsPanel`, `AvatarViewport`).
- `scripts/export-lis-dataset.ts`: dataset → `public/lis/`.

## Commands

```bash
npm install
npm run dev     # exports the dataset, then http://localhost:3000
npm test        # router, stemmer, clip joining, retargeting, face rig, real-dataset checks
npm run lint
npm run build   # exports the dataset, then the static site in dist/
```

Test the recording import on a raw recording folder:

```bash
LIS_RECORDING_DIR=/path/to/recording node --test src/recording/recording.test.ts
```

## GitHub Pages

The build uses a relative base (`base: './'` in `vite.config.ts`), so `dist/` works under any path (`https://<owner>.github.io/<repo>/`). The workflow `.github/workflows/pages.yml` installs, tests, builds (dataset export included) and deploys `dist/` on every push to `main` (repository setting: **Pages → Source: GitHub Actions**).

## Avatars

The **avatar** select in the settings picks the avatar:

- **Woman (default)**: `assets/woman.glb` in the repository root (next to `website/`). The build copies it into `dist/assets/` with a content hash; the dev server serves it from the source folder (`server.fs.allow` in `vite.config.ts`). Any humanoid GLB/VRM works: the rig is analyzed when it loads. If the file cannot be loaded, the viewport shows the procedural avatar and a notice.
- **Procedural**: the avatar generated in code, always available.
- **Load VRM / GLB**: a local file. It stays in the list for the session.

The built-in choice is remembered per browser (localStorage). To add a built-in avatar, put the file in the repository `assets/` folder and add an entry to `BUILTIN_AVATARS` in `src/avatars/avatarCatalog.ts` (with a `new URL('../../../assets/<file>', import.meta.url)` literal); `DEFAULT_AVATAR_ID` sets the default.

## Using an external VRM / GLB avatar

1. Export a free humanoid avatar from **VRoid Studio / VRoid Hub** (`.vrm`) or **Ready Player Me** (`.glb` with ARKit blendshapes).
2. In the web app, open **Settings** in the left column and click **Load VRM / GLB** to test the local file with `@pixiv/three-vrm`.
3. If the morph target names differ from the ARKit standard, update `FACE_BLENDSHAPE_MAP` in `src/recording/recordingRetarget.ts`.

### Face rig: three levels, chosen per channel

The avatar loader drives the 9 face channels (`mouthOpen`, `mouthSmile`, `mouthFrown`, `mouthPucker`, `browUp`, `browDown`, `eyeBlinkLeft`, `eyeBlinkRight`, `cheekPuff`) with the best source the model has, channel by channel:

1. **Authored blendshapes** (ARKit, VRM/RPM visemes, …) when the model has them.
2. **Facial bones** (Rigify `DEF-lid`, `DEF-brow`, `jaw_master`, generic jaw) for the channels still missing.
3. **Automatic facial rig** (`src/three/autoFaceRig.ts`) for whatever is left. Models such as Avaturn exports have neither blendshapes nor face bones, only a `Head` bone. The loader then:
   - finds the face mesh among the meshes skinned to the head;
   - detects the landmarks from its geometry: nose tip, lip line (the shallow hollow between the lips, at about 1/3 from subnasale to chin), eye sockets (profile concavity), brows, face width;
   - defines a displacement field per channel fitted to that face (jaw rotation around a hinge near the ears, lid closure, brow raise, mouth corners, lips, cheeks);
   - builds a **mixed rig**:
     - **Facial bones**, children of `Head`, placed on the landmarks: `AutoFace_Jaw`, `AutoFace_EyelidUpper_L/R`, `AutoFace_EyelidLower_L/R`, `AutoFace_Brow_L/R`, `AutoFace_MouthCorner_L/R`, `AutoFace_Cheek_L/R`. The face vertices get skin weights toward them, fitted so that the bone motion reproduces each field. The bones carry the main motion and appear with **Show skeleton**.
     - **Residual blendshapes**: whatever the bones cannot reproduce, with the canonical channel names. Bones plus blendshapes give exactly the target shape.

   Other head meshes (eyebrows, lashes, beard) follow only near the skin; hair strands do not. Each new avatar gets its own fit when it is loaded. The settings show `auto face rig (N channels)` in the avatar details.

The default (procedural) avatar uses the same mixed-rig engine, with its hand-made deformations as the target: 10 facial bones (no lower lids) plus residual blendshapes. Left/right follow the avatar's side, as in ARKit: `eyeBlinkLeft` closes the avatar's left eye (+X).

Limits of level 3: meshes with closed lips have no mouth interior, so `mouthOpen` stretches the lips instead of showing teeth; painted eyes close by folding the lid texture. Test it on a real model with `LIS_AVATAR_GLB=/path/avatar.glb npx tsx --test src/three/autoFaceRig.test.ts`.
