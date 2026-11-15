<!-- ![Logo](res/img/logo.png) -->


# Found in Translation by Espresso Yourself

![Tags](https://badgen.net/badge/icon/%23Rokoko/14406F1?icon=https://icons.getbootstrap.com/assets/icons/bookmarks-fill.svg&label&labelColor=FFF) ![Language](https://img.shields.io/badge/Built_with-Python-FFD9A0?labelColor=000&logo=py) ![Web](https://img.shields.io/badge/Web-TypeScript_·_Three.js-A0D8FF?labelColor=000) ![Version](https://badgen.net/badge/Version/01.01/666E88?labelColor=000) ![GroupName](https://badgen.net/badge/Team%20Name/Espresso%20Yourself/D9CCF5?labelColor=000) ![Author](https://badgen.net/badge/Authors/Chiara%20Davide%20Gabriele%20Matteo/A5E3F0?labelColor=000)

**"your hand speaks. now everybody listens"**

**Live demo: https://chiarasabaini.github.io/found_in_translation/**

---

# Summary

- [Description](#description)
- [How it works](#how-it-works)
- [The dataset](#the-dataset)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [Under the hood](#under-the-hood)
- [Limits and next steps](#limits-and-next-steps)
- [Notes](#notes)

---

# Description

**Found in Translation** turns written English or Italian into **Italian Sign Language (LIS)**, signed by a 3D avatar.

Text translators work for spoken languages, but sign languages live in the hands, the face and the space around the body. We recorded real LIS signs with **Rokoko Smartgloves** for the fingers and hands and two cameras for the face and the upper body, and built a dataset of signed words and phrases. From the text, the system picks the recordings to play: the whole phrase when the dataset has it, otherwise its words in LIS order. It joins them with smooth transitions and plays them on any humanoid avatar, facial expressions included.

The website works like a text translator: pick the languages, type a phrase, and the avatar signs it. It runs entirely in the browser, as a static site.

---

# How it works

```
 Rokoko Smartgloves ─┐
 face camera ────────┼─► generate.py ─► postprocess.py ─► dataset/data/processed
 body camera ────────┘   (capture)      (clean, trim,       161 samples, 4 CSV each
                                         resample 30 fps)          │
                                                                   ▼ export at build time
 "Where do you live?" ─► English → Italian ─► LIS router ─► dataset clips ─► joined ─► 3D avatar
                         (in the browser)     sentence or     (retargeted)   with smooth
                                              words in LIS                   transitions and
                                              order                          face markers
```

1. **Capture** ([`dataset/generate.py`](dataset/)): a teleprompter shows one Italian sentence at a time. Each take records the gloves (26 joints per hand), the face (478 landmarks + 52 ARKit blendshapes) and the upper body (neck, shoulders, elbows) in sync, starting from a calibration pose.
2. **Post-processing** ([`dataset/postprocess.py`](dataset/)): fills sensor gaps, trims the static start and end, makes the face independent of the camera, resamples to 30 fps and keeps the quaternions continuous.
3. **Routing** ([`model/lis_router.py`](model/), with a browser port in [`website/src/router/`](website/src/router/)): the whole sentence is matched semantically against the dataset sentences, with guards on negation and numbers. When there is no match, the words are reordered into LIS order (time → subject → object → verb → negation → question word) and looked up one by one.
4. **Animation** ([`website/`](website/)): every clip is retargeted onto the avatar skeleton (fingers, arms, torso, head and face). The rest pose between clips is replaced by minimum-jerk transitions, and the LIS face markers for questions and negation are added.
5. **Translation**: English text is translated to Italian in the browser first, and the original English also takes part in the semantic match.

---

# The dataset

| | |
|---|---|
| Samples | 161 takes (`dataset/data/processed`), with the raw recordings in `dataset/data/row` |
| Vocabulary | 111 distinct entries: 93 single words (including the numbers 1–30) and 18 phrases |
| Signers | 4 (Chiara 103 takes, Gabriele 28, Davide 25, Matteo 5) |
| Duration | ~10.3 minutes after trimming, 3.8 s per take on average, 30 fps |
| Per take | `left_hand.csv`, `right_hand.csv`, `face.csv`, `body.csv`, `metadata.json` |

Phrases include: *ciao mi chiamo Chiara*, *piacere di conoscerti*, *dove abiti?*, *io abito a Verona*, *quanti anni hai?*, *ho 22 anni*, *cosa fai domani?*, *a che ora è il treno?*, *il treno è alle 19*, *io torno a casa in treno*, *tu come torni a casa?*, *venerdì alle 19 facciamo aperitivo*, *con chi vieni all'aperitivo?*, *buona giornata*, *buon pomeriggio*, *per favore*.

To add signs, record and post-process them with [`dataset/`](dataset/README.md) and push: the website build exports the new samples automatically.

---

# Repository structure

| Folder | Content |
|---|---|
| [`dataset/`](dataset/) | Capture (`generate.py`), post-processing (`postprocess.py`) and 3D preview (`preview.py`) of the recordings, and the recordings in `data/` (`row/` raw, `processed/` post-processed). |
| [`model/`](model/) | `lis_router.py`: text → dataset recordings (semantic sentence match, otherwise spaCy lemmas in LIS order), with a threshold calibration mode. |
| [`website/`](website/) | The translator web app: browser port of the router, English → Italian translation, retargeting, clip joining, 3D avatar. |
| [`assets/`](assets/) | Avatars (`woman.glb`, the default one). |
| [`.github/workflows/`](.github/workflows/) | Build, test and deploy of the website to GitHub Pages on every push to `main`. |

---

# Getting started

| Part | Needs |
|---|---|
| Dataset capture | Rokoko Smartgloves and the [Rokoko Device SDK](dataset/README.md#files-not-in-the-repository), two cameras (face and body), Python 3 with `numpy`, `opencv-python`, `mediapipe` |
| Router | Python 3.9+, `sentence-transformers`, `spacy` with an Italian model |
| Website | Node.js 20.19+ |

**Record and post-process signs** (details in [`dataset/README.md`](dataset/README.md)):

```bash
cd dataset
pip install -r requirements.txt
python generate.py --height 1.65 --output-dir data/row/output-<name>   # record, one sentence per take
python postprocess.py --input-dir data/row --output-dir data/processed    # clean, trim, 30 fps
python preview.py data/processed/output-<name>/<sample>                   # check a take in 3D
```

**Run the router in Python** (details in [`model/README.md`](model/README.md)):

```bash
cd model
python installer.py                                   # sentence-transformers, spaCy, models
python lis_router.py query ../dataset/data/processed "il treno parte alle 19"
```

**Run the website** (details in [`website/README.md`](website/README.md)):

```bash
cd website
npm install
npm run dev      # exports the dataset, then http://localhost:3000
npm test
npm run build    # static site in dist/
```

The live site updates by itself: every push to `main` tests, builds and deploys it to GitHub Pages.

---

# Under the hood

- **Universal retargeting**: the recordings drive any humanoid GLB/VRM avatar (Ready Player Me, Avaturn, VRoid, Rigify, custom skeletons). Bone directions come from the joint positions, so the avatar proportions do not matter, and the glove yaw (magnetic north) and the camera mirroring are found automatically.
- **Automatic face rig**: avatars without facial blendshapes or bones get facial bones and blendshapes fitted to their own face geometry when they load: jaw, eyelids, brows, mouth corners and cheeks.
- **Clip joining**: each sign keeps its stroke unchanged. The rest pose between signs is replaced by a quintic (minimum-jerk) or cubic Hermite transition on the bone rotations. It matches pose, velocity and acceleration at both ends, and its duration depends on how far the arms move. A hand that is idle in a sign does not drop to rest.
- **Face markers**: raised brows for yes/no questions, furrowed brows for wh-questions, furrowed brows and a frown for negation.
- **In the browser, no server**: the router runs in TypeScript, using the Snowball Italian stemmer, function-word lists and irregular verbs in place of spaCy. The English → Italian translator (`opus-mt-en-it`) and the multilingual sentence encoder (`multilingual-e5-small`) run with transformers.js. The models download once (~250 MB in total) and stay cached. After that, a phrase takes 0.2–0.6 s.

---

# Limits and next steps

- **Vocabulary**: only the words and phrases in the dataset can be signed. Other words are listed as "not signed". Names need fingerspelling (the LIS alphabet), which is not recorded yet.
- **Paraphrases**: the semantic match accepts a paraphrase only above a similarity threshold. Some valid ones stay just below it ("dove vivi?" → *dove abiti?*), and some wrong ones come close ("buonanotte" → *buonasera*). A word-by-word alignment check is in progress.
- **Coarticulation**: transitions between signs are smooth but neutral. In real LIS the next handshape starts forming during the previous sign. That needs a model trained on continuous phrases.
- **Grammar**: the LIS word order and the face markers follow simple rules. LIS also uses space and classifiers, which the system does not model yet.
- **More signers and takes**: most takes come from one signer. More signers would make the signs more robust, and would let a generative model learn them.

---

# Notes

Deus ex mano

---

# Concept Poster

<!-- ![Poster](res/img/concept_poster.png) -->

---

Made with ♡ by humans
