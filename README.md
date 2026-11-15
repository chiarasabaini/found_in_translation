<!-- ![Logo](res/img/logo.png) -->


# Found in Translation by Espresso Yourself

![Tags](https://badgen.net/badge/icon/%23Rokoko/14406F1?icon=https://icons.getbootstrap.com/assets/icons/bookmarks-fill.svg&label&labelColor=FFF) ![Language](https://img.shields.io/badge/Built_with-Python-FFD9A0?labelColor=000&logo=py) ![Version](https://badgen.net/badge/Version/01.01/666E88?labelColor=000) ![GroupName](https://badgen.net/badge/Team%20Name/Espresso%20Yourself/D9CCF5?labelColor=000) ![Author](https://badgen.net/badge/Authors/Chiara%20Davide%20Gabriele%20Matteo/A5E3F0?labelColor=000)

**"your hand speaks. now everybody listens"**

---

# Summary

- [Found in Translation by Espresso Yourself](#found-in-translation-by-espresso-yourself)
- [Summary](#summary)
- [Description](#description)
- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Website](#website)
- [Notes](#notes)

---

# Description

**Found in Translation** translates written Italian or English into **Italian Sign Language (LIS)**, signed by a 3D avatar.

We record real LIS signers with **Rokoko Smartgloves** (fingers and hands) and cameras (face expressions and upper body), and build a dataset of signed words and phrases. From the text, the system finds the recordings to play (the whole phrase if it is in the dataset, otherwise its words in LIS order), joins them with smooth transitions, and retargets the motion onto any humanoid avatar, face included. The website works like a text translator: type a phrase, pick the languages, and the avatar signs it.

---

# Repository structure

| Folder | Content |
|---|---|
| [`dataset/`](dataset/) | Capture of the LIS dataset (gloves + face and body cameras, synchronized), post-processing and 3D preview. Output: 4 CSV files per sample (`left_hand`, `right_hand`, `face`, `body`). The recordings are in `dataset/data/` (`row/` raw, `processed/` post-processed). |
| [`model/`](model/) | Text → LIS animation routing (`lis_router.py`): semantic match of the whole phrase, otherwise spaCy lemmas reordered in LIS order and looked up word by word. The website runs a browser port of it. |
| [`website/`](website/) | The web app: text input, motion retargeting, clip joining and 3D avatar playback. |
| [`assets/`](assets/) | Avatars (`woman.glb`, the default one). |

---

# Requirements

| Part | Needs |
|---|---|
| Dataset capture ([`dataset/`](dataset/)) | Rokoko Smartgloves and the [Rokoko Device SDK](dataset/README.md#files-not-in-the-repository), two cameras (face and body), Python 3 with `numpy`, `opencv-python`, `mediapipe` |
| Router ([`model/`](model/)) | Python 3.9+, `sentence-transformers`, `spacy` with an Italian model (`python model/installer.py`) |
| Website ([`website/`](website/)) | Node.js 20.19+ (Vite 8) |

---

# Website

[`website/`](website/) is a one-page translator: English or Italian text in, LIS out on a 3D avatar. English is translated to Italian in the browser; the router of [`model/`](model/), ported to the browser, picks the dataset recordings (the whole sentence, or the words in LIS order); the clips are joined with smooth transitions (quintic minimum-jerk curves) with the LIS face markers for questions and negation. The build exports [`dataset/data/processed`](dataset/data/processed) for the browser, and every push to `main` deploys the static site to GitHub Pages: https://chiarasabaini.github.io/found_in_translation/

Run it with `cd website && npm install && npm run dev`, then open http://localhost:3000. Details are in [`website/README.md`](website/README.md).

---

# Notes

Deus ex mano

---

# Concept Poster

<!-- ![Poster](res/img/concept_poster.png) -->

---

Made with ♡ by humans
