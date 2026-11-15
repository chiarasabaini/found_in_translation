/**
 * Exports a LIS dataset for the static site (GitHub Pages: no server, no Python).
 *
 * Reads every sample folder under <dataset-root> (metadata.json + left_hand.csv,
 * right_hand.csv, face.csv, body.csv, the layout of dataset/ and model/lis_router.py),
 * retargets it with the same code the site uses, and writes:
 *   <out>/index.json        sentences, word lemmas and POS, clip paths, stroke times
 *   <out>/clips/<id>.json   retargeted animation (bone quaternions + face morphs, 30 fps)
 * The browser router (src/router/lisRouter.ts) loads the index and fetches only the clips
 * it plays, so no CSV parsing happens in the browser.
 *
 *   npx tsx scripts/export-lis-dataset.ts <dataset-root> [--out public/lis] [--source "name"] [--yaw auto|<deg>]
 *
 * metadata.json fields: "sentence" (required), "type": "word" | "sentence", "lemma", "pos"
 * ("VERB", "NOUN", ...; guessed from the word ending when missing).
 */
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, join, relative } from 'node:path';
import { createHash } from 'node:crypto';
import { parseRokokoRecording, type RecordingFileInput } from '../src/recording/rokokoRecording';
import { DEFAULT_RETARGET_OPTIONS, retargetRecording } from '../src/recording/recordingRetarget';
import { detectStroke } from '../src/generation/stitchClips';
import { guessPos, normalizeText, tokens, type LisIndex, type LisIndexEntry } from '../src/router/lisRouter';

const args = process.argv.slice(2);
const root = args.find((a) => !a.startsWith('--'));
const opt = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
if (!root || !existsSync(root)) {
  console.error('Usage: npx tsx scripts/export-lis-dataset.ts <dataset-root> [--out public/lis] [--source "name"] [--yaw auto|<deg>]');
  process.exit(1);
}
const out = opt('out', 'public/lis');
const source = opt('source', basename(root));
const yawArg = opt('yaw', 'auto');
const gloveYawDeg = yawArg === 'auto' ? null : Number(yawArg);

function findSamples(dir: string): string[] {
  const found: string[] = [];
  for (const name of readdirSync(dir).sort()) {
    const p = join(dir, name);
    if (!statSync(p).isDirectory()) continue;
    if (existsSync(join(p, 'metadata.json'))) found.push(p);
    else found.push(...findSamples(p));
  }
  return found;
}

const round = (x: number, k: number) => Math.round(x * k) / k;
const samples = findSamples(root);
rmSync(join(out, 'clips'), { recursive: true, force: true });
mkdirSync(join(out, 'clips'), { recursive: true });

const entries: LisIndexEntry[] = [];
let bytes = 0;
const t0 = performance.now();
for (const dir of samples) {
  let meta: Record<string, unknown>;
  try {
    meta = JSON.parse(readFileSync(join(dir, 'metadata.json'), 'utf8'));
  } catch {
    console.warn(`skip ${dir}: invalid metadata.json`);
    continue;
  }
  const sentence = typeof meta.sentence === 'string' ? meta.sentence.trim() : '';
  const norm = normalizeText(sentence);
  if (!norm) {
    console.warn(`skip ${dir}: no "sentence"`);
    continue;
  }
  const files: RecordingFileInput[] = ['left_hand.csv', 'right_hand.csv', 'face.csv', 'body.csv']
    .filter((f) => existsSync(join(dir, f)))
    .map((f) => ({ name: f, text: readFileSync(join(dir, f), 'utf8') }));
  const rec = parseRokokoRecording(files, basename(dir));
  const error = rec.diagnostics.find((d) => d.level === 'error');
  if (error) {
    console.warn(`skip ${dir}: ${error.message}`);
    continue;
  }
  const { payload } = retargetRecording(rec, {
    ...DEFAULT_RETARGET_OPTIONS,
    gloveYawDeg,
    startSec: 0,
    endSec: rec.durationSec,
    classId: basename(dir),
    phrase: sentence,
    category: 'Dataset',
    description: '',
  });
  // Smaller files: no per-frame QA data, 4 decimals for quaternions, 3 for morphs.
  payload.qaComparison = { group1VsGroup2Right: [], group1VsGroup2Left: [], frames: [] };
  delete payload.source;
  for (const k of Object.keys(payload.boneTracks)) payload.boneTracks[k] = payload.boneTracks[k].map((x) => round(x, 1e4));
  for (const k of Object.keys(payload.morphTracks)) payload.morphTracks[k] = payload.morphTracks[k].map((x) => round(x, 1e3));

  const hash = createHash('sha1').update(relative(root, dir)).digest('hex').slice(0, 6);
  const id = `${basename(dir).replace(/[^\w-]+/g, '_')}-${hash}`;
  const text = JSON.stringify(payload);
  writeFileSync(join(out, 'clips', `${id}.json`), text);
  bytes += text.length;

  const kind = meta.type === 'word' || meta.type === 'sentence' ? meta.type : tokens(norm).length === 1 ? 'word' : 'sentence';
  const stroke = detectStroke(payload);
  const entry: LisIndexEntry = {
    id,
    sentence,
    norm,
    kind,
    clip: `clips/${id}.json`,
    duration: payload.duration,
    strokeStartSec: round(stroke.startSec, 1e3),
    strokeEndSec: round(stroke.endSec, 1e3),
  };
  if (kind === 'word') {
    entry.lemma = typeof meta.lemma === 'string' ? meta.lemma.toLowerCase() : norm;
    entry.pos = typeof meta.pos === 'string' ? meta.pos.toUpperCase() : guessPos(entry.lemma);
  }
  entries.push(entry);
}

const index: LisIndex = { version: 1, source, entries };
writeFileSync(join(out, 'index.json'), JSON.stringify(index, null, 1));
const words = entries.filter((e) => e.kind === 'word').length;
console.log(
  `${entries.length} samples (${words} words, ${entries.length - words} sentences) -> ${out}: ` +
    `${(bytes / 1e6).toFixed(2)} MB of clips, ${((performance.now() - t0) / 1000).toFixed(1)} s`,
);
