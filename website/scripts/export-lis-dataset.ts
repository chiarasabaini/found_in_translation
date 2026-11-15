/**
 * Exports the LIS dataset for the static site (GitHub Pages: no server, no Python).
 *
 *   npx tsx scripts/export-lis-dataset.ts [dataset-root] [--out public/lis] [--source "name"] [--yaw auto|<deg>]
 *
 * Default root: ../dataset/data/processed (the post-processed recordings in the repository).
 * Writes <out>/index.json and <out>/clips/<id>.json (see src/router/exportDataset.ts). The
 * build runs it (npm prebuild / predev); with no dataset it warns and writes nothing, and the
 * page shows that no dataset is loaded.
 *
 * metadata.json fields: "sentence" (required), "type": "word" | "sentence", "lemma", "pos"
 * ("VERB", "NOUN", ...; guessed from the word ending when missing).
 */
import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { exportDataset } from '../src/router/exportDataset';

const args = process.argv.slice(2);
const root = args.find((a, i) => !a.startsWith('--') && !args[i - 1]?.startsWith('--')) ?? '../dataset/data/processed';
const opt = (name: string, fallback: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 && args[i + 1] ? args[i + 1] : fallback;
};
const out = opt('out', 'public/lis');
const source = opt('source', 'LIS dataset');
const yawArg = opt('yaw', 'auto');

if (!existsSync(root)) {
  console.warn(`export-lis-dataset: no dataset at ${root}, nothing exported.`);
  process.exit(0);
}
const t0 = performance.now();
const { index, clips } = exportDataset(root, {
  source,
  gloveYawDeg: yawArg === 'auto' ? null : Number(yawArg),
  onWarning: (m) => console.warn(m),
});
rmSync(out, { recursive: true, force: true });
mkdirSync(join(out, 'clips'), { recursive: true });
let bytes = 0;
for (const [path, payload] of clips) {
  const text = JSON.stringify(payload);
  bytes += text.length;
  writeFileSync(join(out, path), text);
}
writeFileSync(join(out, 'index.json'), JSON.stringify(index, null, 1));
const words = index.entries.filter((e) => e.kind === 'word').length;
console.log(
  `export-lis-dataset: ${index.entries.length} samples (${words} words, ${index.entries.length - words} sentences) -> ${out}: ` +
    `${(bytes / 1e6).toFixed(1)} MB of clips, ${((performance.now() - t0) / 1000).toFixed(1)} s`,
);
