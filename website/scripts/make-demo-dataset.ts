/**
 * Writes a demo LIS dataset in the layout of dataset/ (metadata.json + 4 CSV files per sample),
 * synthesized by the mock motion model from the dictionary classes.
 *
 * The handshapes are placeholders, NOT validated LIS signs: the demo only exercises the
 * router, the export and the clip joining until the real dataset is exported.
 *
 *   npx tsx scripts/make-demo-dataset.ts <out-dir>
 */
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import type { DictionaryPayload } from '../src/types/lis';
import { localizeClass } from '../src/i18n/languages';
import { synthesizeRecording } from '../src/generation/mockMotionModel';

const out = process.argv[2];
if (!out) {
  console.error('Usage: npx tsx scripts/make-demo-dataset.ts <out-dir>');
  process.exit(1);
}
const dict = JSON.parse(readFileSync(new URL('../public/data/dictionary.json', import.meta.url), 'utf8')) as DictionaryPayload;
const classes = dict.classes.map((c) => localizeClass(c, 'it', dict.metadata.base_language ?? 'it'));
const byId = new Map(classes.map((c) => [c.id, c]));

const clean = (s: string) => s.replace(/\.\.\.|…/g, '').replace(/[?!.,]/g, '').trim().toLowerCase();
const slug = (s: string) => clean(s).replace(/[^a-z0-9àèéìòù]+/g, '_');

// One sample per dictionary class, plus a few whole sentences made of several signs.
const samples: Array<{ sentence: string; ids: string[]; pos?: string }> = classes.map((c) => ({ sentence: clean(c.phrase), ids: [c.id] }));
samples.push(
  { sentence: 'ciao come stai', ids: ['ciao', 'come_stai'] },
  { sentence: 'grazie arrivederci', ids: ['grazie', 'arrivederci'] },
  { sentence: 'aiuto per favore', ids: ['aiuto', 'per_favore'] },
);

rmSync(join(out, 'demo'), { recursive: true, force: true });
samples.forEach((s, i) => {
  const dir = join(out, 'demo', `${String(i + 1).padStart(4, '0')}_${slug(s.sentence)}_demo`);
  mkdirSync(dir, { recursive: true });
  const { files } = synthesizeRecording(s.ids.map((id) => ({ text: s.sentence, cls: byId.get(id)! })));
  for (const f of files) writeFileSync(join(dir, f.name), f.text);
  const words = s.sentence.split(' ').length;
  writeFileSync(
    join(dir, 'metadata.json'),
    JSON.stringify({ sentence: s.sentence, type: words === 1 ? 'word' : 'sentence', source: 'demo (mock model, not real LIS)' }, null, 2),
  );
});
console.log(`${samples.length} demo samples in ${join(out, 'demo')}`);
