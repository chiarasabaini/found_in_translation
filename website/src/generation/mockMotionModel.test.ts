import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import type { DictionaryPayload } from '../types/lis';
import { localizeClass, SOURCE_LANGUAGES } from '../i18n/languages';
import { MockMotionModel, planSigns } from './mockMotionModel';
import { parseRokokoRecording } from '../recording/rokokoRecording';
import { DEFAULT_RETARGET_OPTIONS, retargetRecording } from '../recording/recordingRetarget';

const dict = JSON.parse(readFileSync('public/data/dictionary.json', 'utf8')) as DictionaryPayload;
const en = dict.classes.map((c) => localizeClass(c, 'en', 'it'));
const it = dict.classes.map((c) => localizeClass(c, 'it', 'it'));

test('planSigns splits a phrase into dictionary signs', () => {
  const r = planSigns('Hello, thank you! I need help', en);
  assert.deepEqual(r.signs.map((s) => s.cls.id), ['ciao', 'grazie', 'aiuto']);
  const it1 = planSigns('Ciao, grazie mille', it);
  assert.deepEqual(it1.signs.map((s) => s.cls.id), ['ciao', 'grazie']);
  const unk = planSigns('hello pizza', en);
  assert.deepEqual(unk.signs.map((s) => s.cls.id), ['ciao']);
  assert.deepEqual(unk.unmatched, ['pizza']);
});

test('mock model output goes through the real recording import and retargeting', async () => {
  const model = new MockMotionModel(en);
  const out = await model.generate({ text: 'Hello, how are you?', sourceLanguage: 'en', signLanguage: 'LIS' });
  assert.deepEqual(out.segments.map((s) => s.gloss), ['CIAO', 'COME-STAI']);
  assert.equal(out.files.length, 4);

  const rec = parseRokokoRecording(out.files, 'generated');
  assert.ok(rec.left && rec.right && rec.face && rec.body);
  assert.ok(!rec.diagnostics.some((d) => d.level === 'error' || d.level === 'warning'), JSON.stringify(rec.diagnostics));

  const { payload, gloveYawDeg } = retargetRecording(rec, {
    ...DEFAULT_RETARGET_OPTIONS,
    gloveYawDeg: 0,
    startSec: 0,
    endSec: rec.durationSec,
    classId: 'generated',
    phrase: 'Hello, how are you?',
    category: 'Generated',
    description: '',
  });
  assert.equal(gloveYawDeg, 0);
  for (const [bone, flat] of Object.entries(payload.boneTracks)) {
    assert.ok(flat.every(Number.isFinite), bone);
  }
  // In the middle of "ciao" the right wrist must be near the dictionary anchor.
  const ciao = out.segments[0];
  const mid = (ciao.startSec + ciao.endSec) / 2;
  const frame = payload.qaComparison.frames.reduce((a, b) => (Math.abs(b.t - mid) < Math.abs(a.t - mid) ? b : a));
  const anchor = dict.classes.find((c) => c.id === 'ciao')!.anchor.right_hand;
  const d = Math.hypot(...frame.right.achieved_wrist.map((v, i) => v - anchor[i]));
  assert.ok(d < 0.12, `wrist ${frame.right.achieved_wrist} vs anchor ${anchor}: ${d.toFixed(3)} m`);
  // ciao uses an open hand: little finger nearly straight.
  assert.ok(frame.right.curls_deg_g2.little < 25, `little curl ${frame.right.curls_deg_g2.little}`);
  // the automatic yaw estimate must find the mock forward direction too
  const auto = retargetRecording(rec, { ...DEFAULT_RETARGET_OPTIONS, startSec: 0, endSec: 1, classId: 'g', phrase: '', category: '', description: '' });
  assert.ok(Math.abs(auto.gloveYawDeg) < 25, `auto yaw ${auto.gloveYawDeg}`);
});

test('every example phrase of the UI is fully signed', () => {
  for (const lang of SOURCE_LANGUAGES) {
    const classes = dict.classes.map((c) => localizeClass(c, lang.code, 'it'));
    for (const ex of lang.examples) {
      const r = planSigns(ex, classes);
      assert.ok(r.signs.length > 0, `${lang.code}: "${ex}" has no sign`);
      assert.deepEqual(r.unmatched, [], `${lang.code}: "${ex}" leaves ${r.unmatched.join(', ')}`);
    }
  }
});

test('unknown phrases produce no files', async () => {
  const out = await new MockMotionModel(en).generate({ text: 'book a train to Milan', sourceLanguage: 'en', signLanguage: 'LIS' });
  assert.equal(out.files.length, 0);
  assert.ok(out.unmatched.length > 0);
});
