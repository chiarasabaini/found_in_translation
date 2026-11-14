import test from 'node:test';
import assert from 'node:assert/strict';
import { matchPhrase, normalizePhrase } from './matcher';
import type { DictionaryClass } from '../types/lis';

const SAMPLE_CLASSES: DictionaryClass[] = [
  {
    id: 'grazie',
    phrase: 'Grazie',
    category: 'Cortesia',
    description: 'Mano al mento',
    twoHanded: false,
    synonyms: ['grazie mille', 'ti ringrazio'],
    anchor: {
      right_hand: [-0.04, 1.44, 0.26],
      left_hand: [0.2, 0.92, 0.08],
      right_wrist_euler_deg: [0, 0, 0],
      left_wrist_euler_deg: [0, 0, 0],
      motion_arc: 'chin_forward',
      motion_amplitude: [0, 0, 0.15],
      duration_sec: 1.4,
    },
    prototypeClipId: 'grazie_s00_c00',
    duration: 1.4,
    animationUrl: '/data/animations/grazie.json',
  },
  {
    id: 'come_stai',
    phrase: 'Come stai?',
    category: 'Conversazione',
    description: 'Domanda cortese',
    twoHanded: true,
    synonyms: ['come va', 'tutto bene'],
    anchor: {
      right_hand: [-0.18, 1.22, 0.3],
      left_hand: [0.18, 1.22, 0.3],
      right_wrist_euler_deg: [0, 0, 0],
      left_wrist_euler_deg: [0, 0, 0],
      motion_arc: 'symmetric_open',
      motion_amplitude: [0.06, 0, 0.05],
      duration_sec: 1.8,
    },
    prototypeClipId: 'come_stai_s00_c00',
    duration: 1.8,
    animationUrl: '/data/animations/come_stai.json',
  },
  {
    id: 'non_capisco',
    phrase: 'Non capisco',
    category: 'Comprensione',
    description: 'Fronte e scuotimento',
    twoHanded: false,
    synonyms: ['non ho capito', 'puoi ripetere'],
    anchor: {
      right_hand: [-0.16, 1.56, 0.22],
      left_hand: [0.2, 0.92, 0.08],
      right_wrist_euler_deg: [0, 0, 0],
      left_wrist_euler_deg: [0, 0, 0],
      motion_arc: 'temple_twist',
      motion_amplitude: [0.04, 0, 0.04],
      duration_sec: 1.7,
    },
    prototypeClipId: 'non_capisco_s00_c00',
    duration: 1.7,
    animationUrl: '/data/animations/non_capisco.json',
  },
  {
    id: 'acqua',
    phrase: 'Acqua',
    category: 'Bisogni',
    description: 'Bisogno di bere',
    twoHanded: false,
    synonyms: ['bere', 'ho sete'],
    anchor: {
      right_hand: [-0.07, 1.44, 0.24],
      left_hand: [0.2, 0.92, 0.08],
      right_wrist_euler_deg: [0, 0, 0],
      left_wrist_euler_deg: [0, 0, 0],
      motion_arc: 'chin_tap',
      motion_amplitude: [0, -0.05, 0],
      duration_sec: 1.4,
    },
    prototypeClipId: 'acqua_s00_c00',
    duration: 1.4,
    animationUrl: '/data/animations/acqua.json',
  },
];

test('normalizePhrase strips punctuation and diacritics', () => {
  assert.equal(normalizePhrase('  Perché, come stai?! '), 'perche come stai');
});

test('matches exact canonical and synonym phrases', () => {
  const resExact = matchPhrase('Come stai?', SAMPLE_CLASSES);
  assert.equal(resExact.matched, true);
  assert.equal(resExact.best?.classId, 'come_stai');

  const resSyn = matchPhrase('tutto bene', SAMPLE_CLASSES);
  assert.equal(resSyn.matched, true);
  assert.equal(resSyn.best?.classId, 'come_stai');
  assert.equal(resSyn.best?.viaSynonym, true);
});

test('matches minor typos via fuzzy similarity', () => {
  const res = matchPhrase('grazzie mile', SAMPLE_CLASSES);
  assert.equal(res.matched, true);
  assert.equal(res.best?.classId, 'grazie');
});

test('returns unmatched with 3 suggestions for out-of-dictionary phrases', () => {
  const res = matchPhrase('vorrei prenotare un treno per Milano', SAMPLE_CLASSES);
  assert.equal(res.matched, false);
  assert.equal(res.best, null);
  assert.equal(res.suggestions.length, 3);
});
