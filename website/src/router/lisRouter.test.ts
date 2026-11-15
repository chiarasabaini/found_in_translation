import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stemItalian } from './italianStemmer';
import { charSimilarity, guardsOk, LisRouter, normalizeText, type LisIndex, type LisIndexEntry, type SentenceEncoder } from './lisRouter';
import { LisRouterModel } from './lisRouterModel';
import { motionToAnimation } from '../generation/motionToAnimation';
import type { DictionaryPayload } from '../types/lis';
import { localizeClass } from '../i18n/languages';

test('Italian stemmer matches the Snowball reference', () => {
  // Reference values from the official Snowball implementation (snowballstemmer, Python).
  const ref: Record<string, string> = {
    abbandonata: 'abbandon', abbandono: 'abband', mangiare: 'mang', mangio: 'mang', mangiamo: 'mang', treni: 'tren',
    partire: 'part', partono: 'part', acqua: 'acqua', felicemente: 'felic', nazionalità: 'nazional', organizzazione: 'organizz',
    biologia: 'biolog', presenza: 'presenz', attivamente: 'attiv', capisco: 'cap', capire: 'cap', mangiandolo: 'mang',
  };
  for (const [w, exp] of Object.entries(ref)) assert.equal(stemItalian(w), exp, w);
});

test('normalization, guards and character similarity', () => {
  assert.equal(normalizeText("  Ciao, com'è?  "), "ciao com'è");
  assert.ok(guardsOk('vado a casa', 'vado a casa ora'));
  assert.ok(!guardsOk('vado', 'non vado'));
  assert.ok(!guardsOk('alle 19', 'alle 20'));
  assert.ok(!guardsOk('alle diciannove', 'alle 20'));
  assert.equal(charSimilarity('abc', 'abc'), 1);
  assert.ok(charSimilarity('ciao come stai', 'ciao come va') > 0.7);
});

const entry = (id: string, sentence: string, kind: 'word' | 'sentence', pos?: string): LisIndexEntry => ({
  id, sentence, norm: normalizeText(sentence), kind, lemma: kind === 'word' ? sentence : undefined, pos, clip: `clips/${id}.json`, duration: 2,
});
const INDEX: LisIndex = {
  version: 1,
  source: 'test',
  entries: [
    entry('ciao', 'ciao', 'word', 'NOUN'),
    entry('treno', 'treno', 'word', 'NOUN'),
    entry('partire', 'partire', 'word', 'VERB'),
    entry('mangiare', 'mangiare', 'word', 'VERB'),
    entry('andare', 'andare', 'word', 'VERB'),
    entry('casa', 'casa', 'word', 'NOUN'),
    entry('domani', 'domani', 'word', 'NOUN'),
    entry('non', 'non', 'word', 'ADV'),
    entry('dove', 'dove', 'word', 'ADV'),
    entry('mi_chiamo', 'mi chiamo', 'sentence'),
    entry('come_stai', 'come stai', 'sentence'),
    entry('come_stai_2', 'come stai', 'sentence'),
  ],
};

test('router: whole sentence (exact and lexical), with the negation and number guards', async () => {
  const r = new LisRouter(INDEX);
  let res = await r.translate('Come stai?');
  assert.equal(res.mode, 'sentence');
  assert.equal(res.matcher, 'exact');
  assert.deepEqual(res.entries.map((e) => e.id), ['come_stai']); // first take of the sentence
  res = await r.translate('come stai!!');
  assert.equal(res.mode, 'sentence');
});

test('router: words in LIS order with stems, irregular verbs, auxiliaries and modifiers', async () => {
  const r = new LisRouter(INDEX);
  // time -> subject -> object -> verb; "il", "alle" dropped; "19" has no clip
  let res = await r.translate('Domani il treno parte alle 19');
  assert.equal(res.mode, 'words');
  assert.deepEqual(res.entries.map((e) => e.id), ['domani', 'treno', 'partire']);
  assert.deepEqual(res.skipped, ['19']);
  // irregular form + negation at the end + modifier
  res = await r.translate('Non vado a casa');
  assert.deepEqual(res.entries.map((e) => e.id), ['casa', 'andare', 'non']);
  assert.deepEqual(res.modifiers, ['negation']);
  // auxiliary "ho" before a past participle is dropped; the participle reaches its verb by stem
  res = await r.translate('Ho mangiato a casa');
  assert.deepEqual(res.entries.map((e) => e.id), ['casa', 'mangiare']);
  // wh-question word goes last, elided "dov'"
  res = await r.translate("Dov'è casa?");
  assert.deepEqual(res.entries.map((e) => e.id), ['casa', 'dove']);
  assert.deepEqual(res.modifiers, ['question_wh']);
  res = await r.translate('Domani casa?');
  assert.deepEqual(res.modifiers, ['question_yn']);
  // multi-word dataset sentences inside the text are kept as one clip
  res = await r.translate('Ciao, mi chiamo Chiara');
  assert.deepEqual(res.entries.map((e) => e.id), ['ciao', 'mi_chiamo']);
  assert.deepEqual(res.skipped, ['chiara']);
  // nothing known
  res = await r.translate('buongiorno a tutti');
  assert.equal(res.mode, 'none');
});

test('router: semantic match through an encoder, with margin and guards', async () => {
  // Fake encoder: "come va" lands next to "come stai".
  const vecs: Record<string, number[]> = { 'come va': [1, 0.05, 0], 'come stai': [1, 0, 0], 'non come va': [1, 0.05, 0] };
  const encoder: SentenceEncoder = {
    name: 'fake',
    embed: async (texts) =>
      texts.map((t) => {
        const v = vecs[t] ?? [0, 0, 1];
        const n = Math.hypot(...v);
        return Float32Array.from(v.map((x) => x / n));
      }),
  };
  const r = new LisRouter(INDEX);
  await r.setEncoder(encoder);
  let res = await r.translate('come va?');
  assert.equal(res.mode, 'sentence');
  assert.equal(res.matcher, 'semantic');
  assert.deepEqual(res.entries.map((e) => e.id), ['come_stai']);
  // same embedding but a negation the dataset sentence does not have: rejected by the guard
  res = await r.translate('non come va');
  assert.notEqual(res.mode, 'sentence');
});

test('exported demo dataset: router model loads clips, joins them and adds the face markers', async () => {
  const index = JSON.parse(readFileSync('public/lis/index.json', 'utf8')) as LisIndex;
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL) => {
    const path = String(url).replace(/^\/lis\//, 'public/lis/');
    return new Response(readFileSync(path, 'utf8'));
  }) as typeof fetch;
  try {
    const model = new LisRouterModel(index, '/lis/');
    const dict = JSON.parse(readFileSync('public/data/dictionary.json', 'utf8')) as DictionaryPayload;
    model.setDictionary(dict.classes.map((c) => localizeClass(c, 'it', 'it')), (code) => dict.classes.map((c) => localizeClass(c, code, 'it')));

    const it = await model.generate({ text: 'Ciao, mi chiamo Chiara. Non capisco', sourceLanguage: 'it', signLanguage: 'LIS' });
    assert.deepEqual(it.clips?.map((c) => c.gloss), ['CIAO', 'MI-CHIAMO', 'NON-CAPISCO']);
    assert.deepEqual(it.unmatched, ['chiara']);
    assert.deepEqual(it.modifiers, ['negation']);
    const anim = motionToAnimation(it, 'Ciao, mi chiamo Chiara. Non capisco', 'quintic');
    assert.ok(anim);
    assert.equal(anim.transitions.length, 2);
    const brow = anim.payload.morphTracks.browDown;
    assert.ok(Math.max(...brow) >= 0.35, 'negation brows');

    // English input goes through the dictionary to Italian first.
    const en = await model.generate({ text: 'Thank you, goodbye', sourceLanguage: 'en', signLanguage: 'LIS' });
    assert.equal(en.route?.italian, 'grazie arrivederci');
    assert.deepEqual(en.clips?.map((c) => c.gloss), ['GRAZIE-ARRIVEDERCI']);
  } finally {
    globalThis.fetch = realFetch;
  }
});
