import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { stemItalian } from './italianStemmer';
import { charSimilarity, guardsOk, LisRouter, normalizeText, type LisIndex, type LisIndexEntry, type SentenceEncoder } from './lisRouter';
import { LisRouterModel } from './lisRouterModel';
import { motionToAnimation } from '../generation/motionToAnimation';
import { exportDataset } from './exportDataset';
import { guessPos } from './lisRouter';
import { properNames, restoreNames } from './browserTranslator';

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

test('router: function words with a clip, accents, number words, verb guess', async () => {
  const r = new LisRouter({
    version: 1,
    source: 'test',
    entries: [
      entry('con', 'con', 'word', 'NOUN'),
      entry('giu', 'giu', 'word', 'NOUN'),
      entry('si', 'si', 'word', 'NOUN'),
      entry('7', '7', 'word', 'NOUN'),
      entry('casa', 'casa', 'word', 'NOUN'),
    ],
  });
  // "con" has its own clip: signed; "a" has none: dropped.
  let res = await r.translate('con te a casa');
  assert.deepEqual(res.entries.map((e) => e.id), ['con', 'casa']);
  assert.deepEqual(res.skipped, ['tu']);
  // accents: "giù" reaches "giu", "sì" reaches "si"; the clitic "si" does not.
  res = await r.translate('giù, sì');
  assert.deepEqual(res.entries.map((e) => e.id), ['giu', 'si']);
  res = await r.translate('si lava');
  assert.deepEqual(res.entries.map((e) => e.id), []);
  // number words reach the digit clips
  res = await r.translate('sette');
  assert.deepEqual(res.entries.map((e) => e.id), ['7']);
  assert.equal(guessPos('bere'), 'VERB');
  assert.equal(guessPos('casa'), 'NOUN');
});

const DATASET = '../dataset/data/processed';

test('real dataset: export, router model, translation to Italian and cross-lingual match', { skip: !existsSync(DATASET) }, async () => {
  const { index, clips } = exportDataset(DATASET, { source: 'test', gloveYawDeg: null });
  assert.ok(index.entries.length >= 100, `${index.entries.length} samples`);
  assert.ok(index.entries.some((e) => e.kind === 'sentence') && index.entries.some((e) => e.kind === 'word'));
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (url: string | URL) => {
    const payload = clips.get(String(url).replace(/^\/lis\//, ''));
    return payload ? new Response(JSON.stringify(payload)) : new Response('', { status: 404 });
  }) as typeof fetch;
  try {
    const model = new LisRouterModel(index, '/lis/');
    // Italian: whole sentence, then words in LIS order with numbers.
    let out = await model.generate({ text: 'Dove abiti?', sourceLanguage: 'it', signLanguage: 'LIS' });
    assert.equal(out.route?.mode, 'sentence');
    assert.deepEqual(out.clips?.map((c) => c.gloss), ['DOVE-ABITI']);
    out = await model.generate({ text: 'Io mangio a casa alle 19', sourceLanguage: 'it', signLanguage: 'LIS' });
    assert.equal(out.route?.mode, 'words');
    assert.deepEqual(out.clips?.map((c) => c.gloss), ['IO', 'CASA', '19', 'MANGIARE']);
    assert.ok(out.clips?.every((c) => c.payload && c.payload.times.length > 10));
    const anim = motionToAnimation(out, 'Io mangio a casa alle 19', 'quintic');
    assert.equal(anim?.transitions.length, 3);

    // Other languages need their translator; the original text joins the semantic match.
    await assert.rejects(model.generate({ text: 'Thank you', sourceLanguage: 'en', signLanguage: 'LIS' }), /translator/);
    model.setTranslator('en', { model: 'fake', toItalian: async (t) => ({ 'thank you': 'grazie', 'i live in verona': 'vivo a verona' })[t.toLowerCase()] ?? t });
    out = await model.generate({ text: 'Thank you', sourceLanguage: 'en', signLanguage: 'LIS' });
    assert.equal(out.route?.italian, 'grazie');
    assert.deepEqual(out.clips?.map((c) => c.gloss), ['GRAZIE']);
    // "vivo a verona" has no lexical match; a multilingual encoder brings the English
    // sentence next to "io abito a Verona".
    const target = 'io abito a verona';
    await model.setEncoder({
      name: 'fake',
      embed: async (texts) => texts.map((t) => Float32Array.from(t === target || t === 'i live in verona' ? [1, 0] : [0, 1])),
    });
    out = await model.generate({ text: 'I live in Verona', sourceLanguage: 'en', signLanguage: 'LIS' });
    assert.equal(out.route?.mode, 'sentence');
    assert.equal(out.route?.matcher, 'semantic');
    assert.deepEqual(out.clips?.map((c) => c.gloss), ['IO-ABITO-A-VERONA']);
  } finally {
    globalThis.fetch = realFetch;
  }
});

test('translator: proper names are restored, hour times normalized', () => {
  assert.deepEqual(properNames('Hello, my name is Chiara and I live in Verona'), ['Chiara', 'Verona']);
  assert.deepEqual(properNames('Thank you. Goodbye'), []);
  assert.equal(restoreNames('My name is Chiara', 'Mi chiamo Calara.'), 'Mi chiamo Chiara.');
  assert.equal(restoreNames('I live in Verona', 'Vivo a Verona'), 'Vivo a Verona');
  assert.equal(normalizeText('Il treno è alle 19:00.'), 'il treno è alle 19');
});
