import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { localizeClass, SIGN_LANGUAGES, SOURCE_LANGUAGES } from './languages';
import { matchPhrase } from '../utils/matcher';
import type { DictionaryPayload } from '../types/lis';

const dict = JSON.parse(readFileSync('public/data/dictionary.json', 'utf8')) as DictionaryPayload;
const base = dict.metadata.base_language ?? 'it';

test('every dictionary class has text for every source language', () => {
  for (const lang of SOURCE_LANGUAGES) {
    for (const cls of dict.classes) {
      const loc = localizeClass(cls, lang.code, base);
      assert.ok(loc.phrase, `${cls.id} has a ${lang.code} phrase`);
      if (lang.code !== base) {
        assert.ok(cls.translations?.[lang.code], `${cls.id} has translations.${lang.code}`);
      }
    }
  }
  assert.ok(SIGN_LANGUAGES.some((s) => s.code === dict.metadata.sign_language));
});

test('English phrases match the right LIS signs', () => {
  const en = dict.classes.map((c) => localizeClass(c, 'en', base));
  const cases: Array<[string, string]> = [
    ['Thanks a lot', 'grazie'],
    ['hello everyone', 'ciao'],
    ['Where is the bathroom?', 'bagno'],
    ["I don't understand", 'non_capisco'],
    ['i need help', 'aiuto'],
    ['how are you doing', 'come_stai'],
  ];
  for (const [q, id] of cases) {
    const res = matchPhrase(q, en, dict.metadata.match_threshold, dict.metadata.max_suggestions);
    assert.equal(res.best?.classId, id, `"${q}" -> ${id}, got ${res.best?.classId}`);
  }
  assert.equal(matchPhrase('book a train to Milan', en, dict.metadata.match_threshold).matched, false);
});

test('Italian phrases still match with the base text', () => {
  const it = dict.classes.map((c) => localizeClass(c, 'it', base));
  assert.equal(matchPhrase('grazie mille', it).best?.classId, 'grazie');
  assert.equal(matchPhrase("dov'è il bagno", it).best?.classId, 'bagno');
});
