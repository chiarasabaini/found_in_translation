/**
 * Phrase -> sequence of dictionary signs.
 *
 * Used by the mock motion model and by the Dictionary mode of the interpreter: a phrase with
 * several dictionary phrases ("Hello, how are you? Thank you") becomes several signs, which are
 * then joined with smooth transitions (stitchClips).
 */

import type { DictionaryClass } from '../types/lis';
import { matchPhrase, normalizePhrase } from '../utils/matcher';

export interface PlannedSign {
  text: string;
  cls: DictionaryClass;
}

/**
 * Splits a phrase into dictionary signs: longest word n-grams first, strict similarity.
 * Returns the signs in order and the words that match nothing.
 */
export function planSigns(text: string, classes: DictionaryClass[]): { signs: PlannedSign[]; unmatched: string[] } {
  const whole = matchPhrase(text, classes, 0.9, 1);
  if (whole.matched && whole.best) {
    const cls = classes.find((c) => c.id === whole.best?.classId);
    if (cls) return { signs: [{ text: text.trim(), cls }], unmatched: [] };
  }
  const tokens = normalizePhrase(text).split(' ').filter(Boolean);
  const signs: PlannedSign[] = [];
  const unmatched: string[] = [];
  let i = 0;
  while (i < tokens.length) {
    let found = false;
    for (let len = Math.min(6, tokens.length - i); len >= 1; len--) {
      const chunk = tokens.slice(i, i + len).join(' ');
      const res = matchPhrase(chunk, classes, len === 1 ? 0.88 : 0.8, 1);
      if (res.matched && res.best) {
        const cls = classes.find((c) => c.id === res.best?.classId);
        if (cls) {
          signs.push({ text: chunk, cls });
          i += len;
          found = true;
          break;
        }
      }
    }
    if (!found) {
      unmatched.push(tokens[i]);
      i++;
    }
  }
  return { signs, unmatched };
}

export function glossOf(cls: DictionaryClass): string {
  return cls.id.toUpperCase().replace(/_/g, '-');
}
