/**
 * Snowball Italian stemmer (https://snowballstem.org/algorithms/italian/stemmer.html).
 *
 * The browser has no spaCy lemmatizer: the router matches words through their stems, so
 * "mangio", "mangiamo", "mangiare" all reach the clip of "mangiare". Irregular forms
 * ("vado" -> "andare") are handled by a lookup table in lisRouter.ts.
 */

const VOWELS = 'aeiouàèìòù';
const isVowel = (c: string | undefined) => !!c && VOWELS.includes(c);

/** Start of the region after the first non-vowel following a vowel. */
function regionAfter(word: string, from: number): number {
  for (let i = from + 1; i < word.length; i++) {
    if (!isVowel(word[i]) && isVowel(word[i - 1])) return i + 1;
  }
  return word.length;
}

function computeRV(word: string): number {
  if (word.length < 2) return word.length;
  if (!isVowel(word[1])) {
    // Second letter is a consonant: RV starts after the next vowel.
    for (let i = 2; i < word.length; i++) if (isVowel(word[i])) return i + 1;
    return word.length;
  }
  if (isVowel(word[0]) && isVowel(word[1])) {
    // First two letters are vowels: RV starts after the next consonant.
    for (let i = 2; i < word.length; i++) if (!isVowel(word[i])) return i + 1;
    return word.length;
  }
  // Consonant-vowel: RV starts after the third letter.
  return Math.min(3, word.length);
}

const endsIn = (w: string, suffixes: string[], minStart: number): string | null => {
  let best: string | null = null;
  for (const s of suffixes) {
    if (w.endsWith(s) && w.length - s.length >= minStart && (!best || s.length > best.length)) best = s;
  }
  return best;
};

/** Longest matching suffix, regardless of region. */
const longestSuffix = (w: string, suffixes: string[]): string | null => {
  let best: string | null = null;
  for (const s of suffixes) if (w.endsWith(s) && (!best || s.length > best.length)) best = s;
  return best;
};

const STEP0_SUFFIXES = [
  'ci', 'gli', 'la', 'le', 'li', 'lo', 'mi', 'ne', 'si', 'ti', 'vi', 'sene', 'gliela', 'gliele', 'glieli', 'glielo',
  'gliene', 'mela', 'mele', 'meli', 'melo', 'mene', 'tela', 'tele', 'teli', 'telo', 'tene', 'cela', 'cele', 'celi',
  'celo', 'cene', 'vela', 'vele', 'veli', 'velo', 'vene',
];
/** Step 1 suffix classes (Snowball). */
const STEP1: Array<{ suffixes: string[]; apply: (w: string, start: number, R: { rv: number; r1: number; r2: number }) => string | null }> = [
  {
    // delete if in R2
    suffixes: [
      'anza', 'anze', 'ico', 'ici', 'ica', 'ice', 'iche', 'ichi', 'ismo', 'ismi', 'abile', 'abili', 'ibile', 'ibili',
      'ista', 'iste', 'isti', 'istà', 'istè', 'istì', 'oso', 'osi', 'osa', 'ose', 'mente', 'atrice', 'atrici', 'ante', 'anti',
    ],
    apply: (w, start, R) => (start >= R.r2 ? w.slice(0, start) : null),
  },
  {
    // delete if in R2; then a preceding "ic" if in R2
    suffixes: ['azione', 'azioni', 'atore', 'atori'],
    apply: (w, start, R) => {
      if (start < R.r2) return null;
      const out = w.slice(0, start);
      return out.endsWith('ic') && out.length - 2 >= R.r2 ? out.slice(0, -2) : out;
    },
  },
  { suffixes: ['logia', 'logie'], apply: (w, start, R) => (start >= R.r2 ? `${w.slice(0, start)}log` : null) },
  { suffixes: ['uzione', 'uzioni', 'usione', 'usioni'], apply: (w, start, R) => (start >= R.r2 ? `${w.slice(0, start)}u` : null) },
  { suffixes: ['enza', 'enze'], apply: (w, start, R) => (start >= R.r2 ? `${w.slice(0, start)}ente` : null) },
  { suffixes: ['amento', 'amenti', 'imento', 'imenti'], apply: (w, start, R) => (start >= R.rv ? w.slice(0, start) : null) },
  {
    // delete if in R1; then "iv" (+ "at") or "os"/"ic"/"abil" if in R2
    suffixes: ['amente'],
    apply: (w, start, R) => {
      if (start < R.r1) return null;
      let out = w.slice(0, start);
      if (out.endsWith('iv') && out.length - 2 >= R.r2) {
        out = out.slice(0, -2);
        if (out.endsWith('at') && out.length - 2 >= R.r2) out = out.slice(0, -2);
      } else {
        const s = endsIn(out, ['os', 'ic', 'abil'], R.r2);
        if (s) out = out.slice(0, -s.length);
      }
      return out;
    },
  },
  {
    suffixes: ['ità'],
    apply: (w, start, R) => {
      if (start < R.r2) return null;
      const out = w.slice(0, start);
      const s = endsIn(out, ['abil', 'ic', 'iv'], R.r2);
      return s ? out.slice(0, -s.length) : out;
    },
  },
  {
    suffixes: ['ivo', 'ivi', 'iva', 'ive'],
    apply: (w, start, R) => {
      if (start < R.r2) return null;
      let out = w.slice(0, start);
      if (out.endsWith('at') && out.length - 2 >= R.r2) {
        out = out.slice(0, -2);
        if (out.endsWith('ic') && out.length - 2 >= R.r2) out = out.slice(0, -2);
      }
      return out;
    },
  },
];

const STEP2_SUFFIXES = [
  'ammo', 'ando', 'ano', 'are', 'arono', 'asse', 'assero', 'assi', 'assimo', 'ata', 'ate', 'ati', 'ato', 'ava', 'avamo',
  'avano', 'avate', 'avi', 'avo', 'emmo', 'enda', 'ende', 'endi', 'endo', 'erà', 'erai', 'eranno', 'ere', 'erebbe',
  'erebbero', 'erei', 'eremmo', 'eremo', 'ereste', 'eresti', 'erete', 'erò', 'erono', 'essero', 'ete', 'eva', 'evamo',
  'evano', 'evate', 'evi', 'evo', 'iamo', 'immo', 'irà', 'irai', 'iranno', 'ire', 'irebbe', 'irebbero', 'irei',
  'iremmo', 'iremo', 'ireste', 'iresti', 'irete', 'irò', 'irono', 'isca', 'iscano', 'isce', 'isci', 'isco', 'iscono',
  'issero', 'ita', 'ite', 'iti', 'ito', 'iva', 'ivamo', 'ivano', 'ivate', 'ivi', 'ivo', 'ono', 'uta', 'ute', 'uti',
  'uto', 'ar', 'ir',
];

export function stemItalian(input: string): string {
  let w = input.toLowerCase();
  if (w.length < 3) return w;

  // Prelude: acute accents to grave, qu -> qU, u/i between vowels -> U/I.
  w = w.replace(/á/g, 'à').replace(/é/g, 'è').replace(/í/g, 'ì').replace(/ó/g, 'ò').replace(/ú/g, 'ù');
  w = w.replace(/qu/g, 'qU');
  const chars = w.split('');
  for (let i = 1; i < chars.length - 1; i++) {
    if (isVowel(chars[i - 1]) && isVowel(chars[i + 1])) {
      if (chars[i] === 'u') chars[i] = 'U';
      else if (chars[i] === 'i') chars[i] = 'I';
    }
  }
  w = chars.join('');

  const regions = (word: string) => {
    const r1 = regionAfter(word, 0);
    return { rv: computeRV(word), r1, r2: regionAfter(word, r1) };
  };
  let { rv, r1, r2 } = regions(w);

  // Step 0: attached pronouns after ando/endo (deleted) or ar/er/ir (replaced by e).
  const s0 = longestSuffix(w, STEP0_SUFFIXES);
  if (s0 && w.length - s0.length >= rv) {
    const stem = w.slice(0, -s0.length);
    if (/(ando|endo)$/.test(stem)) w = stem;
    else if (/(ar|er|ir)$/.test(stem)) w = `${stem}e`;
  }
  ({ rv, r1, r2 } = regions(w));

  // Step 1: standard suffix removal (longest suffix of all classes).
  let step1Done = false;
  let bestSuffix = '';
  let bestRule: (typeof STEP1)[number] | null = null;
  for (const rule of STEP1) {
    for (const suf of rule.suffixes) {
      if (w.endsWith(suf) && suf.length > bestSuffix.length) {
        bestSuffix = suf;
        bestRule = rule;
      }
    }
  }
  if (bestRule) {
    const out = bestRule.apply(w, w.length - bestSuffix.length, { rv, r1, r2 });
    if (out !== null) {
      w = out;
      step1Done = true;
    }
  }

  // Step 2: verb suffixes in RV, only when step 1 removed nothing.
  if (!step1Done) {
    ({ rv } = regions(w));
    const s2 = endsIn(w, STEP2_SUFFIXES, rv);
    if (s2) w = w.slice(0, -s2.length);
  }

  // Step 3a: final vowel in RV, then a preceding i in RV.
  ({ rv } = regions(w));
  if (/[aeioàèìò]$/.test(w) && w.length - 1 >= rv) {
    w = w.slice(0, -1);
    if (w.endsWith('i') && w.length - 1 >= rv) w = w.slice(0, -1);
  }
  // Step 3b: ch -> c, gh -> g in RV.
  ({ rv } = regions(w));
  if (/(ch|gh)$/.test(w) && w.length - 1 >= rv) w = w.slice(0, -1);

  return w.replace(/I/g, 'i').replace(/U/g, 'u');
}
