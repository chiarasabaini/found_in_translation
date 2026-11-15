/**
 * Browser port of model/lis_router.py: from Italian text to the dataset clips to play.
 *
 * Same flow as the Python router:
 *   1. whole-sentence match against the dataset sentences
 *      - exact normalized match, or
 *      - semantic match (sentence encoder, when loaded): score >= tau, margin over the best
 *        different sentence, guards on negation and numbers, or
 *      - lexical match (no encoder): character similarity >= lexicalTau, same guards and margin
 *      -> mode "sentence"
 *   2. otherwise: words in LIS order (time -> subject -> object -> verb -> negation -> question
 *      word), each looked up among the single-word clips -> mode "words".
 *      Unlike the Python router, multi-word dataset sentences found inside the text
 *      ("ciao, mi chiamo Chiara" -> "ciao" + "mi chiamo") are used as one unit first, so
 *      multi-word clips are not lost in the fallback
 *   3. nothing found -> mode "none"
 *
 * The browser has no spaCy: POS tags and lemmas come from
 *   - function-word lists (articles, prepositions, conjunctions, auxiliaries, copula),
 *   - a table of irregular verb forms,
 *   - the Snowball Italian stemmer (regular inflections),
 *   - the POS of the matched dataset word (exported with the index),
 *   - word position for subject / object (before / after the first verb, Italian SVO).
 */

import { stemItalian } from './italianStemmer';

export interface LisIndexEntry {
  id: string;
  /** Dataset sentence (metadata.json "sentence"). */
  sentence: string;
  norm: string;
  kind: 'word' | 'sentence';
  /** Lemma of a word clip (metadata.json "lemma", or the word itself). */
  lemma?: string;
  /** POS of a word clip: "VERB", "NOUN", ... (metadata.json "pos", or a guess). */
  pos?: string;
  /** Clip JSON path, relative to the index. */
  clip: string;
  duration: number;
  strokeStartSec?: number;
  strokeEndSec?: number;
}

export interface LisIndex {
  version: 1;
  /** Where the clips come from, shown in the UI ("demo", dataset name, ...). */
  source: string;
  entries: LisIndexEntry[];
}

export type RouterMode = 'sentence' | 'words' | 'none';

export interface RouterResult {
  mode: RouterMode;
  entries: LisIndexEntry[];
  /** Best sentence similarity found (1 = exact). */
  score: number;
  /** Best dataset sentence (also when rejected). */
  matchedLabel: string | null;
  /** How the sentence match was scored. */
  matcher: 'exact' | 'semantic' | 'lexical';
  /** Lemmas in LIS order ("words" mode). */
  order: string[];
  /** Lemmas without a clip. */
  skipped: string[];
  /** Face channels to overlay: "negation", "question_wh", "question_yn". */
  modifiers: string[];
}

/** Sentence encoder: unit-length embeddings, one per text. */
export interface SentenceEncoder {
  readonly name: string;
  embed(texts: string[]): Promise<Float32Array[]>;
}

export interface RouterOptions {
  tau: number;
  margin: number;
  minCoverage: number;
  /** Character similarity needed for a lexical whole-sentence match. */
  lexicalTau: number;
}

export const DEFAULT_ROUTER_OPTIONS: RouterOptions = { tau: 0.88, margin: 0.02, minCoverage: 0, lexicalTau: 0.9 };

// ---------------------------------------------------------------------------
// Italian lexicons (data: they stay in Italian)
// ---------------------------------------------------------------------------
export const NEG = new Set(['non', 'mai', 'niente', 'nulla', 'nessuno', 'nessuna', 'senza']);
export const WH = new Set(['chi', 'che', 'cosa', 'dove', 'quando', 'perché', 'perche', 'come', 'quanto', 'quale', 'quali']);
export const TIME = new Set([
  'ieri', 'oggi', 'domani', 'dopodomani', 'adesso', 'ora', 'stasera', 'stamattina', 'mattina', 'pomeriggio', 'sera', 'notte',
]);
// No "uno/una/sei": they are also an article / a verb and would give false positives.
const NUMBER_WORDS: Record<string, number> = {
  zero: 0, due: 2, tre: 3, quattro: 4, cinque: 5, sette: 7, otto: 8, nove: 9, dieci: 10, undici: 11, dodici: 12,
  tredici: 13, quattordici: 14, quindici: 15, sedici: 16, diciassette: 17, diciotto: 18, diciannove: 19, venti: 20,
  trenta: 30, quaranta: 40, cinquanta: 50,
};

/** Words spaCy tags DET, ADP, CCONJ, SCONJ, PART (dropped, LIS does not sign them). */
const FUNCTION_WORDS = new Set([
  // articles
  'il', 'lo', 'la', 'i', 'gli', 'le', 'l', 'un', 'uno', 'una', 'del', 'dello', 'della', 'dei', 'degli', 'delle',
  // prepositions (simple and articulated)
  'di', 'a', 'da', 'in', 'con', 'su', 'per', 'tra', 'fra', 'al', 'allo', 'alla', 'ai', 'agli', 'alle', 'dal', 'dallo',
  'dalla', 'dai', 'dagli', 'dalle', 'nel', 'nello', 'nella', 'nei', 'negli', 'nelle', 'col', 'coi', 'sul', 'sullo',
  'sulla', 'sui', 'sugli', 'sulle', 'd',
  // conjunctions
  'e', 'ed', 'o', 'od', 'ma', 'però', 'pero', 'anche', 'oppure', 'se', 'perciò', 'quindi', 'mentre', 'poiché', 'affinché',
  // particles and clitics
  'ci', 'ne', 'si', 'vi', 'mi', 'ti', 'c', 'm', 't', 's',
]);

/** Clitic pronouns: never signed on their own ("si" here is not "sì"). */
const CLITICS = new Set(['ci', 'ne', 'si', 'vi', 'mi', 'ti', 'c', 'm', 't', 's', 'l', 'd']);

/** Forms of "essere" (copula or auxiliary: LIS has no copula). */
const ESSERE = new Set([
  'essere', 'sono', 'sei', 'è', 'è', 'siamo', 'siete', 'ero', 'eri', 'era', 'eravamo', 'eravate', 'erano', 'sarò',
  'sarai', 'sarà', 'saremo', 'sarete', 'saranno', 'sia', 'siano', 'fossi', 'fosse', 'fossero', 'stato', 'stata', 'stati',
  'state', 'sarei', 'sarebbe', 'sarebbero',
]);
const AVERE = new Set(['ho', 'hai', 'ha', 'abbiamo', 'avete', 'hanno', 'avevo', 'avevi', 'aveva', 'avevano', 'avrò', 'avrà']);

/** Elided forms before an apostrophe ("dov'è", "com'è", "quant'è"). */
const ELISIONS: Record<string, string> = {
  dov: 'dove', com: 'come', cos: 'cosa', quand: 'quando', quant: 'quanto', qual: 'quale', nient: 'niente',
  bell: 'bello', buon: 'buono', sant: 'santo', tutt: 'tutto', quell: 'quello', quest: 'questo',
};

/** Irregular forms -> lemma (the stemmer cannot reach these). */
const IRREGULAR: Record<string, string> = {
  vado: 'andare', vai: 'andare', va: 'andare', vanno: 'andare', andiamo: 'andare', andate: 'andare', vada: 'andare',
  faccio: 'fare', fai: 'fare', fa: 'fare', facciamo: 'fare', fate: 'fare', fanno: 'fare', fatto: 'fare',
  dico: 'dire', dici: 'dire', dice: 'dire', diciamo: 'dire', dite: 'dire', dicono: 'dire', detto: 'dire',
  sto: 'stare', stai: 'stare', sta: 'stare', stiamo: 'stare', state: 'stare', stanno: 'stare',
  voglio: 'volere', vuoi: 'volere', vuole: 'volere', vogliamo: 'volere', volete: 'volere', vogliono: 'volere',
  posso: 'potere', puoi: 'potere', può: 'potere', possiamo: 'potere', potete: 'potere', possono: 'potere',
  devo: 'dovere', devi: 'dovere', deve: 'dovere', dobbiamo: 'dovere', dovete: 'dovere', devono: 'dovere',
  vengo: 'venire', vieni: 'venire', viene: 'venire', veniamo: 'venire', venite: 'venire', vengono: 'venire',
  esco: 'uscire', esci: 'uscire', esce: 'uscire', usciamo: 'uscire', uscite: 'uscire', escono: 'uscire',
  so: 'sapere', sai: 'sapere', sa: 'sapere', sappiamo: 'sapere', sapete: 'sapere', sanno: 'sapere',
  bevo: 'bere', bevi: 'bere', beve: 'bere', beviamo: 'bere', bevete: 'bere', bevono: 'bere',
  ho: 'avere', hai: 'avere', ha: 'avere', abbiamo: 'avere', avete: 'avere', hanno: 'avere',
  io: 'io', me: 'io', tu: 'tu', te: 'tu', lui: 'lui', lei: 'lei', noi: 'noi', voi: 'voi', loro: 'loro',
  mio: 'mio', mia: 'mio', miei: 'mio', mie: 'mio', tuo: 'tuo', tua: 'tuo', tuoi: 'tuo', tue: 'tuo',
};

// ---------------------------------------------------------------------------
// Text helpers (same normalization as the Python router)
// ---------------------------------------------------------------------------

export function normalizeText(text: string): string {
  return text
    .normalize('NFC')
    .toLowerCase()
    .trim()
    // "19:00", "19.00" -> "19" (times on the hour, as the translator writes them)
    .replace(/\b(\d{1,2})[:.]00\b/g, '$1')
    .replace(/[^\p{L}\p{N}_\s']/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

/** Lowercase without accents: "giù" and "giu", "sì" and "si" reach the same clip. */
export function fold(word: string): string {
  return word.normalize('NFD').replace(/\p{M}+/gu, '').normalize('NFC');
}

export function tokens(norm: string): string[] {
  return norm.match(/[\p{L}\p{N}_]+/gu) ?? [];
}

function numbersOf(norm: string): string {
  const out = new Set<number>();
  for (const t of tokens(norm)) {
    if (/^\d+$/.test(t)) out.add(Number(t));
    else if (t in NUMBER_WORDS) out.add(NUMBER_WORDS[t]);
  }
  return [...out].sort((a, b) => a - b).join(',');
}

const hasNeg = (norm: string) => tokens(norm).some((t) => NEG.has(t));

/** Embeddings confuse "vado" / "non vado" and "alle 19" / "alle 20". */
export function guardsOk(a: string, b: string): boolean {
  return hasNeg(a) === hasNeg(b) && numbersOf(a) === numbersOf(b);
}

/** Character similarity in [0, 1] (1 - normalized Levenshtein distance). */
export function charSimilarity(a: string, b: string): number {
  if (a === b) return 1;
  const n = a.length;
  const m = b.length;
  if (!n || !m) return 0;
  let prev = Array.from({ length: m + 1 }, (_, j) => j);
  for (let i = 1; i <= n; i++) {
    const cur = [i];
    for (let j = 1; j <= m; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return 1 - prev[m] / Math.max(n, m);
}

/** Guess of the POS of a dataset word when metadata.json has none: infinitives are verbs. */
export function guessPos(word: string): string {
  return /(are|ere|ire|rre)$/.test(word) && word.length >= 4 ? 'VERB' : 'NOUN';
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

interface Probe {
  idx: number;
  score: number;
  second: number;
  guards: boolean;
  matcher: RouterResult['matcher'];
}

export class LisRouter {
  readonly entries: LisIndexEntry[];
  private readonly opts: RouterOptions;
  private readonly exact = new Map<string, number>();
  /** Multi-word dataset sentences, longest first: token lists -> entry. */
  private readonly phrases: Array<{ toks: string[]; entry: LisIndexEntry }> = [];
  /** Word index: lemma, surface form and stem -> entry. */
  private readonly words = new Map<string, LisIndexEntry>();
  private readonly stems = new Map<string, LisIndexEntry>();
  private embeddings: Float32Array[] | null = null;
  private encoder: SentenceEncoder | null = null;

  constructor(index: LisIndex, options: Partial<RouterOptions> = {}) {
    this.opts = { ...DEFAULT_ROUTER_OPTIONS, ...options };
    this.entries = index.entries;
    this.entries.forEach((e, i) => {
      if (!this.exact.has(e.norm)) this.exact.set(e.norm, i);
    });
    const seenPhrase = new Set<string>();
    for (const e of this.entries) {
      const toks = tokens(e.norm);
      if (toks.length >= 2 && !seenPhrase.has(e.norm)) {
        seenPhrase.add(e.norm);
        this.phrases.push({ toks, entry: e });
      }
    }
    this.phrases.sort((a, b) => b.toks.length - a.toks.length);
    for (const e of this.entries) {
      const isWord = e.kind === 'word';
      if (!isWord) continue;
      const lemma = (e.lemma ?? e.norm).toLowerCase();
      // Same key: the first sample wins (as in the Python router).
      for (const key of [lemma, e.norm, fold(lemma), fold(e.norm)]) if (!this.words.has(key)) this.words.set(key, e);
      for (const key of [stemItalian(lemma), stemItalian(e.norm)]) if (!this.stems.has(key)) this.stems.set(key, e);
    }
  }

  /** Uses a sentence encoder for the whole-sentence match (embeds the dataset sentences once). */
  async setEncoder(encoder: SentenceEncoder | null): Promise<void> {
    if (!encoder) {
      this.encoder = null;
      this.embeddings = null;
      return;
    }
    const emb = await encoder.embed(this.entries.map((e) => e.norm));
    this.encoder = encoder;
    this.embeddings = emb;
  }

  get encoderName(): string | null {
    return this.encoder?.name ?? null;
  }

  private lexicalProbe(norm: string): Probe {
    let best = -1;
    let bestScore = -1;
    for (let i = 0; i < this.entries.length; i++) {
      const s = charSimilarity(norm, this.entries[i].norm);
      if (s > bestScore) [best, bestScore] = [i, s];
    }
    let second = -1;
    for (const e of this.entries) if (e.norm !== this.entries[best].norm) second = Math.max(second, charSimilarity(norm, e.norm));
    return { idx: best, score: bestScore, second, guards: guardsOk(norm, this.entries[best].norm), matcher: 'lexical' };
  }

  private async semanticProbe(norm: string): Promise<Probe> {
    const [q] = await this.encoder!.embed([norm]);
    const sims = this.embeddings!.map((e) => {
      let s = 0;
      for (let k = 0; k < e.length; k++) s += e[k] * q[k];
      return s;
    });
    let best = 0;
    for (let i = 1; i < sims.length; i++) if (sims[i] > sims[best]) best = i;
    let second = -1;
    sims.forEach((s, i) => {
      if (this.entries[i].norm !== this.entries[best].norm) second = Math.max(second, s);
    });
    return { idx: best, score: sims[best], second, guards: guardsOk(norm, this.entries[best].norm), matcher: 'semantic' };
  }

  private accepts(p: Probe): boolean {
    const tau = p.matcher === 'lexical' ? this.opts.lexicalTau : this.opts.tau;
    return p.guards && p.score >= tau && p.score - p.second >= this.opts.margin;
  }

  /** Lemma and clip of one word. */
  private lookup(token: string): { lemma: string; entry: LisIndexEntry | null } {
    const lemma = IRREGULAR[token] ?? (token in NUMBER_WORDS ? String(NUMBER_WORDS[token]) : token);
    const entry =
      this.words.get(token) ??
      this.words.get(lemma) ??
      this.words.get(fold(token)) ??
      this.stems.get(stemItalian(lemma)) ??
      this.stems.get(stemItalian(token)) ??
      null;
    return { lemma: entry ? (entry.lemma ?? entry.norm) : lemma, entry };
  }

  /** True when the dataset has a clip for exactly this word (function words are kept then). */
  private hasOwnClip(token: string): boolean {
    return this.words.has(token) || (!CLITICS.has(token) && this.words.has(fold(token)));
  }

  /** Words of the text in LIS order, with the face modifiers. */
  lisOrder(raw: string): { items: Array<{ lemma: string; entry: LisIndexEntry | null }>; modifiers: string[] } {
    const allToks = tokens(normalizeText(raw)).map((t) => ELISIONS[t] ?? t);
    // Multi-word dataset sentences inside the text become one unit (longest first).
    const units: Array<string | LisIndexEntry> = [];
    for (let i = 0; i < allToks.length; ) {
      const hit = this.phrases.find((p) => p.toks.every((t, k) => allToks[i + k] === t));
      if (hit) {
        units.push(hit.entry);
        i += hit.toks.length;
      } else {
        units.push(allToks[i]);
        i++;
      }
    }
    const toks = units.map((u) => (typeof u === 'string' ? u : ''));
    const endsQ = raw.trim().endsWith('?');
    type Item = { lemma: string; entry: LisIndexEntry | null };
    const time: Item[] = [];
    const subj: Item[] = [];
    const obj: Item[] = [];
    const verb: Item[] = [];
    const neg: Item[] = [];
    const wh: Item[] = [];
    let seenVerb = false;
    let negInPhrase = false;
    units.forEach((u, i) => {
      if (typeof u !== 'string') {
        // A whole phrase clip keeps its place: before the first verb with the subject, after it with the object.
        const item = { lemma: u.norm, entry: u };
        if (!seenVerb) subj.push(item);
        else obj.push(item);
        if (tokens(u.norm).some((t) => NEG.has(t)) && !neg.length) negInPhrase = true;
        return;
      }
      const t = u;
      if (WH.has(t) && (endsQ || i === 0)) return void wh.push(this.lookup(t));
      if (NEG.has(t)) return void neg.push(this.lookup(t));
      // Function words are not signed, unless the dataset recorded a sign for them ("con", "o").
      if ((FUNCTION_WORDS.has(t) && (CLITICS.has(t) || !this.hasOwnClip(t))) || ESSERE.has(t)) return;
      // "avere" before a past participle is an auxiliary ("ho mangiato").
      if (AVERE.has(t) && /(ato|uto|ito|ata|uta|ita|ati|uti|iti)$/.test(toks[i + 1] ?? '')) return;
      if (TIME.has(t)) return void time.push(this.lookup(t));
      const item = this.lookup(t);
      const isVerb = item.entry ? item.entry.pos === 'VERB' : IRREGULAR[t] !== undefined && /(are|ere|ire)$/.test(IRREGULAR[t]);
      if (isVerb) {
        seenVerb = true;
        verb.push(item);
      } else if (!seenVerb) {
        subj.push(item);
      } else {
        obj.push(item);
      }
    });
    const modifiers: string[] = [];
    if (neg.length || negInPhrase) modifiers.push('negation');
    if (wh.length) modifiers.push('question_wh');
    else if (endsQ) modifiers.push('question_yn');
    // time -> subject -> object -> verb -> negation -> question word
    return { items: [...time, ...subj, ...obj, ...verb, ...neg, ...wh], modifiers };
  }

  /**
   * Routes an Italian text. `alsoMatch`: other texts for the semantic sentence match only
   * (e.g. the original English, since the encoder is multilingual); the best accepted wins.
   */
  async translate(text: string, alsoMatch: string[] = []): Promise<RouterResult> {
    const norm = normalizeText(text);
    const empty: RouterResult = {
      mode: 'none', entries: [], score: 0, matchedLabel: null, matcher: 'lexical', order: [], skipped: [], modifiers: [],
    };
    if (!norm || !this.entries.length) return empty;

    let probe: Probe;
    if (this.exact.has(norm)) probe = { idx: this.exact.get(norm)!, score: 1, second: -1, guards: true, matcher: 'exact' };
    else if (this.encoder && this.embeddings) {
      probe = await this.semanticProbe(norm);
      for (const extra of alsoMatch) {
        const p = await this.semanticProbe(normalizeText(extra));
        // Guards compare numbers and negation with the Italian text.
        p.guards = guardsOk(norm, this.entries[p.idx].norm);
        if (this.accepts(p) && (!this.accepts(probe) || p.score > probe.score)) probe = p;
      }
    } else probe = this.lexicalProbe(norm);

    const best = this.entries[probe.idx];
    const base = { score: probe.score, matchedLabel: best.sentence, matcher: probe.matcher };
    if (probe.matcher === 'exact' || this.accepts(probe)) {
      return { ...empty, ...base, mode: 'sentence', entries: [best] };
    }

    const { items, modifiers } = this.lisOrder(text);
    const found = items.filter((x) => x.entry);
    const order = items.map((x) => x.lemma);
    const skipped = items.filter((x) => !x.entry).map((x) => x.lemma);
    const coverage = items.length ? found.length / items.length : 0;
    if (!found.length || coverage < this.opts.minCoverage) {
      return { ...empty, ...base, order, skipped, modifiers };
    }
    return {
      ...base,
      mode: 'words',
      entries: found.map((x) => x.entry!),
      order: found.map((x) => x.lemma),
      skipped,
      modifiers,
    };
  }
}
