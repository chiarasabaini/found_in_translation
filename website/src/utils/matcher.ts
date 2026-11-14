import type { DictionaryClass, MatchCandidate, MatchResult } from '../types/lis';

export function normalizePhrase(text: string): string {
  return text
    .toLowerCase()
    .trim()
    .normalize('NFKD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z0-9\s]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function levenshteinDistance(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  let prev = new Array<number>(n + 1);
  let curr = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost);
    }
    const tmp = prev;
    prev = curr;
    curr = tmp;
  }
  return prev[n];
}

function charNgrams(str: string, n = 3): Set<string> {
  const padded = `  ${str}  `;
  const grams = new Set<string>();
  for (let i = 0; i <= padded.length - n; i++) {
    grams.add(padded.slice(i, i + n));
  }
  return grams;
}

export function phraseSimilarity(a: string, b: string): number {
  if (!a || !b) return 0;
  if (a === b) return 1;

  const maxLen = Math.max(a.length, b.length);
  const levSim = 1 - levenshteinDistance(a, b) / Math.max(1, maxLen);

  const tokensA = new Set(a.split(' ').filter(Boolean));
  const tokensB = new Set(b.split(' ').filter(Boolean));
  let intersection = 0;
  for (const t of tokensA) {
    if (tokensB.has(t)) intersection++;
  }
  const union = new Set([...tokensA, ...tokensB]).size;
  const jaccard = union > 0 ? intersection / union : 0;

  const gramsA = charNgrams(a, 3);
  const gramsB = charNgrams(b, 3);
  let gramInter = 0;
  for (const g of gramsA) {
    if (gramsB.has(g)) gramInter++;
  }
  const cosine =
    gramsA.size > 0 && gramsB.size > 0
      ? gramInter / Math.sqrt(gramsA.size * gramsB.size)
      : 0;

  // Substring containment boost for short queries inside longer phrases (e.g. "acqua" vs "vorrei dell acqua")
  const containsBoost =
    a.length >= 3 && (b.includes(a) || a.includes(b))
      ? Math.min(a.length, b.length) / Math.max(a.length, b.length)
      : 0;

  return Math.max(levSim, 0.45 * jaccard + 0.35 * cosine + 0.2 * levSim, 0.72 * containsBoost + 0.28 * levSim);
}

export function matchPhrase(
  query: string,
  classes: DictionaryClass[],
  threshold = 0.52,
  maxSuggestions = 3,
): MatchResult {
  const normalizedQuery = normalizePhrase(query);
  if (!normalizedQuery) {
    return {
      matched: false,
      normalizedQuery,
      best: null,
      suggestions: [],
    };
  }

  const candidates: MatchCandidate[] = classes.map((cls) => {
    const normCanonical = normalizePhrase(cls.phrase);
    let bestScore = phraseSimilarity(normalizedQuery, normCanonical);
    let matchedText = cls.phrase;
    let viaSynonym = false;
    let matchType: MatchCandidate['matchType'] =
      normalizedQuery === normCanonical ? 'exact' : 'fuzzy';

    for (const syn of cls.synonyms) {
      const normSyn = normalizePhrase(syn);
      const sScore = phraseSimilarity(normalizedQuery, normSyn);
      if (sScore > bestScore) {
        bestScore = sScore;
        matchedText = syn;
        viaSynonym = true;
        matchType = normalizedQuery === normSyn ? 'synonym' : 'fuzzy';
      }
    }

    return {
      classId: cls.id,
      phrase: cls.phrase,
      category: cls.category,
      matchedText,
      score: Math.round(bestScore * 1000) / 1000,
      viaSynonym,
      matchType,
    };
  });

  candidates.sort((a, b) => b.score - a.score);
  const top = candidates[0] ?? null;
  const matched = top !== null && top.score >= threshold;

  return {
    matched,
    normalizedQuery,
    best: matched ? top : null,
    suggestions: candidates.slice(0, maxSuggestions),
  };
}
