/**
 * Language settings.
 *
 * The UI is always in English. Two choices are configurable:
 *   - source language: the language of the phrase the user types (matched against the dictionary);
 *   - sign language: the target sign language, which selects the dictionary and its animations.
 *
 * To add a sign language, add an entry to SIGN_LANGUAGES with its own dictionary.json
 * (same format as public/data/dictionary.json). To add a source language, add it to
 * SOURCE_LANGUAGES and give every dictionary class a `translations.<code>` entry.
 */

import { useEffect, useState } from 'react';
import type { DictionaryClass } from '../types/lis';

export interface SourceLanguage {
  code: string;
  label: string;
  placeholder: string;
  /** Example phrases shown as one-click buttons in the interpreter. */
  examples: string[];
}

export interface SignLanguage {
  code: string;
  label: string;
  dictionaryUrl: string;
  qaSummaryUrl: string;
}

export const SOURCE_LANGUAGES: SourceLanguage[] = [
  {
    code: 'en',
    label: 'English',
    placeholder: 'e.g. "Thanks a lot" or "Where is the bathroom?"',
    examples: [
      'Hello',
      'Thank you',
      'Hello, how are you?',
      'My name is...',
      'I need help, please',
      'Water, thank you',
      "Sorry, I don't understand",
      'Goodbye',
    ],
  },
  {
    code: 'it',
    label: 'Italiano',
    placeholder: 'es. "Grazie mille" o "Dov\'è il bagno?"',
    examples: [
      'Ciao',
      'Grazie',
      'Ciao, come stai?',
      'Mi chiamo...',
      'Aiuto, per favore',
      'Acqua, grazie',
      'Scusa, non capisco',
      'Arrivederci',
    ],
  },
];

export const SIGN_LANGUAGES: SignLanguage[] = [
  {
    code: 'LIS',
    label: 'LIS – Italian Sign Language',
    dictionaryUrl: '/data/dictionary.json',
    qaSummaryUrl: '/data/qa_summary.json',
  },
];

export interface LanguageSettings {
  sourceLanguage: string;
  signLanguage: string;
}

const STORAGE_KEY = 'lis-motion-studio.languages';
const DEFAULTS: LanguageSettings = { sourceLanguage: 'en', signLanguage: 'LIS' };

function readStored(): LanguageSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS;
    const parsed = JSON.parse(raw) as Partial<LanguageSettings>;
    return {
      sourceLanguage: SOURCE_LANGUAGES.some((l) => l.code === parsed.sourceLanguage)
        ? (parsed.sourceLanguage as string)
        : DEFAULTS.sourceLanguage,
      signLanguage: SIGN_LANGUAGES.some((l) => l.code === parsed.signLanguage)
        ? (parsed.signLanguage as string)
        : DEFAULTS.signLanguage,
    };
  } catch {
    return DEFAULTS;
  }
}

/** Language settings remembered per browser. */
export function useLanguageSettings(): [LanguageSettings, (next: LanguageSettings) => void] {
  const [settings, setSettings] = useState<LanguageSettings>(readStored);
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      // storage unavailable: keep the choice for this session only
    }
  }, [settings]);
  return [settings, setSettings];
}

export function getSourceLanguage(code: string): SourceLanguage {
  return SOURCE_LANGUAGES.find((l) => l.code === code) ?? SOURCE_LANGUAGES[0];
}

export function getSignLanguage(code: string): SignLanguage {
  return SIGN_LANGUAGES.find((l) => l.code === code) ?? SIGN_LANGUAGES[0];
}

/**
 * Returns the class with phrase, category, description and synonyms in `language`.
 * Falls back to the base fields when the dictionary has no text for that language.
 */
export function localizeClass(cls: DictionaryClass, language: string, baseLanguage = 'it'): DictionaryClass {
  if (language === baseLanguage) return cls;
  const t = cls.translations?.[language];
  if (!t) return cls;
  return {
    ...cls,
    phrase: t.phrase,
    category: t.category ?? cls.category,
    description: t.description ?? cls.description,
    synonyms: t.synonyms ?? [],
  };
}
