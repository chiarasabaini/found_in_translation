/**
 * Language settings.
 *
 * The UI is always in English. Two choices are configurable:
 *   - source language: the language of the phrase the user types. The LIS router works on
 *     Italian; other languages are translated to Italian in the browser first
 *     (src/router/browserTranslator.ts);
 *   - sign language: the target sign language, which selects the exported dataset.
 */

import { useEffect, useState } from 'react';

export interface SourceLanguage {
  code: string;
  label: string;
  placeholder: string;
  /** Model that translates this language to Italian in the browser (none for Italian). */
  toItalianModel: string | null;
}

export interface SignLanguage {
  code: string;
  label: string;
  /** Exported dataset index, relative to the site (see scripts/export-lis-dataset.ts). */
  indexPath: string;
}

export const SOURCE_LANGUAGES: SourceLanguage[] = [
  {
    code: 'en',
    label: 'English',
    placeholder: 'Enter text, e.g. "Where do you live?"',
    toItalianModel: 'Xenova/opus-mt-en-it',
  },
  {
    code: 'it',
    label: 'Italiano',
    placeholder: 'Scrivi una frase, es. "Dove abiti?"',
    toItalianModel: null,
  },
];

export const SIGN_LANGUAGES: SignLanguage[] = [
  {
    code: 'LIS',
    label: 'LIS – Italian Sign Language',
    indexPath: 'lis/index.json',
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
