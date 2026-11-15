import React from 'react';
import { ArrowRight, Languages, Loader2 } from 'lucide-react';
import { SIGN_LANGUAGES, SOURCE_LANGUAGES, getSourceLanguage, type LanguageSettings } from '../i18n/languages';
import type { GlossSegment } from '../generation/motionModel';

export interface TranslationResult {
  text: string;
  segments: GlossSegment[];
  unmatched: string[];
  /** True when the model returned no motion. */
  empty: boolean;
}

interface TranslatePanelProps {
  languages: LanguageSettings;
  onChangeLanguages: (next: LanguageSettings) => void;
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  busy: boolean;
  result: TranslationResult | null;
  error: string | null;
}

const MAX_CHARS = 500;

const langSelect =
  'min-w-0 flex-1 rounded-lg border border-slate-200 bg-white px-2.5 py-1.5 text-sm font-medium text-slate-800 focus:border-blue-600 focus:outline-none cursor-pointer';

/** Translator input of the left column: languages, text, result. */
export const TranslatePanel: React.FC<TranslatePanelProps> = ({
  languages,
  onChangeLanguages,
  value,
  onChange,
  onSubmit,
  busy,
  result,
  error,
}) => {
  const source = getSourceLanguage(languages.sourceLanguage);
  const canSend = !busy && value.trim().length > 0;

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        if (canSend) onSubmit();
      }}
      className="space-y-3"
    >
      {/* Languages */}
      <div className="flex items-center gap-2" role="group" aria-label="Translation languages">
        <label htmlFor="source-language" className="sr-only">
          Translate from
        </label>
        <select
          id="source-language"
          value={languages.sourceLanguage}
          onChange={(e) => onChangeLanguages({ ...languages, sourceLanguage: e.target.value })}
          className={langSelect}
        >
          {SOURCE_LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.label}
            </option>
          ))}
        </select>
        <ArrowRight className="w-4 h-4 text-slate-400 shrink-0" aria-hidden="true" />
        <label htmlFor="sign-language" className="sr-only">
          Translate to
        </label>
        <select
          id="sign-language"
          value={languages.signLanguage}
          onChange={(e) => onChangeLanguages({ ...languages, signLanguage: e.target.value })}
          className={langSelect}
          title={SIGN_LANGUAGES.find((l) => l.code === languages.signLanguage)?.label}
        >
          {SIGN_LANGUAGES.map((l) => (
            <option key={l.code} value={l.code}>
              {l.code}
            </option>
          ))}
        </select>
      </div>

      {/* Text */}
      <div className="rounded-2xl border border-slate-200 bg-white focus-within:border-blue-600 focus-within:ring-2 focus-within:ring-blue-600/15">
        <label htmlFor="phrase-input" className="sr-only">
          Text to translate
        </label>
        <textarea
          id="phrase-input"
          rows={5}
          value={value}
          maxLength={MAX_CHARS}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              if (canSend) onSubmit();
            }
          }}
          placeholder={source.placeholder}
          lang={source.code}
          className="block w-full resize-none bg-transparent px-3.5 pt-3 pb-1 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none"
        />
        <div className="flex items-center justify-between gap-2 px-3 pb-2.5">
          <span className="text-xs text-slate-400 tabular-nums">
            {value.length} / {MAX_CHARS}
          </span>
          <button
            type="submit"
            disabled={!canSend}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-sm font-medium text-white bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 transition-colors cursor-pointer"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <Languages className="w-4 h-4" aria-hidden="true" />}
            {busy ? 'Translating…' : 'Translate'}
          </button>
        </div>
      </div>
      <p className="text-xs text-slate-400">Enter to translate · Shift+Enter for a new line</p>

      {/* Last translation: the signs and the words not signed */}
      {(result || error) && (
        <div className="space-y-2" aria-live="polite">
          {error ? (
            <p className="rounded-lg px-3 py-2 text-xs bg-red-50 border border-red-200 text-red-800">{error}</p>
          ) : result?.empty ? (
            <p className="rounded-lg px-3 py-2 text-xs bg-amber-50 border border-amber-200 text-amber-900">
              No sign found for “{result.text}”
            </p>
          ) : (
            <>
              <div className="flex flex-wrap gap-1.5">
                {result?.segments.map((s, i) => (
                  <span
                    key={i}
                    title={`“${s.text}” · ${s.startSec.toFixed(2)}–${s.endSec.toFixed(2)} s`}
                    className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-slate-50 border border-slate-200 text-slate-800"
                  >
                    {s.gloss}
                  </span>
                ))}
              </div>
              {!!result?.unmatched.length && (
                <p className="text-xs text-amber-800">Not signed: {result.unmatched.join(', ')}</p>
              )}
            </>
          )}
        </div>
      )}
    </form>
  );
};
