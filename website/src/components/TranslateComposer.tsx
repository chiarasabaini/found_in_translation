import React, { useEffect, useRef } from 'react';
import { ArrowRight, ArrowUp, Loader2 } from 'lucide-react';
import { SIGN_LANGUAGES, SOURCE_LANGUAGES, getSourceLanguage, type LanguageSettings } from '../i18n/languages';
import type { GlossSegment } from '../generation/motionModel';

export interface TranslationResult {
  text: string;
  segments: GlossSegment[];
  unmatched: string[];
  /** True when the model returned no motion. */
  empty: boolean;
}

interface TranslateComposerProps {
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
  'appearance-none bg-transparent hover:bg-slate-100 rounded-full pl-3 pr-3 py-1 text-sm font-medium text-slate-800 focus:outline-none focus:ring-2 focus:ring-blue-600/20 cursor-pointer';

/** Chat-style input with translator language pickers, fixed at the bottom of the page. */
export const TranslateComposer: React.FC<TranslateComposerProps> = ({
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
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Grow with the text, up to about 6 lines.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [value]);

  const canSend = !busy && value.trim().length > 0;

  return (
    <div className="w-full flex flex-col items-center gap-2">
      {/* Last translation: the signs and the words not signed */}
      {(result || error) && (
        <div className="max-w-full flex flex-wrap items-center justify-center gap-1.5 px-2" aria-live="polite">
          {error ? (
            <span className="px-3 py-1 rounded-full text-xs bg-red-50 border border-red-200 text-red-800">{error}</span>
          ) : result?.empty ? (
            <span className="px-3 py-1 rounded-full text-xs bg-amber-50 border border-amber-200 text-amber-900">
              No sign found for “{result.text}”
            </span>
          ) : (
            <>
              {result?.segments.map((s, i) => (
                <span
                  key={i}
                  title={`“${s.text}” · ${s.startSec.toFixed(2)}–${s.endSec.toFixed(2)} s`}
                  className="px-2.5 py-0.5 rounded-full text-xs font-mono font-semibold bg-white/95 border border-slate-200 text-slate-800 shadow-sm"
                >
                  {s.gloss}
                </span>
              ))}
              {!!result?.unmatched.length && (
                <span className="px-2.5 py-0.5 rounded-full text-xs bg-amber-50/95 border border-amber-200 text-amber-900">
                  Not signed: {result.unmatched.join(', ')}
                </span>
              )}
            </>
          )}
        </div>
      )}

      <form
        onSubmit={(e) => {
          e.preventDefault();
          if (canSend) onSubmit();
        }}
        className="w-full bg-white border border-slate-200 rounded-3xl shadow-lg shadow-slate-900/5 px-3 pt-2 pb-2.5"
      >
        {/* Languages */}
        <div className="flex items-center gap-1" role="group" aria-label="Translation languages">
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
          >
            {SIGN_LANGUAGES.map((l) => (
              <option key={l.code} value={l.code}>
                {l.label}
              </option>
            ))}
          </select>
        </div>

        {/* Text */}
        <div className="flex items-end gap-2 pl-3">
          <label htmlFor="phrase-input" className="sr-only">
            Text to translate
          </label>
          <textarea
            id="phrase-input"
            ref={textareaRef}
            rows={1}
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
            className="flex-1 min-w-0 resize-none bg-transparent py-2 text-base text-slate-900 placeholder:text-slate-400 focus:outline-none"
          />
          <button
            type="submit"
            disabled={!canSend}
            aria-label={busy ? 'Translating' : 'Translate'}
            title="Translate (Enter)"
            className="mb-0.5 inline-flex items-center justify-center w-9 h-9 rounded-full text-white bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 transition-colors cursor-pointer shrink-0"
          >
            {busy ? <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" /> : <ArrowUp className="w-4 h-4" aria-hidden="true" />}
          </button>
        </div>
      </form>
    </div>
  );
};
