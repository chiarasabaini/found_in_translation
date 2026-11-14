import React from 'react';
import { Send, AlertCircle, CheckCircle2, Cpu, Download, Loader2 } from 'lucide-react';
import type { DictionaryClass, MatchResult, SignAnimationPayload } from '../types/lis';
import { getSourceLanguage } from '../i18n/languages';
import type { GeneratedMotion } from '../generation/motionModel';

export type InterpreterMode = 'model' | 'dictionary';

export interface GenerationResult {
  text: string;
  motion: GeneratedMotion;
}

interface PhraseInputPanelProps {
  query: string;
  onChangeQuery: (value: string) => void;
  onSubmitQuery: (textOverride?: string) => void;
  matchResult: MatchResult | null;
  classes: DictionaryClass[];
  activeClassId: string | null;
  activeAnimation: SignAnimationPayload | null;
  loadingAnim: boolean;
  onSelectClass: (cls: DictionaryClass) => void;
  /** Classes whose animation comes from a glove recording. */
  recordedIds?: Set<string>;
  /** Code of the language the user types in (see src/i18n/languages.ts). */
  sourceLanguage: string;
  /** Code of the target sign language. */
  signLanguage: string;
  mode: InterpreterMode;
  onChangeMode: (mode: InterpreterMode) => void;
  modelName: string;
  isMockModel: boolean;
  generating: boolean;
  generation: GenerationResult | null;
  onDownloadFile: (name: string, text: string) => void;
}

export const PhraseInputPanel: React.FC<PhraseInputPanelProps> = ({
  query,
  onChangeQuery,
  onSubmitQuery,
  matchResult,
  classes,
  activeClassId,
  activeAnimation,
  loadingAnim,
  onSelectClass,
  recordedIds,
  sourceLanguage,
  signLanguage,
  mode,
  onChangeMode,
  modelName,
  isMockModel,
  generating,
  generation,
  onDownloadFile,
}) => {
  const source = getSourceLanguage(sourceLanguage);
  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmitQuery();
  };

  return (
    <aside className="w-full lg:w-[410px] shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 flex flex-col overflow-y-auto">
      <div className="p-5 border-b border-slate-100">
        <h1 className="font-display text-xl font-semibold tracking-tight text-slate-900">
          Translate into {signLanguage}
        </h1>
        <p className="mt-1 text-sm text-slate-500">
          {mode === 'model'
            ? `Type a phrase in ${source.label}. The motion model generates the glove, body and face streams and the avatar signs them in ${signLanguage}.`
            : `Type a phrase in ${source.label}. The app matches it against the closed dictionary and plays the stored ${signLanguage} clip.`}
        </p>

        <div className="mt-3 flex items-center bg-slate-100 p-0.5 rounded-lg w-fit" role="group" aria-label="Interpreter mode">
          {(
            [
              ['model', 'Motion model'],
              ['dictionary', 'Dictionary clips'],
            ] as const
          ).map(([m, label]) => (
            <button
              key={m}
              type="button"
              onClick={() => onChangeMode(m)}
              aria-pressed={mode === m}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                mode === m ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        <form onSubmit={handleFormSubmit} className="mt-3.5 space-y-2.5">
          <div>
            <label
              htmlFor="phrase-input"
              className="block text-xs font-medium text-slate-700 mb-1"
            >
              Phrase ({source.label})
            </label>
            <textarea
              id="phrase-input"
              rows={2}
              value={query}
              onChange={(e) => onChangeQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  onSubmitQuery();
                }
              }}
              placeholder={source.placeholder}
              lang={source.code}
              className="w-full rounded-lg border border-slate-300 bg-white px-3.5 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-600/15 resize-none"
            />
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-slate-500 font-mono tabular-nums">
              {classes.length} signs · 30 fps
            </span>
            <button
              type="submit"
              disabled={generating}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-60 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {generating ? (
                <Loader2 className="w-3.5 h-3.5 animate-spin" aria-hidden="true" />
              ) : (
                <Send className="w-3.5 h-3.5" aria-hidden="true" />
              )}
              {generating ? 'Generating…' : 'Sign phrase'}
            </button>
          </div>
        </form>

        {/* Example phrases */}
        <div className="mt-3">
          <div className="text-xs text-slate-500 mb-1.5">Try an example:</div>
          <div className="flex flex-wrap gap-1.5">
            {source.examples.map((ex) => (
              <button
                key={ex}
                type="button"
                onClick={() => onSubmitQuery(ex)}
                disabled={generating}
                className="px-2.5 py-1 text-xs text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-full disabled:opacity-50 cursor-pointer"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>

        {/* Model result */}
        {mode === 'model' && generation && (
          <div className="mt-4 pt-4 border-t border-slate-100" aria-live="polite">
            <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3.5 space-y-2.5">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 text-sm font-medium text-slate-900 min-w-0">
                  <Cpu className="w-4 h-4 text-blue-600 shrink-0" aria-hidden="true" />
                  <span className="truncate">&ldquo;{generation.text}&rdquo;</span>
                </div>
                {isMockModel && (
                  <span className="shrink-0 px-1.5 py-0.5 rounded text-xs font-medium bg-amber-50 border border-amber-200 text-amber-800">
                    mock model
                  </span>
                )}
              </div>
              {generation.motion.segments.length > 0 ? (
                <div className="flex flex-wrap items-center gap-1.5">
                  {generation.motion.segments.map((s, i) => (
                    <span
                      key={i}
                      title={`"${s.text}" · ${s.startSec.toFixed(2)}–${s.endSec.toFixed(2)} s`}
                      className="px-2 py-0.5 rounded-md text-xs font-mono font-semibold bg-white border border-slate-200 text-slate-800"
                    >
                      {s.gloss}
                    </span>
                  ))}
                </div>
              ) : (
                generation.motion.files.length > 0 && (
                  <p className="text-xs text-slate-600">The model returned motion without sign segments.</p>
                )
              )}
              {generation.motion.unmatched.length > 0 && (
                <p className="text-xs text-amber-800">
                  Not signed (no sign in the {signLanguage} dictionary): {generation.motion.unmatched.join(', ')}
                </p>
              )}
              {generation.motion.files.length === 0 && (
                <p className="text-xs text-amber-800">
                  The model produced no motion for this phrase. Try one of the dictionary phrases below.
                </p>
              )}
              {generation.motion.files.length > 0 && (
                <div>
                  <div className="text-xs text-slate-500 mb-1">Model output (recording format):</div>
                  <div className="flex flex-wrap gap-1.5">
                    {generation.motion.files.map((f) => (
                      <button
                        key={f.name}
                        type="button"
                        onClick={() => onDownloadFile(f.name, f.text)}
                        className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono text-slate-700 bg-white hover:bg-slate-100 border border-slate-200 rounded-md cursor-pointer"
                      >
                        <Download className="w-3 h-3" aria-hidden="true" />
                        {f.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              <p className="text-xs text-slate-400">{modelName}</p>
            </div>
          </div>
        )}

        {/* Match Feedback Area */}
        {mode === 'dictionary' && matchResult && matchResult.normalizedQuery && (
          <div className="mt-4 pt-4 border-t border-slate-100" aria-live="polite">
            {matchResult.matched && matchResult.best ? (
              <div className="rounded-lg border border-slate-200 bg-slate-50/70 p-3.5">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-center gap-2 text-slate-900 font-medium text-sm">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" aria-hidden="true" />
                    <span>{matchResult.best.phrase}</span>
                  </div>
                  <span className="font-mono text-xs text-slate-600 tabular-nums shrink-0">
                    {Math.round(matchResult.best.score * 100)}%
                  </span>
                </div>
                <div className="mt-1.5 text-xs text-slate-500 flex flex-wrap items-center gap-x-2 gap-y-0.5">
                  <span>{matchResult.best.category}</span>
                  <span aria-hidden="true">·</span>
                  <span>
                    {matchResult.best.viaSynonym
                      ? `Synonym: "${matchResult.best.matchedText}"`
                      : `${matchResult.best.matchType === 'exact' ? 'Exact' : 'Fuzzy'} match`}
                  </span>
                  {activeAnimation && (
                    <>
                      <span aria-hidden="true">·</span>
                      <span className="font-mono tabular-nums">
                        {activeAnimation.prototypeClipId}
                      </span>
                      {activeAnimation.source?.kind === 'recording' && (
                        <span className="px-1.5 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800">
                          real recording
                        </span>
                      )}
                    </>
                  )}
                </div>
                {activeAnimation?.description && (
                  <p className="mt-2 text-xs text-slate-600 leading-relaxed">
                    {activeAnimation.description}
                  </p>
                )}
              </div>
            ) : (
              <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-3.5">
                <div className="flex items-center gap-2 text-amber-900 font-medium text-sm">
                  <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" aria-hidden="true" />
                  <span>Phrase not in the dictionary</span>
                </div>
                <p className="mt-1 text-xs text-amber-800">
                  Pick one of the {matchResult.suggestions.length} closest phrases:
                </p>
                <div className="mt-2.5 flex flex-col gap-1.5">
                  {matchResult.suggestions.map((s) => (
                    <button
                      key={s.classId}
                      type="button"
                      onClick={() => onSubmitQuery(s.phrase)}
                      className="w-full flex items-center justify-between px-3 py-2 text-left text-xs font-medium text-slate-800 bg-white hover:bg-slate-50 border border-amber-200/80 rounded-md transition-colors cursor-pointer"
                    >
                      <span className="truncate">{s.phrase}</span>
                      <span className="font-mono text-slate-500 tabular-nums shrink-0 ml-2">
                        {Math.round(s.score * 100)}%
                      </span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {loadingAnim && (
          <p className="mt-3 text-xs text-slate-500">Loading the 3D track…</p>
        )}
      </div>

      {/* Closed Dictionary Quick List */}
      <div className="p-5 flex-1">
        <div className="flex items-center justify-between mb-2.5">
          <h2 className="font-display text-sm font-semibold text-slate-900">
            Dictionary phrases
          </h2>
          <span className="text-xs text-slate-500 font-mono tabular-nums">
            {classes.length} signs
          </span>
        </div>

        <div className="divide-y divide-slate-100 border border-slate-200 rounded-lg overflow-hidden">
          {classes.map((cls) => {
            const isActive = cls.id === activeClassId;
            return (
              <button
                key={cls.id}
                type="button"
                onClick={() => onSelectClass(cls)}
                aria-pressed={isActive}
                className={`w-full px-3.5 py-1.5 text-left flex items-center justify-between gap-3 transition-colors cursor-pointer ${
                  isActive
                    ? 'bg-blue-50/80 text-blue-950'
                    : 'bg-white hover:bg-slate-50 text-slate-800'
                }`}
              >
                <div className="min-w-0">
                  <div className="text-sm font-medium truncate">{cls.phrase}</div>
                  <div className="text-xs text-slate-500 truncate">
                    {cls.category} · {cls.twoHanded ? '2 hands' : '1 hand'}
                    {recordedIds?.has(cls.id) && (
                      <span className="text-emerald-700"> · recorded</span>
                    )}
                  </div>
                </div>
                <span className="font-mono text-xs text-slate-500 tabular-nums shrink-0">
                  {cls.duration.toFixed(1)}s
                </span>
              </button>
            );
          })}
        </div>
      </div>
    </aside>
  );
};
