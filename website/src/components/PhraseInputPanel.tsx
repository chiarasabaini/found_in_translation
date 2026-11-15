import React from 'react';
import { Send, AlertCircle, CheckCircle2, Download, Loader2 } from 'lucide-react';
import type { DictionaryClass, MatchResult, SignAnimationPayload } from '../types/lis';
import { getSourceLanguage } from '../i18n/languages';
import type { GeneratedMotion } from '../generation/motionModel';
import { motionOutputFiles } from '../generation/motionToAnimation';
import type { TransitionCurve } from '../generation/stitchClips';

export type InterpreterMode = 'model' | 'dictionary';

/** Dictionary mode: several dictionary signs found in the phrase, played joined. */
export interface DictionarySequence {
  signs: Array<{ text: string; phrase: string; gloss: string; classId: string; startSec: number; endSec: number }>;
  /** Words that match no dictionary sign. */
  unmatched: string[];
}

export interface GenerationResult {
  text: string;
  motion: GeneratedMotion;
}

interface PhraseInputPanelProps {
  query: string;
  onChangeQuery: (value: string) => void;
  onSubmitQuery: (textOverride?: string) => void;
  matchResult: MatchResult | null;
  dictSequence: DictionarySequence | null;
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
  /** Curve of the transitions between per-sign clips. */
  transitionCurve: TransitionCurve;
  onChangeTransitionCurve: (curve: TransitionCurve) => void;
}

const MODES: ReadonlyArray<readonly [InterpreterMode, string]> = [
  ['model', 'Motion model'],
  ['dictionary', 'Dictionary'],
];

const TransitionCurveSelect: React.FC<{
  id: string;
  value: TransitionCurve;
  onChange: (curve: TransitionCurve) => void;
}> = ({ id, value, onChange }) => (
  <div className="mt-2 flex items-center gap-2 text-xs text-slate-500">
    <label htmlFor={id}>Transitions between signs</label>
    <select
      id={id}
      value={value}
      onChange={(e) => onChange(e.target.value as TransitionCurve)}
      className="rounded-md border border-slate-300 bg-white px-1.5 py-0.5 text-xs text-slate-900 cursor-pointer"
    >
      <option value="quintic">Quintic (minimum jerk)</option>
      <option value="cubic">Cubic Hermite</option>
    </select>
  </div>
);

export const PhraseInputPanel: React.FC<PhraseInputPanelProps> = ({
  query,
  onChangeQuery,
  onSubmitQuery,
  matchResult,
  dictSequence,
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
  transitionCurve,
  onChangeTransitionCurve,
}) => {
  const source = getSourceLanguage(sourceLanguage);
  const outputFiles = generation ? motionOutputFiles(generation.motion) : [];
  const hasClips = !!generation?.motion.clips?.length;
  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmitQuery();
  };

  return (
    <aside className="w-full lg:w-[380px] shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 flex flex-col overflow-y-auto">
      <div className="p-5">
        <h1 className="font-display text-xl font-semibold tracking-tight text-slate-900">
          Translate into {signLanguage}
        </h1>
        <div className="mt-3 flex items-center bg-slate-100 p-0.5 rounded-lg w-fit" role="group" aria-label="Interpreter mode">
            {MODES.map(([m, label]) => (
              <button
                key={m}
                type="button"
                onClick={() => onChangeMode(m)}
                aria-pressed={mode === m}
                title={
                  m === 'model'
                    ? 'The motion model generates the signs from any phrase'
                    : 'Play the stored clip of a dictionary phrase'
                }
                className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                  mode === m ? 'bg-white text-slate-900 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {label}
              </button>
            ))}
        </div>

        <form onSubmit={handleFormSubmit} className="mt-3">
          <label htmlFor="phrase-input" className="sr-only">
            Phrase ({source.label})
          </label>
          <div className="relative">
            <textarea
              id="phrase-input"
              rows={3}
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
              className="w-full rounded-xl border border-slate-300 bg-white pl-3.5 pr-14 py-2.5 text-sm text-slate-900 placeholder:text-slate-400 focus:border-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-600/15 resize-none"
            />
            <button
              type="submit"
              disabled={generating || !query.trim()}
              aria-label={generating ? 'Generating' : 'Sign phrase'}
              title="Sign phrase (Enter)"
              className="absolute right-2 bottom-3 inline-flex items-center justify-center w-9 h-9 text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-40 rounded-lg transition-colors cursor-pointer"
            >
              {generating ? (
                <Loader2 className="w-4 h-4 animate-spin" aria-hidden="true" />
              ) : (
                <Send className="w-4 h-4" aria-hidden="true" />
              )}
            </button>
          </div>
          <p className="mt-1.5 text-xs text-slate-400">
            {source.label} → {signLanguage} · Enter to sign, Shift+Enter for a new line
          </p>
        </form>

        {/* Model result */}
        {mode === 'model' && generation && (
          <div className="mt-4 space-y-2" aria-live="polite">
            {generation.motion.segments.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5">
                {generation.motion.segments.map((s, i) => (
                  <span
                    key={i}
                    title={`"${s.text}" · ${s.startSec.toFixed(2)}–${s.endSec.toFixed(2)} s`}
                    className="px-2 py-0.5 rounded-md text-xs font-mono font-semibold bg-slate-50 border border-slate-200 text-slate-800"
                  >
                    {s.gloss}
                  </span>
                ))}
              </div>
            )}
            {generation.motion.unmatched.length > 0 && (
              <p className="text-xs text-amber-800">
                Not signed (no {signLanguage} sign yet): {generation.motion.unmatched.join(', ')}
              </p>
            )}
            {outputFiles.length === 0 && (
              <p className="text-xs text-amber-800">The model produced no motion for this phrase.</p>
            )}
            {outputFiles.length > 0 && (
              <details className="group text-xs text-slate-500">
                <summary className="cursor-pointer select-none hover:text-slate-800" title={modelName}>
                  Model output files
                  {isMockModel && (
                    <span className="ml-1.5 px-1.5 py-0.5 rounded font-medium bg-amber-50 border border-amber-200 text-amber-800">
                      mock
                    </span>
                  )}
                </summary>
                {hasClips && (
                  <TransitionCurveSelect id="transition-curve" value={transitionCurve} onChange={onChangeTransitionCurve} />
                )}
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {outputFiles.map((f) => (
                    <button
                      key={f.name}
                      type="button"
                      onClick={() => onDownloadFile(f.name, f.text)}
                      className="inline-flex items-center gap-1 px-2 py-0.5 font-mono text-slate-700 bg-white hover:bg-slate-100 border border-slate-200 rounded-md cursor-pointer"
                    >
                      <Download className="w-3 h-3" aria-hidden="true" />
                      {f.name}
                    </button>
                  ))}
                </div>
              </details>
            )}
          </div>
        )}

        {/* Dictionary sequence: several signs joined */}
        {mode === 'dictionary' && dictSequence && (
          <div className="mt-4 rounded-lg border border-slate-200 bg-slate-50/70 p-3.5 space-y-2" aria-live="polite">
            <div className="flex items-center gap-2 text-slate-900 font-medium text-sm">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" aria-hidden="true" />
              <span>{dictSequence.signs.length} signs joined</span>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              {dictSequence.signs.map((x, i) => (
                <span
                  key={i}
                  title={`"${x.text}" · ${x.startSec.toFixed(2)}–${x.endSec.toFixed(2)} s`}
                  className="px-2 py-0.5 rounded-md text-xs font-medium bg-white border border-slate-200 text-slate-800"
                >
                  {x.phrase}
                </span>
              ))}
            </div>
            {dictSequence.unmatched.length > 0 && (
              <p className="text-xs text-amber-800">Not in the dictionary: {dictSequence.unmatched.join(', ')}</p>
            )}
            <TransitionCurveSelect id="dict-transition-curve" value={transitionCurve} onChange={onChangeTransitionCurve} />
          </div>
        )}

        {/* Dictionary match feedback */}
        {mode === 'dictionary' && !dictSequence && matchResult && matchResult.normalizedQuery && (
          <div className="mt-4" aria-live="polite">
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
                  {activeAnimation?.source?.kind === 'recording' && (
                    <span className="px-1.5 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800">
                      real recording
                    </span>
                  )}
                </div>
                {activeAnimation?.description && (
                  <p className="mt-2 text-xs text-slate-600 leading-relaxed">{activeAnimation.description}</p>
                )}
              </div>
            ) : (
              <div className="rounded-lg border border-amber-200 bg-amber-50/70 p-3.5">
                <div className="flex items-center gap-2 text-amber-900 font-medium text-sm">
                  <AlertCircle className="w-4 h-4 text-amber-600 shrink-0" aria-hidden="true" />
                  <span>Phrase not in the dictionary</span>
                </div>
                <p className="mt-1 text-xs text-amber-800">Closest phrases:</p>
                <div className="mt-2 flex flex-col gap-1.5">
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

        {loadingAnim && <p className="mt-3 text-xs text-slate-500">Loading the 3D track…</p>}
      </div>

      {/* Dictionary phrases: only in dictionary mode */}
      {mode === 'dictionary' && (
        <div className="px-5 pb-5 flex-1">
          <div className="flex items-center justify-between mb-2.5">
            <h2 className="font-display text-sm font-semibold text-slate-900">Dictionary phrases</h2>
            <span className="text-xs text-slate-500 font-mono tabular-nums">{classes.length} signs</span>
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
                  title={cls.description}
                  className={`w-full px-3.5 py-1.5 text-left flex items-center justify-between gap-3 transition-colors cursor-pointer ${
                    isActive ? 'bg-blue-50/80 text-blue-950' : 'bg-white hover:bg-slate-50 text-slate-800'
                  }`}
                >
                  <div className="min-w-0">
                    <div className="text-sm font-medium truncate">{cls.phrase}</div>
                    <div className="text-xs text-slate-500 truncate">
                      {cls.category} · {cls.twoHanded ? '2 hands' : '1 hand'}
                      {recordedIds?.has(cls.id) && <span className="text-emerald-700"> · recorded</span>}
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
      )}
    </aside>
  );
};
