import React from 'react';
import { Play } from 'lucide-react';
import type { DictionaryClass } from '../types/lis';

interface DictionaryViewProps {
  classes: DictionaryClass[];
  onPlayInInterpreter: (cls: DictionaryClass) => void;
  /** Classes whose animation comes from a glove recording. */
  recordedIds?: Set<string>;
  /** Code of the target sign language. */
  signLanguage: string;
}

export const DictionaryView: React.FC<DictionaryViewProps> = ({
  classes,
  onPlayInInterpreter,
  recordedIds,
  signLanguage,
}) => {
  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-6 lg:p-8">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div>
            <h1 className="font-display text-2xl font-semibold text-slate-900">
              Closed {signLanguage} dictionary
            </h1>
            <p className="mt-1 text-sm text-slate-600">
              Defined in <code className="font-mono text-xs bg-slate-200/70 px-1.5 py-0.5 rounded">config/dictionary.yaml</code>: phrases and synonyms per source language, spatial anchors, and the prototype clip selected with DTW (or a real recording).
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 tabular-nums">
            {classes.length} signs · deterministic, client-side
          </span>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl overflow-hidden divide-y divide-slate-100">
          {classes.map((cls) => (
            <div
              key={cls.id}
              className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4"
            >
              <div className="space-y-1.5 min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1">
                  <h2 className="font-display text-base font-semibold text-slate-900">
                    {cls.phrase}
                  </h2>
                  <span className="text-xs text-slate-500">
                    {cls.category} · {cls.twoHanded ? 'Two-handed' : 'One-handed'} ·{' '}
                    <span className="font-mono tabular-nums">{cls.duration.toFixed(2)}s</span>
                  </span>
                  {recordedIds?.has(cls.id) && (
                    <span className="px-1.5 py-0.5 rounded-md text-xs font-medium bg-emerald-50 border border-emerald-200 text-emerald-800">
                      Real recording
                    </span>
                  )}
                </div>
                <p className="text-sm text-slate-600">{cls.description}</p>
                <div className="text-xs text-slate-500 flex flex-wrap gap-x-3 gap-y-1 pt-0.5">
                  <span>
                    <strong className="font-medium text-slate-700">Synonyms:</strong>{' '}
                    {cls.synonyms.join(', ')}
                  </span>
                  <span className="font-mono tabular-nums">
                    Right anchor [{cls.anchor.right_hand.join(', ')}] · Prototype {cls.prototypeClipId}
                  </span>
                </div>
              </div>

              <button
                type="button"
                onClick={() => onPlayInInterpreter(cls)}
                className="inline-flex items-center gap-1.5 px-4 py-2 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer self-start md:self-center"
              >
                <Play className="w-3.5 h-3.5" aria-hidden="true" />
                Play in 3D
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
