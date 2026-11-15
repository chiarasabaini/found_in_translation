import React, { useEffect, useRef, useState } from 'react';
import { Download, Settings2, Upload, X } from 'lucide-react';
import { BUILTIN_AVATARS, DEFAULT_AVATAR_ID, type AvatarChoice } from '../avatars/avatarCatalog';
import type { TransitionCurve } from '../generation/stitchClips';
import type { RecordingFileInput } from '../recording/rokokoRecording';

const FILE_OPTION = '__file__';

interface SettingsMenuProps {
  avatar: AvatarChoice;
  onChangeAvatar: (next: AvatarChoice) => void;
  /** Format, bones and face rig of the current avatar. */
  avatarInfo: string;
  transitionCurve: TransitionCurve;
  onChangeTransitionCurve: (curve: TransitionCurve) => void;
  modelName: string;
  isMockModel: boolean;
  /** Files of the last model output (recording format). */
  outputFiles: RecordingFileInput[];
  onDownloadFile: (file: RecordingFileInput) => void;
}

const fieldSelect =
  'w-full rounded-lg border border-slate-300 bg-white px-2.5 py-1.5 text-sm text-slate-900 focus:border-blue-600 focus:outline-none cursor-pointer';

/** Gear button with avatar, transition and model-output settings. */
export const SettingsMenu: React.FC<SettingsMenuProps> = ({
  avatar,
  onChangeAvatar,
  avatarInfo,
  transitionCurve,
  onChangeTransitionCurve,
  modelName,
  isMockModel,
  outputFiles,
  onDownloadFile,
}) => {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  // Last local file, kept in the list so the user can switch back to it.
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  useEffect(() => {
    if (!open) return;
    const onDown = (e: PointerEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('pointerdown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const handleSelectAvatar = (value: string) => {
    if (value === FILE_OPTION) {
      if (uploadedFile) onChangeAvatar({ kind: 'file', file: uploadedFile });
      return;
    }
    onChangeAvatar({ kind: 'builtin', id: value });
  };

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        aria-label="Settings"
        title="Settings"
        className="inline-flex items-center justify-center w-10 h-10 rounded-full bg-white/90 hover:bg-white border border-slate-200 text-slate-700 shadow-sm cursor-pointer"
      >
        {open ? <X className="w-4 h-4" aria-hidden="true" /> : <Settings2 className="w-4 h-4" aria-hidden="true" />}
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Settings"
          className="absolute right-0 mt-2 w-[min(320px,calc(100vw-2rem))] bg-white border border-slate-200 rounded-2xl shadow-xl p-4 space-y-4 text-sm"
        >
          <section className="space-y-2">
            <label htmlFor="avatar-select" className="block text-xs font-medium text-slate-500">
              Avatar
            </label>
            <select
              id="avatar-select"
              value={avatar.kind === 'file' ? FILE_OPTION : avatar.id}
              onChange={(e) => handleSelectAvatar(e.target.value)}
              className={fieldSelect}
            >
              {BUILTIN_AVATARS.map((a) => (
                <option key={a.id} value={a.id}>
                  {a.id === DEFAULT_AVATAR_ID ? `${a.label} (default)` : a.label}
                </option>
              ))}
              {uploadedFile && <option value={FILE_OPTION}>{uploadedFile.name}</option>}
            </select>
            <input
              ref={fileInputRef}
              type="file"
              accept=".vrm,.glb,.gltf"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) {
                  setUploadedFile(file);
                  onChangeAvatar({ kind: 'file', file });
                }
                e.target.value = '';
              }}
              className="sr-only"
              aria-label="Select a VRM or GLB avatar file"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-800 bg-slate-100 hover:bg-slate-200 rounded-lg cursor-pointer"
            >
              <Upload className="w-3.5 h-3.5" aria-hidden="true" />
              Load VRM / GLB
            </button>
            {avatarInfo && <p className="text-xs text-slate-500 leading-relaxed">{avatarInfo}</p>}
          </section>

          <section className="space-y-2">
            <label htmlFor="transition-curve" className="block text-xs font-medium text-slate-500">
              Transitions between signs
            </label>
            <select
              id="transition-curve"
              value={transitionCurve}
              onChange={(e) => onChangeTransitionCurve(e.target.value as TransitionCurve)}
              className={fieldSelect}
            >
              <option value="quintic">Quintic (minimum jerk)</option>
              <option value="cubic">Cubic Hermite</option>
            </select>
          </section>

          <section className="space-y-2">
            <div className="text-xs font-medium text-slate-500">
              Motion model
              {isMockModel && (
                <span className="ml-1.5 px-1.5 py-0.5 rounded font-medium bg-amber-50 border border-amber-200 text-amber-800">mock</span>
              )}
            </div>
            <p className="text-xs text-slate-600">{modelName}</p>
            {outputFiles.length > 0 && (
              <details className="text-xs text-slate-600">
                <summary className="cursor-pointer select-none hover:text-slate-900">
                  Download the last output ({outputFiles.length} files)
                </summary>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {outputFiles.map((f) => (
                  <button
                    key={f.name}
                    type="button"
                    onClick={() => onDownloadFile(f)}
                    className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono text-slate-700 bg-white hover:bg-slate-100 border border-slate-200 rounded-md cursor-pointer"
                  >
                    <Download className="w-3 h-3" aria-hidden="true" />
                    {f.name}
                  </button>
                ))}
              </div>
              </details>
            )}
          </section>
        </div>
      )}
    </div>
  );
};
