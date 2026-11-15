import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Download,
  FolderOpen,
  Info,
  Play,
  Save,
  Trash2,
  Upload,
  XCircle,
} from 'lucide-react';
import type { DictionaryClass, SignAnimationPayload } from '../types/lis';
import { AvatarViewport } from './AvatarViewport';
import {
  parseRokokoRecording,
  type RecordingFileInput,
  type RokokoRecording,
} from '../recording/rokokoRecording';
import {
  DEFAULT_RETARGET_OPTIONS,
  detectSigningSegments,
  estimateGloveYawDeg,
  retargetRecording,
} from '../recording/recordingRetarget';
import { downloadJson } from '../recording/clipStore';
import type { AvatarChoice } from '../avatars/avatarCatalog';

interface RecordingStudioViewProps {
  classes: DictionaryClass[];
  recordedClips: Record<string, SignAnimationPayload>;
  onAssignClip: (cls: DictionaryClass, payload: SignAnimationPayload) => Promise<boolean>;
  onRemoveClip: (classId: string) => void;
  onPlayInInterpreter: (cls: DictionaryClass) => void;
  avatar: AvatarChoice;
  avatarError: string | null;
  onAvatarError: (err: string | null) => void;
}

type TuneOptions = typeof DEFAULT_RETARGET_OPTIONS;

/** Reads every .csv file from a drop, including files inside dropped folders. */
async function readDroppedFiles(dt: DataTransfer): Promise<File[]> {
  const entries: FileSystemEntry[] = [];
  for (const item of Array.from(dt.items)) {
    const entry = item.webkitGetAsEntry?.();
    if (entry) entries.push(entry);
  }
  if (!entries.length) return Array.from(dt.files);
  const out: File[] = [];
  const walk = async (entry: FileSystemEntry): Promise<void> => {
    if (entry.isFile) {
      const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
      out.push(file);
    } else if (entry.isDirectory) {
      const reader = (entry as FileSystemDirectoryEntry).createReader();
      // readEntries returns batches; call until empty
      for (;;) {
        const batch = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
        if (!batch.length) break;
        for (const e of batch) await walk(e);
      }
    }
  };
  for (const e of entries) await walk(e);
  return out;
}

function folderNameOf(files: File[]): string {
  const rel = (files[0] as File & { webkitRelativePath?: string })?.webkitRelativePath;
  if (rel && rel.includes('/')) return rel.split('/')[0];
  return 'recording';
}

const StreamChip: React.FC<{ ok: boolean; label: string }> = ({ ok, label }) => (
  <span
    className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium border ${
      ok ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-slate-50 border-slate-200 text-slate-400'
    }`}
  >
    {ok ? <CheckCircle2 className="w-3 h-3" aria-hidden="true" /> : <XCircle className="w-3 h-3" aria-hidden="true" />}
    {label}
  </span>
);

const Slider: React.FC<{
  id: string;
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit?: string;
  onChange: (v: number) => void;
}> = ({ id, label, value, min, max, step, unit = '', onChange }) => (
  <div>
    <div className="flex items-center justify-between text-xs">
      <label htmlFor={id} className="font-medium text-slate-700">
        {label}
      </label>
      <span className="font-mono tabular-nums text-slate-600">
        {value.toFixed(step < 1 ? 2 : 0)}
        {unit}
      </span>
    </div>
    <input
      id={id}
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      className="w-full accent-blue-600 cursor-pointer"
    />
  </div>
);

const Toggle: React.FC<{ label: string; checked: boolean; disabled?: boolean; onChange: (v: boolean) => void }> = ({
  label,
  checked,
  disabled,
  onChange,
}) => (
  <label className={`flex items-center gap-2 text-xs ${disabled ? 'text-slate-400' : 'text-slate-700'} cursor-pointer`}>
    <input
      type="checkbox"
      checked={checked && !disabled}
      disabled={disabled}
      onChange={(e) => onChange(e.target.checked)}
      className="accent-blue-600"
    />
    {label}
  </label>
);

export const RecordingStudioView: React.FC<RecordingStudioViewProps> = ({
  classes,
  recordedClips,
  onAssignClip,
  onRemoveClip,
  onPlayInInterpreter,
  avatar,
  avatarError,
  onAvatarError,
}) => {
  const folderInputRef = useRef<HTMLInputElement | null>(null);
  const filesInputRef = useRef<HTMLInputElement | null>(null);

  const [recording, setRecording] = useState<RokokoRecording | null>(null);
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [range, setRange] = useState<[number, number]>([0, 0]);
  const [tune, setTune] = useState<TuneOptions>({ ...DEFAULT_RETARGET_OPTIONS });
  const [autoYaw, setAutoYaw] = useState(0);
  const [threshold, setThreshold] = useState(0.4);
  const [targetClassId, setTargetClassId] = useState<string>('');
  const [playTrigger, setPlayTrigger] = useState(0);
  const [status, setStatus] = useState<{ kind: 'ok' | 'error'; text: string } | null>(null);
  const [preview, setPreview] = useState<SignAnimationPayload | null>(null);

  useEffect(() => {
    if (!targetClassId && classes.length) setTargetClassId(classes[0].id);
  }, [classes, targetClassId]);

  // Folder picking is a non-standard attribute that React does not type.
  useEffect(() => {
    folderInputRef.current?.setAttribute('webkitdirectory', '');
    folderInputRef.current?.setAttribute('directory', '');
  }, []);

  const targetClass = classes.find((c) => c.id === targetClassId) ?? null;

  const loadFiles = async (files: File[], name?: string) => {
    const csv = files.filter((f) => f.name.toLowerCase().endsWith('.csv'));
    if (!csv.length) {
      setStatus({ kind: 'error', text: 'No .csv file found. Select the recording folder.' });
      return;
    }
    setLoading(true);
    setStatus(null);
    try {
      const inputs: RecordingFileInput[] = await Promise.all(
        csv.map(async (f) => ({ name: f.name, text: await f.text() })),
      );
      const rec = parseRokokoRecording(inputs, name ?? folderNameOf(files));
      setRecording(rec);
      setAutoYaw(estimateGloveYawDeg(rec));
      setTune((t) => ({ ...t, gloveYawDeg: null, leftYawOffsetDeg: 0, rightYawOffsetDeg: 0 }));
      const prof = detectSigningSegments(rec, threshold);
      const first = prof.segments[0];
      setRange(first ? [first.startSec, first.endSec] : [0, rec.durationSec]);
      setPlayTrigger((n) => n + 1);
    } catch (err) {
      setStatus({ kind: 'error', text: err instanceof Error ? err.message : 'Cannot read the files' });
    } finally {
      setLoading(false);
    }
  };

  const profile = useMemo(
    () => (recording ? detectSigningSegments(recording, threshold) : null),
    [recording, threshold],
  );

  const hasError = !!recording?.diagnostics.some((d) => d.level === 'error');

  // Rebuild the preview clip (debounced) whenever the segment or calibration changes.
  useEffect(() => {
    if (!recording || hasError) {
      setPreview(null);
      return;
    }
    const handle = window.setTimeout(() => {
      try {
        const cls = targetClass;
        const { payload } = retargetRecording(recording, {
          ...tune,
          startSec: range[0],
          endSec: range[1],
          classId: cls?.id ?? 'recording',
          phrase: cls ? `${cls.phrase} · recording` : recording.name,
          category: cls?.category ?? 'Recording',
          description: cls?.description ?? '',
        });
        setPreview(payload);
      } catch (err) {
        setStatus({ kind: 'error', text: err instanceof Error ? err.message : 'Retargeting error' });
      }
    }, 160);
    return () => window.clearTimeout(handle);
  }, [recording, hasError, tune, range, targetClass]);

  const effectiveYaw = tune.gloveYawDeg ?? autoYaw;
  const duration = recording?.durationSec ?? 0;
  const assignedCount = Object.keys(recordedClips).length;

  const handleAssign = async () => {
    if (!preview || !targetClass) return;
    const payload: SignAnimationPayload = {
      ...preview,
      classId: targetClass.id,
      phrase: targetClass.phrase,
      category: targetClass.category,
      description: targetClass.description,
    };
    const durable = await onAssignClip(targetClass, payload);
    setStatus({
      kind: 'ok',
      text: durable
        ? `"${targetClass.phrase}" now uses this recording. It is saved in this browser.`
        : `"${targetClass.phrase}" uses this recording for this session only (browser storage is not available).`,
    });
  };

  const handleDownload = () => {
    if (!preview) return;
    const id = targetClass?.id ?? 'recording';
    downloadJson(`${id}.json`, { ...preview, classId: id, phrase: targetClass?.phrase ?? preview.phrase });
  };

  // Activity sparkline geometry
  const sparkW = 360;
  const sparkH = 56;
  const sparkPath = useMemo(() => {
    if (!profile || !profile.times.length || duration <= 0) return '';
    const step = Math.max(1, Math.floor(profile.times.length / 360));
    let d = '';
    for (let i = 0; i < profile.times.length; i += step) {
      const x = (profile.times[i] / duration) * sparkW;
      const y = sparkH - 4 - profile.activity[i] * (sparkH - 8);
      d += `${d ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`;
    }
    return d;
  }, [profile, duration]);

  return (
    <div className="flex-1 flex flex-col lg:flex-row overflow-hidden bg-slate-50">
      <aside className="w-full lg:w-[440px] shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 overflow-y-auto">
        <div className="p-5 border-b border-slate-100">
          <h1 className="font-display text-xl font-semibold tracking-tight text-slate-900">Glove recordings</h1>
          <p className="mt-1 text-sm text-slate-500">
            Load a Rokoko Smartgloves II recording (solved hands, 26 joints) with camera face and body tracking. Pick the segment, check the avatar and assign it to a dictionary sign.
          </p>

          {/* Drop zone */}
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={async (e) => {
              e.preventDefault();
              setDragOver(false);
              // DataTransfer is only readable synchronously inside the event handler.
              const firstItem = e.dataTransfer.items?.[0]?.webkitGetAsEntry?.();
              const folderName = firstItem?.isDirectory ? firstItem.name : undefined;
              const dropped = await readDroppedFiles(e.dataTransfer);
              await loadFiles(dropped, folderName);
            }}
            className={`mt-4 rounded-xl border-2 border-dashed p-4 text-center transition-colors ${
              dragOver ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-slate-50'
            }`}
          >
            <Upload className="w-5 h-5 mx-auto text-slate-500" aria-hidden="true" />
            <p className="mt-1.5 text-xs text-slate-600">
              Drop the folder here (e.g. <span className="font-mono">test_1_…</span>) or the files{' '}
              <span className="font-mono">left_hand.csv</span>, <span className="font-mono">right_hand.csv</span>,{' '}
              <span className="font-mono">face.csv</span>, <span className="font-mono">body.csv</span>
            </p>
            <div className="mt-3 flex flex-wrap justify-center gap-2">
              <button
                type="button"
                onClick={() => folderInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 rounded-lg cursor-pointer"
              >
                <FolderOpen className="w-3.5 h-3.5" aria-hidden="true" />
                Choose folder
              </button>
              <button
                type="button"
                onClick={() => filesInputRef.current?.click()}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white hover:bg-slate-100 border border-slate-200 rounded-lg cursor-pointer"
              >
                Choose CSV files
              </button>
            </div>
            <input
              ref={folderInputRef}
              type="file"
              multiple
              className="sr-only"
              aria-label="Select the recording folder"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                e.target.value = '';
                void loadFiles(files);
              }}
            />
            <input
              ref={filesInputRef}
              type="file"
              multiple
              accept=".csv,text/csv"
              className="sr-only"
              aria-label="Select the recording CSV files"
              onChange={(e) => {
                const files = Array.from(e.target.files ?? []);
                e.target.value = '';
                void loadFiles(files);
              }}
            />
          </div>
          {loading && <p className="mt-2 text-xs text-slate-500">Reading the CSV files…</p>}
          {status && (
            <div
              role="status"
              className={`mt-3 rounded-lg border px-3 py-2 text-xs ${
                status.kind === 'ok'
                  ? 'bg-emerald-50 border-emerald-200 text-emerald-900'
                  : 'bg-red-50 border-red-200 text-red-800'
              }`}
            >
              {status.text}
            </div>
          )}
        </div>

        {recording && (
          <>
            {/* Summary + diagnostics */}
            <section className="p-5 border-b border-slate-100 space-y-3">
              <div className="flex items-baseline justify-between gap-3">
                <h2 className="font-display text-sm font-semibold text-slate-900 truncate">{recording.name}</h2>
                <span className="font-mono text-xs text-slate-500 tabular-nums shrink-0">
                  {recording.frameCount} frames · {recording.durationSec.toFixed(1)} s ·{' '}
                  {recording.medianFrameIntervalMs > 0 ? (1000 / recording.medianFrameIntervalMs).toFixed(1) : '–'} Hz
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                <StreamChip ok={!!recording.right} label="Right hand" />
                <StreamChip ok={!!recording.left} label="Left hand" />
                <StreamChip ok={!!recording.face} label="Face" />
                <StreamChip ok={!!recording.body} label="Body" />
              </div>
              {recording.diagnostics.length > 0 && (
                <ul className="space-y-1.5">
                  {recording.diagnostics.map((d, i) => (
                    <li key={i} className="flex gap-1.5 text-xs leading-relaxed">
                      {d.level === 'error' ? (
                        <XCircle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-red-600" aria-hidden="true" />
                      ) : d.level === 'warning' ? (
                        <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0 text-amber-600" aria-hidden="true" />
                      ) : (
                        <Info className="w-3.5 h-3.5 mt-0.5 shrink-0 text-slate-400" aria-hidden="true" />
                      )}
                      <span className={d.level === 'error' ? 'text-red-800' : 'text-slate-600'}>{d.message}</span>
                    </li>
                  ))}
                </ul>
              )}
            </section>

            {!hasError && (
              <>
                {/* Segment */}
                <section className="p-5 border-b border-slate-100 space-y-3">
                  <h2 className="font-display text-sm font-semibold text-slate-900">Segment</h2>
                  <p className="text-xs text-slate-500">
                    Activity = raised forearm. Detected segments are a suggestion: adjust start and end to the sign.
                  </p>
                  <svg
                    viewBox={`0 0 ${sparkW} ${sparkH}`}
                    className="w-full h-14 rounded-md bg-slate-50 border border-slate-200 cursor-pointer"
                    role="img"
                    aria-label="Hand activity over time"
                    onClick={(e) => {
                      if (duration <= 0) return;
                      const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
                      const t = ((e.clientX - rect.left) / rect.width) * duration;
                      const seg = profile?.segments.find((s) => t >= s.startSec && t <= s.endSec);
                      if (seg) setRange([seg.startSec, seg.endSec]);
                    }}
                  >
                    {duration > 0 &&
                      profile?.segments.map((s, i) => (
                        <rect
                          key={i}
                          x={(s.startSec / duration) * sparkW}
                          y={0}
                          width={((s.endSec - s.startSec) / duration) * sparkW}
                          height={sparkH}
                          fill="#dbeafe"
                        />
                      ))}
                    {duration > 0 && (
                      <rect
                        x={(range[0] / duration) * sparkW}
                        y={1}
                        width={Math.max(1, ((range[1] - range[0]) / duration) * sparkW)}
                        height={sparkH - 2}
                        fill="none"
                        stroke="#2563eb"
                        strokeWidth={2}
                        rx={3}
                      />
                    )}
                    <line
                      x1={0}
                      x2={sparkW}
                      y1={sparkH - 4 - threshold * (sparkH - 8)}
                      y2={sparkH - 4 - threshold * (sparkH - 8)}
                      stroke="#94a3b8"
                      strokeDasharray="3 3"
                      strokeWidth={1}
                    />
                    <path d={sparkPath} fill="none" stroke="#0f172a" strokeWidth={1.4} />
                  </svg>

                  <div className="flex flex-wrap gap-1.5">
                    {profile?.segments.map((s, i) => {
                      const active = Math.abs(range[0] - s.startSec) < 1e-3 && Math.abs(range[1] - s.endSec) < 1e-3;
                      return (
                        <button
                          key={i}
                          type="button"
                          onClick={() => {
                            setRange([s.startSec, s.endSec]);
                            setPlayTrigger((n) => n + 1);
                          }}
                          aria-pressed={active}
                          className={`px-2 py-1 rounded-md text-xs font-mono tabular-nums border cursor-pointer ${
                            active
                              ? 'bg-blue-600 border-blue-600 text-white'
                              : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
                          }`}
                        >
                          {i + 1}: {s.startSec.toFixed(1)}–{s.endSec.toFixed(1)}s
                        </button>
                      );
                    })}
                    <button
                      type="button"
                      onClick={() => setRange([0, duration])}
                      className="px-2 py-1 rounded-md text-xs border bg-white border-slate-200 text-slate-700 hover:bg-slate-50 cursor-pointer"
                    >
                      All
                    </button>
                  </div>

                  <Slider
                    id="seg-start"
                    label="Start"
                    value={range[0]}
                    min={0}
                    max={duration}
                    step={0.05}
                    unit=" s"
                    onChange={(v) => setRange(([, b]) => [Math.min(v, b - 0.2), b])}
                  />
                  <Slider
                    id="seg-end"
                    label="End"
                    value={range[1]}
                    min={0}
                    max={duration}
                    step={0.05}
                    unit=" s"
                    onChange={(v) => setRange(([a]) => [a, Math.max(v, a + 0.2)])}
                  />
                  <Slider
                    id="seg-threshold"
                    label="Activity threshold (segment detection)"
                    value={threshold}
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    onChange={setThreshold}
                  />
                </section>

                {/* Calibration */}
                <section className="p-5 border-b border-slate-100 space-y-3">
                  <div className="flex items-center justify-between">
                    <h2 className="font-display text-sm font-semibold text-slate-900">Calibration</h2>
                    <button
                      type="button"
                      onClick={() => setTune({ ...DEFAULT_RETARGET_OPTIONS })}
                      className="text-xs text-blue-700 hover:underline cursor-pointer"
                    >
                      Reset
                    </button>
                  </div>
                  <p className="text-xs text-slate-500">
                    The gloves measure orientation relative to magnetic north. The horizontal rotation (yaw) aligns them with the body: the automatic estimate assumes the forearms point forward while signing.
                  </p>
                  <Slider
                    id="yaw-global"
                    label={`Glove yaw ${tune.gloveYawDeg === null ? '(auto)' : '(manual)'}`}
                    value={effectiveYaw}
                    min={-180}
                    max={180}
                    step={1}
                    unit="°"
                    onChange={(v) => setTune((t) => ({ ...t, gloveYawDeg: v }))}
                  />
                  {tune.gloveYawDeg !== null && (
                    <button
                      type="button"
                      onClick={() => setTune((t) => ({ ...t, gloveYawDeg: null }))}
                      className="text-xs text-blue-700 hover:underline cursor-pointer"
                    >
                      Back to the automatic estimate ({autoYaw.toFixed(0)}°)
                    </button>
                  )}
                  <Slider
                    id="yaw-right"
                    label="Right hand correction"
                    value={tune.rightYawOffsetDeg}
                    min={-90}
                    max={90}
                    step={1}
                    unit="°"
                    onChange={(v) => setTune((t) => ({ ...t, rightYawOffsetDeg: v }))}
                  />
                  <Slider
                    id="yaw-left"
                    label="Left hand correction"
                    value={tune.leftYawOffsetDeg}
                    min={-90}
                    max={90}
                    step={1}
                    unit="°"
                    onChange={(v) => setTune((t) => ({ ...t, leftYawOffsetDeg: v }))}
                  />
                  <Slider
                    id="smoothing"
                    label="Smoothing (0 = raw)"
                    value={tune.smoothing}
                    min={0}
                    max={1}
                    step={0.05}
                    onChange={(v) => setTune((t) => ({ ...t, smoothing: v }))}
                  />
                  <Slider
                    id="head-gain"
                    label="Head movement gain"
                    value={tune.headGain}
                    min={0}
                    max={2}
                    step={0.05}
                    onChange={(v) => setTune((t) => ({ ...t, headGain: v }))}
                  />
                  <Slider
                    id="fade"
                    label="Fade from/to rest pose"
                    value={tune.fadeSec}
                    min={0}
                    max={0.6}
                    step={0.05}
                    unit=" s"
                    onChange={(v) => setTune((t) => ({ ...t, fadeSec: v }))}
                  />
                  <div className="grid grid-cols-2 gap-2 pt-1">
                    <Toggle
                      label="Arms from body.csv"
                      checked={tune.useBodyArms}
                      disabled={!recording.body}
                      onChange={(v) => setTune((t) => ({ ...t, useBodyArms: v }))}
                    />
                    <Toggle
                      label="Torso from body.csv"
                      checked={tune.useTorso}
                      disabled={!recording.body}
                      onChange={(v) => setTune((t) => ({ ...t, useTorso: v }))}
                    />
                    <Toggle
                      label="Head from face.csv"
                      checked={tune.useHead}
                      disabled={!recording.face}
                      onChange={(v) => setTune((t) => ({ ...t, useHead: v }))}
                    />
                    <Toggle
                      label="Facial expressions"
                      checked={tune.useFaceMorphs}
                      disabled={!recording.face}
                      onChange={(v) => setTune((t) => ({ ...t, useFaceMorphs: v }))}
                    />
                  </div>
                </section>

                {/* Assign */}
                <section className="p-5 border-b border-slate-100 space-y-3">
                  <h2 className="font-display text-sm font-semibold text-slate-900">Assign to the dictionary</h2>
                  <div>
                    <label htmlFor="assign-class" className="block text-xs font-medium text-slate-700 mb-1">
                      Sign
                    </label>
                    <select
                      id="assign-class"
                      value={targetClassId}
                      onChange={(e) => setTargetClassId(e.target.value)}
                      className="w-full rounded-lg border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-900 focus:border-blue-600 focus:outline-none"
                    >
                      {classes.map((c) => (
                        <option key={c.id} value={c.id}>
                          {c.phrase}
                          {recordedClips[c.id] ? ' · recorded' : ''}
                        </option>
                      ))}
                    </select>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <button
                      type="button"
                      onClick={() => setPlayTrigger((n) => n + 1)}
                      disabled={!preview}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 disabled:opacity-40 rounded-lg cursor-pointer"
                    >
                      <Play className="w-3.5 h-3.5" aria-hidden="true" />
                      Preview
                    </button>
                    <button
                      type="button"
                      onClick={() => void handleAssign()}
                      disabled={!preview || !targetClass}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-blue-600 hover:bg-blue-700 disabled:opacity-40 rounded-lg cursor-pointer"
                    >
                      <Save className="w-3.5 h-3.5" aria-hidden="true" />
                      Use for &quot;{targetClass?.phrase ?? '…'}&quot;
                    </button>
                    <button
                      type="button"
                      onClick={handleDownload}
                      disabled={!preview}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 disabled:opacity-40 rounded-lg cursor-pointer"
                    >
                      <Download className="w-3.5 h-3.5" aria-hidden="true" />
                      Download JSON
                    </button>
                  </div>
                  <p className="text-xs text-slate-500">
                    The JSON has the same format as <span className="font-mono">public/data/animations/&lt;id&gt;.json</span>: copy it there to make it permanent for everyone.
                  </p>
                </section>
              </>
            )}
          </>
        )}

        {/* Assigned clips */}
        <section className="p-5 space-y-2.5">
          <div className="flex items-center justify-between">
            <h2 className="font-display text-sm font-semibold text-slate-900">Recorded signs</h2>
            <span className="font-mono text-xs text-slate-500 tabular-nums">
              {assignedCount}/{classes.length}
            </span>
          </div>
          {assignedCount === 0 ? (
            <p className="text-xs text-slate-500">No sign replaced yet: the interpreter uses the pipeline animations.</p>
          ) : (
            <ul className="divide-y divide-slate-100 border border-slate-200 rounded-lg overflow-hidden">
              {classes
                .filter((c) => recordedClips[c.id])
                .map((c) => {
                  const clip = recordedClips[c.id];
                  return (
                    <li key={c.id} className="flex items-center justify-between gap-2 px-3 py-2 bg-white">
                      <div className="min-w-0">
                        <div className="text-sm font-medium text-slate-900 truncate">{c.phrase}</div>
                        <div className="font-mono text-xs text-slate-500 truncate">
                          {clip.prototypeClipId} · {clip.duration.toFixed(2)} s
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        <button
                          type="button"
                          onClick={() => onPlayInInterpreter(c)}
                          aria-label={`Play ${c.phrase} in the interpreter`}
                          className="p-1.5 rounded-md text-slate-600 hover:bg-slate-100 cursor-pointer"
                        >
                          <Play className="w-3.5 h-3.5" aria-hidden="true" />
                        </button>
                        <button
                          type="button"
                          onClick={() => downloadJson(`${c.id}.json`, clip)}
                          aria-label={`Download the JSON of ${c.phrase}`}
                          className="p-1.5 rounded-md text-slate-600 hover:bg-slate-100 cursor-pointer"
                        >
                          <Download className="w-3.5 h-3.5" aria-hidden="true" />
                        </button>
                        <button
                          type="button"
                          onClick={() => onRemoveClip(c.id)}
                          aria-label={`Remove the recording of ${c.phrase}`}
                          className="p-1.5 rounded-md text-red-600 hover:bg-red-50 cursor-pointer"
                        >
                          <Trash2 className="w-3.5 h-3.5" aria-hidden="true" />
                        </button>
                      </div>
                    </li>
                  );
                })}
            </ul>
          )}
        </section>
      </aside>

      <section aria-label="Recording avatar preview" className="flex-1 min-h-[420px] lg:h-full relative">
        <AvatarViewport
          animation={preview}
          playTrigger={playTrigger}
          avatar={avatar}
          avatarError={avatarError}
          onAvatarError={onAvatarError}
        />
      </section>
    </div>
  );
};
