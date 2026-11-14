import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type {
  DictionaryClass,
  DictionaryPayload,
  MatchResult,
  QaSummaryPayload,
  SignAnimationPayload,
} from './types/lis';
import { matchPhrase } from './utils/matcher';
import { TopBar, type AppViewMode } from './components/TopBar';
import { PhraseInputPanel } from './components/PhraseInputPanel';
import { AvatarViewport } from './components/AvatarViewport';
import { QaComparisonView } from './components/QaComparisonView';
import { DictionaryView } from './components/DictionaryView';
import { ArchitectureView } from './components/ArchitectureView';
import { RecordingStudioView } from './components/RecordingStudioView';
import { deleteRecordedClip, downloadText, loadRecordedClips, saveRecordedClip } from './recording/clipStore';
import { getSignLanguage, localizeClass, useLanguageSettings } from './i18n/languages';
import type { GenerationResult, InterpreterMode } from './components/PhraseInputPanel';
import { configuredModelUrl, HttpMotionModel, type MotionGenerator } from './generation/motionModel';
import { MockMotionModel } from './generation/mockMotionModel';
import { parseRokokoRecording } from './recording/rokokoRecording';
import { DEFAULT_RETARGET_OPTIONS, retargetRecording } from './recording/recordingRetarget';

/** Recorded clips are stored per sign language: "<sign language>:<class id>". */
const clipKey = (signLanguage: string, classId: string) => `${signLanguage}:${classId}`;

export default function App() {
  const [viewMode, setViewMode] = useState<AppViewMode>('interpreter');
  const [languages, setLanguages] = useLanguageSettings();
  const signLanguage = getSignLanguage(languages.signLanguage);

  const [dictionary, setDictionary] = useState<DictionaryPayload | null>(null);
  const [qaSummary, setQaSummary] = useState<QaSummaryPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [query, setQuery] = useState('');
  const [matchResult, setMatchResult] = useState<MatchResult | null>(null);
  const [activeClassId, setActiveClassId] = useState<string | null>(null);
  const [activeAnimation, setActiveAnimation] = useState<SignAnimationPayload | null>(null);
  const [loadingAnim, setLoadingAnim] = useState(false);
  const [playTrigger, setPlayTrigger] = useState(0);

  const [interpreterMode, setInterpreterMode] = useState<InterpreterMode>('model');
  const [generating, setGenerating] = useState(false);
  const [generation, setGeneration] = useState<GenerationResult | null>(null);
  const generationAbortRef = useRef<AbortController | null>(null);

  const [customAvatarFile, setCustomAvatarFile] = useState<File | null>(null);
  const [customAvatarError, setCustomAvatarError] = useState<string | null>(null);

  const animCacheRef = useRef<Map<string, SignAnimationPayload>>(new Map());

  // Clips recorded with the gloves replace the pipeline animation of their class.
  const [allRecordedClips, setAllRecordedClips] = useState<Record<string, SignAnimationPayload>>({});
  const allRecordedClipsRef = useRef(allRecordedClips);
  allRecordedClipsRef.current = allRecordedClips;
  const signCodeRef = useRef(signLanguage.code);
  signCodeRef.current = signLanguage.code;

  const recordedClips = useMemo(() => {
    const prefix = `${signLanguage.code}:`;
    const out: Record<string, SignAnimationPayload> = {};
    for (const [key, clip] of Object.entries(allRecordedClips)) {
      if (key.startsWith(prefix)) out[key.slice(prefix.length)] = clip;
    }
    return out;
  }, [allRecordedClips, signLanguage.code]);

  const loadClassAnimation = useCallback(async (cls: DictionaryClass, autoPlay: boolean) => {
    setActiveClassId(cls.id);
    const recorded = allRecordedClipsRef.current[clipKey(signCodeRef.current, cls.id)];
    if (recorded) {
      setActiveAnimation(recorded);
      if (autoPlay) setPlayTrigger((n) => n + 1);
      return;
    }
    const cacheKey = clipKey(signCodeRef.current, cls.id);
    const cached = animCacheRef.current.get(cacheKey);
    if (cached) {
      setActiveAnimation(cached);
      if (autoPlay) setPlayTrigger((n) => n + 1);
      return;
    }
    setLoadingAnim(true);
    try {
      const res = await fetch(cls.animationUrl);
      if (!res.ok) throw new Error(`Cannot load ${cls.animationUrl}`);
      const payload = (await res.json()) as SignAnimationPayload;
      animCacheRef.current.set(cacheKey, payload);
      setActiveAnimation(payload);
      if (autoPlay) setPlayTrigger((n) => n + 1);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Cannot load the animation');
    } finally {
      setLoadingAnim(false);
    }
  }, []);

  // Load the dictionary of the selected sign language (and the stored recordings once).
  useEffect(() => {
    let cancelled = false;
    const init = async () => {
      setLoadError(null);
      setDictionary(null);
      setActiveAnimation(null);
      setActiveClassId(null);
      setMatchResult(null);
      try {
        const [dictRes, qaRes, stored] = await Promise.all([
          fetch(signLanguage.dictionaryUrl),
          fetch(signLanguage.qaSummaryUrl),
          loadRecordedClips().catch(() => ({}) as Record<string, SignAnimationPayload>),
        ]);
        if (!dictRes.ok) throw new Error(`Dictionary not found: ${signLanguage.dictionaryUrl}`);
        const dictData = (await dictRes.json()) as DictionaryPayload;
        const qaData = qaRes.ok ? ((await qaRes.json()) as QaSummaryPayload) : null;
        if (cancelled) return;
        allRecordedClipsRef.current = stored;
        setAllRecordedClips(stored);
        setDictionary(dictData);
        setQaSummary(qaData);
        if (dictData.classes.length > 0) {
          void loadClassAnimation(dictData.classes[0], false);
        }
      } catch (err) {
        if (!cancelled) {
          setLoadError(err instanceof Error ? err.message : 'Cannot initialize the app data');
        }
      }
    };
    void init();
    return () => {
      cancelled = true;
    };
  }, [loadClassAnimation, signLanguage.dictionaryUrl, signLanguage.qaSummaryUrl]);

  // The typed phrase belongs to the previous source language: start again.
  useEffect(() => {
    setQuery('');
    setMatchResult(null);
    setGeneration(null);
  }, [languages.sourceLanguage, languages.signLanguage]);

  // Dictionary text in the source language; durations of recorded clips where present.
  const classes = useMemo(() => {
    const base = dictionary?.metadata.base_language ?? 'it';
    return (dictionary?.classes ?? []).map((c) => {
      const localized = localizeClass(c, languages.sourceLanguage, base);
      const rec = recordedClips[c.id];
      return rec ? { ...localized, duration: rec.duration, prototypeClipId: rec.prototypeClipId } : localized;
    });
  }, [dictionary, recordedClips, languages.sourceLanguage]);
  const recordedIds = useMemo(() => new Set(Object.keys(recordedClips)), [recordedClips]);

  // Animation payloads carry the base-language text: show the source-language text instead.
  const displayedAnimation = useMemo(() => {
    if (!activeAnimation) return null;
    const cls = classes.find((c) => c.id === activeAnimation.classId);
    if (!cls) return activeAnimation;
    return { ...activeAnimation, phrase: cls.phrase, category: cls.category, description: cls.description };
  }, [activeAnimation, classes]);

  const runMatch = (text: string) =>
    matchPhrase(
      text,
      classes,
      dictionary?.metadata.match_threshold ?? 0.52,
      dictionary?.metadata.max_suggestions ?? 3,
    );

  // The real model when VITE_MOTION_MODEL_URL is set, otherwise the in-browser mock.
  const generator: MotionGenerator = useMemo(() => {
    const url = configuredModelUrl();
    return url ? new HttpMotionModel(url) : new MockMotionModel(classes);
  }, [classes]);

  /** Text -> motion model -> recording-format CSV -> same import and retargeting as real recordings. */
  const handleGenerate = async (text: string) => {
    if (!text.trim()) return;
    generationAbortRef.current?.abort();
    const ctrl = new AbortController();
    generationAbortRef.current = ctrl;
    setGenerating(true);
    setLoadError(null);
    try {
      const motion = await generator.generate(
        { text, sourceLanguage: languages.sourceLanguage, signLanguage: signLanguage.code },
        ctrl.signal,
      );
      if (ctrl.signal.aborted) return;
      setGeneration({ text, motion });
      if (!motion.files.length) return;
      const rec = parseRokokoRecording(motion.files, 'generated');
      const error = rec.diagnostics.find((d) => d.level === 'error');
      if (error) throw new Error(`Model output not valid: ${error.message}`);
      const { payload } = retargetRecording(rec, {
        ...DEFAULT_RETARGET_OPTIONS,
        startSec: 0,
        endSec: rec.durationSec,
        classId: 'generated',
        phrase: text,
        category: 'Generated',
        description: motion.segments.map((s) => s.gloss).join(' · '),
      });
      payload.prototypeClipId = motion.model;
      setActiveClassId(null);
      setActiveAnimation(payload);
      setPlayTrigger((n) => n + 1);
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return;
      setLoadError(err instanceof Error ? err.message : 'Motion generation failed');
    } finally {
      if (generationAbortRef.current === ctrl) setGenerating(false);
    }
  };

  const handleSubmitQuery = (textOverride?: string) => {
    if (!dictionary) return;
    const text = textOverride !== undefined ? textOverride : query;
    if (textOverride !== undefined) setQuery(textOverride);
    if (interpreterMode === 'model') {
      void handleGenerate(text);
      return;
    }
    const res = runMatch(text);
    setMatchResult(res);
    if (res.matched && res.best) {
      const targetCls = classes.find((c) => c.id === res.best?.classId);
      if (targetCls) void loadClassAnimation(targetCls, true);
    }
  };

  const handleSelectClass = (cls: DictionaryClass) => {
    setQuery(cls.phrase);
    if (interpreterMode === 'model') {
      void handleGenerate(cls.phrase);
      return;
    }
    if (dictionary) setMatchResult(runMatch(cls.phrase));
    void loadClassAnimation(cls, true);
  };

  const handleAssignRecordedClip = async (cls: DictionaryClass, payload: SignAnimationPayload) => {
    const key = clipKey(signLanguage.code, cls.id);
    const next = { ...allRecordedClipsRef.current, [key]: payload };
    allRecordedClipsRef.current = next;
    setAllRecordedClips(next);
    if (activeClassId === cls.id) setActiveAnimation(payload);
    return saveRecordedClip(key, payload);
  };

  const handleRemoveRecordedClip = (classId: string) => {
    const key = clipKey(signLanguage.code, classId);
    const next = { ...allRecordedClipsRef.current };
    delete next[key];
    allRecordedClipsRef.current = next;
    setAllRecordedClips(next);
    void deleteRecordedClip(key);
    if (activeClassId === classId) {
      const cls = classes.find((c) => c.id === classId);
      if (cls) void loadClassAnimation(cls, false);
    }
  };

  // Other views play stored clips: show them in dictionary mode.
  const playInInterpreter = (cls: DictionaryClass) => {
    setViewMode('interpreter');
    setInterpreterMode('dictionary');
    setQuery(cls.phrase);
    if (dictionary) setMatchResult(runMatch(cls.phrase));
    void loadClassAnimation(cls, true);
  };

  return (
    <div className="h-screen w-screen flex flex-col overflow-hidden bg-slate-50 text-slate-900">
      <TopBar
        activeView={viewMode}
        onChangeView={setViewMode}
        languages={languages}
        onChangeLanguages={setLanguages}
        customAvatarName={customAvatarFile?.name ?? null}
        onUploadAvatarFile={(file) => setCustomAvatarFile(file)}
        onResetAvatar={() => {
          setCustomAvatarFile(null);
          setCustomAvatarError(null);
        }}
      />

      {loadError && (
        <div role="alert" className="bg-red-50 border-b border-red-200 px-6 py-2 text-xs text-red-800">
          {loadError}
        </div>
      )}

      {viewMode === 'interpreter' && (
        <main className="flex-1 flex flex-col lg:flex-row overflow-hidden">
          <PhraseInputPanel
            query={query}
            onChangeQuery={setQuery}
            onSubmitQuery={handleSubmitQuery}
            matchResult={matchResult}
            classes={classes}
            activeClassId={activeClassId}
            activeAnimation={displayedAnimation}
            loadingAnim={loadingAnim}
            onSelectClass={handleSelectClass}
            recordedIds={recordedIds}
            sourceLanguage={languages.sourceLanguage}
            signLanguage={signLanguage.code}
            mode={interpreterMode}
            onChangeMode={setInterpreterMode}
            modelName={generator.name}
            isMockModel={generator.isMock}
            generating={generating}
            generation={generation}
            onDownloadFile={(name, text) => downloadText(name, text)}
          />
          <section aria-label="3D avatar viewport" className="flex-1 min-h-[420px] lg:h-full relative">
            <AvatarViewport
              animation={displayedAnimation}
              playTrigger={playTrigger}
              customAvatarFile={customAvatarFile}
              customAvatarError={customAvatarError}
              onCustomAvatarError={setCustomAvatarError}
            />
          </section>
        </main>
      )}

      {viewMode === 'qa' && (
        <main className="flex-1 flex overflow-hidden">
          <QaComparisonView
            classes={classes}
            activeClassId={activeClassId}
            activeAnimation={displayedAnimation}
            qaSummary={qaSummary}
            onSelectClass={(cls) => void loadClassAnimation(cls, false)}
            customAvatarFile={customAvatarFile}
            customAvatarError={customAvatarError}
            onCustomAvatarError={setCustomAvatarError}
          />
        </main>
      )}

      {viewMode === 'dictionary' && (
        <main className="flex-1 flex overflow-hidden">
          <DictionaryView
            classes={classes}
            recordedIds={recordedIds}
            signLanguage={signLanguage.code}
            onPlayInInterpreter={playInInterpreter}
          />
        </main>
      )}

      {viewMode === 'recordings' && (
        <main className="flex-1 flex overflow-hidden">
          <RecordingStudioView
            classes={classes}
            recordedClips={recordedClips}
            onAssignClip={handleAssignRecordedClip}
            onRemoveClip={handleRemoveRecordedClip}
            onPlayInInterpreter={playInInterpreter}
            customAvatarFile={customAvatarFile}
            customAvatarError={customAvatarError}
            onCustomAvatarError={setCustomAvatarError}
          />
        </main>
      )}

      {viewMode === 'architecture' && (
        <main className="flex-1 flex overflow-hidden">
          <ArchitectureView />
        </main>
      )}
    </div>
  );
}
