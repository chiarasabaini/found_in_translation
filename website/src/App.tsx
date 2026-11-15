import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
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
import { deleteRecordedClip, downloadText, loadRecordedClips, saveRecordedClip } from './recording/clipStore';
import { getSignLanguage, localizeClass, useLanguageSettings } from './i18n/languages';
import { useAvatarChoice } from './avatars/avatarCatalog';
import type { GenerationResult, InterpreterMode } from './components/PhraseInputPanel';
import { configuredModelUrl, HttpMotionModel, type GeneratedMotion, type MotionGenerator } from './generation/motionModel';
import { motionToAnimation } from './generation/motionToAnimation';
import type { TransitionCurve } from './generation/stitchClips';
import { MockMotionModel } from './generation/mockMotionModel';

// Secondary views load on first use, so the interpreter starts faster.
const QaComparisonView = lazy(() => import('./components/QaComparisonView').then((m) => ({ default: m.QaComparisonView })));
const RecordingStudioView = lazy(() =>
  import('./components/RecordingStudioView').then((m) => ({ default: m.RecordingStudioView })),
);

const ViewFallback = () => (
  <div className="flex-1 flex items-center justify-center text-sm text-slate-500">Loading…</div>
);

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
  const interpreterModeRef = useRef(interpreterMode);
  interpreterModeRef.current = interpreterMode;
  const [generating, setGenerating] = useState(false);
  const [generation, setGeneration] = useState<GenerationResult | null>(null);
  const [transitionCurve, setTransitionCurve] = useState<TransitionCurve>('quintic');
  const transitionCurveRef = useRef(transitionCurve);
  transitionCurveRef.current = transitionCurve;
  const lastMotionRef = useRef<{ text: string; motion: GeneratedMotion } | null>(null);
  const generationAbortRef = useRef<AbortController | null>(null);

  const [avatar, setAvatar] = useAvatarChoice();
  const [avatarError, setAvatarError] = useState<string | null>(null);

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
        // Dictionary mode starts on the first sign; model mode waits for a phrase.
        if (interpreterModeRef.current === 'dictionary' && dictData.classes.length > 0) {
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
    lastMotionRef.current = null;
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

  /** Shows the animation built from a model output (stitching per-sign clips if needed). */
  const showMotion = useCallback((text: string, motion: GeneratedMotion, curve: TransitionCurve, autoPlay: boolean) => {
    const anim = motionToAnimation(motion, text, curve);
    setGeneration({ text, motion: anim ? { ...motion, segments: anim.segments } : motion });
    if (!anim) return;
    setActiveClassId(null);
    setActiveAnimation(anim.payload);
    if (autoPlay) setPlayTrigger((n) => n + 1);
  }, []);

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
      lastMotionRef.current = { text, motion };
      showMotion(text, motion, transitionCurveRef.current, true);
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return;
      setLoadError(err instanceof Error ? err.message : 'Motion generation failed');
    } finally {
      if (generationAbortRef.current === ctrl) setGenerating(false);
    }
  };

  // A new transition curve re-joins the last model output (no new request to the model).
  useEffect(() => {
    const last = lastMotionRef.current;
    if (!last?.motion.clips?.length) return;
    try {
      showMotion(last.text, last.motion, transitionCurve, true);
    } catch (err) {
      setLoadError(err instanceof Error ? err.message : 'Cannot join the model clips');
    }
  }, [transitionCurve, showMotion]);

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
        avatar={avatar}
        onChangeAvatar={(next) => {
          setAvatarError(null);
          setAvatar(next);
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
            transitionCurve={transitionCurve}
            onChangeTransitionCurve={setTransitionCurve}
          />
          <section aria-label="3D avatar viewport" className="flex-1 min-h-[420px] lg:h-full relative">
            <AvatarViewport
              animation={displayedAnimation}
              playTrigger={playTrigger}
              avatar={avatar}
              avatarError={avatarError}
              onAvatarError={setAvatarError}
            />
          </section>
        </main>
      )}

      <Suspense fallback={<ViewFallback />}>
      {viewMode === 'qa' && (
        <main className="flex-1 flex overflow-hidden">
          <QaComparisonView
            classes={classes}
            activeClassId={activeClassId}
            activeAnimation={displayedAnimation}
            qaSummary={qaSummary}
            onSelectClass={(cls) => void loadClassAnimation(cls, false)}
            avatar={avatar}
            avatarError={avatarError}
            onAvatarError={setAvatarError}
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
            avatar={avatar}
            avatarError={avatarError}
            onAvatarError={setAvatarError}
          />
        </main>
      )}

      </Suspense>
    </div>
  );
}
