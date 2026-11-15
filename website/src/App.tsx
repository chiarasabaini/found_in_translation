import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DictionaryPayload, SignAnimationPayload } from './types/lis';
import { AvatarViewport } from './components/AvatarViewport';
import { TranslateComposer, type TranslationResult } from './components/TranslateComposer';
import { SettingsMenu } from './components/SettingsMenu';
import { getSignLanguage, localizeClass, useLanguageSettings } from './i18n/languages';
import { useAvatarChoice } from './avatars/avatarCatalog';
import { configuredModelUrl, HttpMotionModel, type GeneratedMotion, type MotionGenerator } from './generation/motionModel';
import { MockMotionModel } from './generation/mockMotionModel';
import { motionOutputFiles, motionToAnimation } from './generation/motionToAnimation';
import type { TransitionCurve } from './generation/stitchClips';
import { downloadText } from './utils/download';

/**
 * Found in Translation: one page, like a text translator.
 * The phrase goes to the motion model; its output (one clip per sign, or one recording) is
 * retargeted and joined, and the avatar signs it.
 */
export default function App() {
  const [languages, setLanguages] = useLanguageSettings();
  const signLanguage = getSignLanguage(languages.signLanguage);

  const [dictionary, setDictionary] = useState<DictionaryPayload | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [query, setQuery] = useState('');
  const [animation, setAnimation] = useState<SignAnimationPayload | null>(null);
  const [playTrigger, setPlayTrigger] = useState(0);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<TranslationResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lastMotion, setLastMotion] = useState<{ text: string; motion: GeneratedMotion } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [transitionCurve, setTransitionCurve] = useState<TransitionCurve>('quintic');
  const [avatar, setAvatar] = useAvatarChoice();
  const [avatarError, setAvatarError] = useState<string | null>(null);
  const [avatarInfo, setAvatarInfo] = useState('');

  // The composer height sets where the player sits.
  const composerRef = useRef<HTMLDivElement | null>(null);
  const [composerHeight, setComposerHeight] = useState(120);
  useEffect(() => {
    const el = composerRef.current;
    if (!el) return;
    const obs = new ResizeObserver(() => setComposerHeight(el.getBoundingClientRect().height));
    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  // Dictionary of the sign language: the vocabulary of the mock model.
  useEffect(() => {
    let cancelled = false;
    setDictionary(null);
    setLoadError(null);
    fetch(signLanguage.dictionaryUrl)
      .then((res) => {
        if (!res.ok) throw new Error(`Dictionary not found: ${signLanguage.dictionaryUrl}`);
        return res.json() as Promise<DictionaryPayload>;
      })
      .then((d) => !cancelled && setDictionary(d))
      .catch((err) => !cancelled && setLoadError(err instanceof Error ? err.message : 'Cannot load the app data'));
    return () => {
      cancelled = true;
    };
  }, [signLanguage.dictionaryUrl]);

  // New languages: the last phrase and result no longer apply.
  useEffect(() => {
    setQuery('');
    setResult(null);
    setError(null);
    setLastMotion(null);
  }, [languages.sourceLanguage, languages.signLanguage]);

  const classes = useMemo(() => {
    const base = dictionary?.metadata.base_language ?? 'it';
    return (dictionary?.classes ?? []).map((c) => localizeClass(c, languages.sourceLanguage, base));
  }, [dictionary, languages.sourceLanguage]);

  // The real model when VITE_MOTION_MODEL_URL is set, otherwise the in-browser mock.
  const generator: MotionGenerator = useMemo(() => {
    const url = configuredModelUrl();
    return url ? new HttpMotionModel(url) : new MockMotionModel(classes);
  }, [classes]);

  /** Model output -> animation (per-sign clips are joined with the chosen transition curve). */
  const show = useCallback((text: string, motion: GeneratedMotion, curve: TransitionCurve) => {
    try {
      const anim = motionToAnimation(motion, text, curve);
      setResult({
        text,
        segments: anim ? anim.segments : motion.segments,
        unmatched: motion.unmatched,
        empty: !anim,
      });
      setError(null);
      if (anim) {
        setAnimation(anim.payload);
        setPlayTrigger((n) => n + 1);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cannot animate the model output');
    }
  }, []);

  const handleTranslate = async () => {
    const text = query.trim();
    if (!text || !dictionary) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setBusy(true);
    setError(null);
    try {
      const motion = await generator.generate(
        { text, sourceLanguage: languages.sourceLanguage, signLanguage: signLanguage.code },
        ctrl.signal,
      );
      if (ctrl.signal.aborted) return;
      setLastMotion({ text, motion });
      show(text, motion, transitionCurve);
    } catch (err) {
      if ((err as Error)?.name === 'AbortError') return;
      setError(err instanceof Error ? err.message : 'Translation failed');
    } finally {
      if (abortRef.current === ctrl) setBusy(false);
    }
  };

  const handleChangeCurve = (curve: TransitionCurve) => {
    setTransitionCurve(curve);
    // Re-join the last output without a new model request.
    if (lastMotion?.motion.clips?.length) show(lastMotion.text, lastMotion.motion, curve);
  };

  return (
    <div className="relative h-dvh w-screen overflow-hidden bg-[#E8EEF5] text-slate-900">
      <div className="absolute inset-0">
        <AvatarViewport
          animation={animation}
          playTrigger={playTrigger}
          avatar={avatar}
          avatarError={avatarError}
          onAvatarError={setAvatarError}
          bottomInset={composerHeight + 28}
          onAvatarInfo={setAvatarInfo}
        />
      </div>

      <header className="absolute top-0 inset-x-0 z-20 pointer-events-none flex justify-center pt-4 px-16">
        <h1 className="font-display text-xl sm:text-2xl font-semibold tracking-tight text-slate-900 text-center">
          Found in Translation
        </h1>
      </header>

      <div className="absolute top-3 right-3 z-30">
        <SettingsMenu
          avatar={avatar}
          onChangeAvatar={(next) => {
            setAvatarError(null);
            setAvatar(next);
          }}
          avatarInfo={avatarInfo}
          transitionCurve={transitionCurve}
          onChangeTransitionCurve={handleChangeCurve}
          modelName={generator.name}
          isMockModel={generator.isMock}
          outputFiles={lastMotion ? motionOutputFiles(lastMotion.motion) : []}
          onDownloadFile={(f) => downloadText(f.name, f.text)}
        />
      </div>

      <div className="absolute inset-x-0 bottom-0 z-20 flex justify-center px-4 pb-4 sm:pb-6 pointer-events-none">
        <div ref={composerRef} className="w-full max-w-2xl pointer-events-auto">
          <TranslateComposer
            languages={languages}
            onChangeLanguages={setLanguages}
            value={query}
            onChange={setQuery}
            onSubmit={() => void handleTranslate()}
            busy={busy || !dictionary}
            result={result}
            error={error ?? loadError}
          />
        </div>
      </div>
    </div>
  );
}
