import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { DictionaryPayload, SignAnimationPayload } from './types/lis';
import { AvatarViewport } from './components/AvatarViewport';
import { TranslatePanel, type TranslationResult } from './components/TranslatePanel';
import { SettingsPanel } from './components/SettingsPanel';
import { getSignLanguage, localizeClass, useLanguageSettings } from './i18n/languages';
import { useAvatarChoice } from './avatars/avatarCatalog';
import { configuredModelUrl, HttpMotionModel, type GeneratedMotion, type MotionGenerator } from './generation/motionModel';
import { MockMotionModel } from './generation/mockMotionModel';
import { motionOutputFiles, motionToAnimation } from './generation/motionToAnimation';
import type { TransitionCurve } from './generation/stitchClips';
import { downloadText } from './utils/download';
import { BASE_URL } from './utils/baseUrl';
import { LisRouterModel, type SemanticState } from './router/lisRouterModel';
import { loadBrowserEncoder } from './router/browserEncoder';


const SEMANTIC_KEY = 'found-in-translation.semantic';
const readSemanticPref = () => {
  try {
    return localStorage.getItem(SEMANTIC_KEY) === 'on';
  } catch {
    return false;
  }
};

/**
 * Found in Translation: one page, like a text translator.
 * The phrase goes to the motion generator: the LIS router over the exported dataset
 * (public/lis/), a model server (VITE_MOTION_MODEL_URL), or the in-browser mock. Its output
 * (one clip per sign, or one recording) is joined and the avatar signs it.
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

  const [showSkeleton, setShowSkeleton] = useState(false);

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

  // The exported dataset (public/lis/index.json), if the site has one.
  const [routerModel, setRouterModel] = useState<LisRouterModel | null | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    void LisRouterModel.load(BASE_URL).then((m) => !cancelled && setRouterModel(m));
    return () => {
      cancelled = true;
    };
  }, []);
  useEffect(() => {
    if (!routerModel || !dictionary) return;
    const base = dictionary.metadata.base_language ?? 'it';
    routerModel.setDictionary(
      dictionary.classes.map((c) => localizeClass(c, base, base)),
      (code) => dictionary.classes.map((c) => localizeClass(c, code, base)),
    );
  }, [routerModel, dictionary]);

  // Model server when VITE_MOTION_MODEL_URL is set, else the LIS router, else the mock.
  const generator: MotionGenerator = useMemo(() => {
    const url = configuredModelUrl();
    if (url) return new HttpMotionModel(url);
    if (routerModel) return routerModel;
    return new MockMotionModel(classes);
  }, [classes, routerModel]);

  // Optional semantic sentence match for the router (encoder downloaded once, ~120 MB).
  const [semantic, setSemantic] = useState<SemanticState>({ status: 'off' });
  const enableSemantic = useCallback(
    async (on: boolean) => {
      if (!routerModel) return;
      try {
        localStorage.setItem(SEMANTIC_KEY, on ? 'on' : 'off');
      } catch {
        // storage unavailable
      }
      if (!on) {
        await routerModel.setEncoder(null);
        setSemantic({ status: 'off' });
        return;
      }
      setSemantic({ status: 'loading', fraction: null });
      try {
        const encoder = await loadBrowserEncoder((p) => setSemantic({ status: 'loading', fraction: p.fraction }));
        await routerModel.setEncoder(encoder);
        setSemantic({ status: 'on' });
      } catch (err) {
        setSemantic({ status: 'error', message: err instanceof Error ? err.message : 'Cannot load the encoder' });
      }
    },
    [routerModel],
  );
  useEffect(() => {
    if (routerModel && readSemanticPref()) void enableSemantic(true);
  }, [routerModel, enableSemantic]);

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
    <div className="h-dvh w-screen flex flex-col lg:flex-row overflow-hidden bg-[#E8EEF5] text-slate-900">
      {/* Left column: title, translator, settings */}
      <aside className="order-last lg:order-first w-full lg:w-[380px] shrink-0 bg-white border-t lg:border-t-0 lg:border-r border-slate-200 flex flex-col max-h-[55dvh] lg:max-h-none overflow-y-auto">
        <header className="px-5 pt-5 pb-3">
          <h1 className="font-display text-xl font-semibold tracking-tight text-slate-900">Found in Translation</h1>
          <p className="mt-0.5 text-sm text-slate-500">Text to Italian Sign Language</p>
        </header>
        <div className="px-5 pb-5 flex-1">
          <TranslatePanel
            languages={languages}
            onChangeLanguages={setLanguages}
            value={query}
            onChange={setQuery}
            onSubmit={() => void handleTranslate()}
            busy={busy || !dictionary || routerModel === undefined}
            result={result}
            error={error ?? loadError}
          />
        </div>
        <SettingsPanel
          avatar={avatar}
          onChangeAvatar={(next) => {
            setAvatarError(null);
            setAvatar(next);
          }}
          avatarInfo={avatarInfo}
          showSkeleton={showSkeleton}
          onToggleSkeleton={setShowSkeleton}
          transitionCurve={transitionCurve}
          onChangeTransitionCurve={handleChangeCurve}
          modelName={generator.name}
          isMockModel={generator.isMock}
          semantic={generator === routerModel ? semantic : null}
          onToggleSemantic={(on) => void enableSemantic(on)}
          lastRoute={lastMotion?.motion.route ?? null}
          outputFiles={lastMotion ? motionOutputFiles(lastMotion.motion) : []}
          onDownloadFile={(f) => downloadText(f.name, f.text)}
        />
      </aside>

      {/* Avatar, with the player at the bottom center */}
      <main className="relative flex-1 min-h-0">
        <AvatarViewport
          animation={animation}
          playTrigger={playTrigger}
          avatar={avatar}
          avatarError={avatarError}
          onAvatarError={setAvatarError}
          bottomInset={20}
          showSkeleton={showSkeleton}
          onAvatarInfo={setAvatarInfo}
        />
      </main>
    </div>
  );
}
