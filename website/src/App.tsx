import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { SignAnimationPayload } from './types/lis';
import { AvatarViewport } from './components/AvatarViewport';
import { TranslatePanel, type TranslationResult } from './components/TranslatePanel';
import { SettingsPanel } from './components/SettingsPanel';
import { getSignLanguage, getSourceLanguage, useLanguageSettings } from './i18n/languages';
import { useAvatarChoice } from './avatars/avatarCatalog';
import { configuredModelUrl, HttpMotionModel, type GeneratedMotion, type MotionGenerator } from './generation/motionModel';
import { motionOutputFiles, motionToAnimation } from './generation/motionToAnimation';
import type { TransitionCurve } from './generation/stitchClips';
import { downloadText } from './utils/download';
import { BASE_URL } from './utils/baseUrl';
import { LisRouterModel, type ModelState } from './router/lisRouterModel';
import { loadBrowserEncoder } from './router/browserEncoder';
import { loadBrowserTranslator } from './router/browserTranslator';

const SEMANTIC_KEY = 'found-in-translation.semantic';
/** Semantic sentence match: on unless the user turned it off. */
const readSemanticPref = () => {
  try {
    return localStorage.getItem(SEMANTIC_KEY) !== 'off';
  } catch {
    return true;
  }
};

/**
 * Found in Translation: one page, like a text translator.
 * The phrase (translated to Italian in the browser when needed) goes to the LIS router over
 * the exported dataset (public/lis/), or to a model server (VITE_MOTION_MODEL_URL). The clips
 * are joined and the avatar signs them.
 */
export default function App() {
  const [languages, setLanguages] = useLanguageSettings();
  const signLanguage = getSignLanguage(languages.signLanguage);
  const sourceLanguage = getSourceLanguage(languages.sourceLanguage);

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

  // New languages: the last phrase and result no longer apply.
  useEffect(() => {
    setQuery('');
    setResult(null);
    setError(null);
    setLastMotion(null);
  }, [languages.sourceLanguage, languages.signLanguage]);

  // The exported dataset of the sign language (undefined while loading, null when missing).
  const [routerModel, setRouterModel] = useState<LisRouterModel | null | undefined>(undefined);
  useEffect(() => {
    let cancelled = false;
    setRouterModel(undefined);
    void LisRouterModel.load(`${BASE_URL}${signLanguage.indexPath}`).then((m) => !cancelled && setRouterModel(m));
    return () => {
      cancelled = true;
    };
  }, [signLanguage.indexPath]);

  // Model server when VITE_MOTION_MODEL_URL is set, else the LIS router.
  const generator: MotionGenerator | null = useMemo(() => {
    const url = configuredModelUrl();
    if (url) return new HttpMotionModel(url);
    return routerModel ?? null;
  }, [routerModel]);

  // Semantic sentence match (multilingual encoder, ~120 MB downloaded once).
  const [semantic, setSemantic] = useState<ModelState>({ status: 'off' });
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

  // Translator to Italian for the source language (~130 MB downloaded once, then cached).
  const [translator, setTranslator] = useState<ModelState>({ status: 'off' });
  useEffect(() => {
    const model = sourceLanguage.toItalianModel;
    if (!routerModel || !model) {
      setTranslator({ status: 'off' });
      return;
    }
    if (routerModel.hasTranslator(sourceLanguage.code)) {
      setTranslator({ status: 'on' });
      return;
    }
    let cancelled = false;
    setTranslator({ status: 'loading', fraction: null });
    loadBrowserTranslator(model, (p) => !cancelled && setTranslator({ status: 'loading', fraction: p.fraction }))
      .then((t) => {
        routerModel.setTranslator(sourceLanguage.code, t);
        if (!cancelled) setTranslator({ status: 'on' });
      })
      .catch((err) => {
        if (!cancelled) setTranslator({ status: 'error', message: err instanceof Error ? err.message : 'Cannot load the translator' });
      });
    return () => {
      cancelled = true;
    };
  }, [routerModel, sourceLanguage.code, sourceLanguage.toItalianModel]);

  /** Model output -> animation (per-sign clips are joined with the chosen transition curve). */
  const show = useCallback((text: string, motion: GeneratedMotion, curve: TransitionCurve) => {
    try {
      const anim = motionToAnimation(motion, text, curve);
      setResult({
        text,
        segments: anim ? anim.segments : motion.segments,
        unmatched: motion.unmatched,
        empty: !anim,
        italian: motion.route && motion.route.italian !== text ? motion.route.italian : null,
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

  const needsTranslator = !!routerModel && generator === routerModel && !!sourceLanguage.toItalianModel;
  const ready = !!generator && (!needsTranslator || translator.status === 'on');

  const handleTranslate = async () => {
    const text = query.trim();
    if (!text || !generator || !ready) return;
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

  // What the panel says while it cannot translate yet.
  const status =
    routerModel === undefined
      ? 'Loading the LIS dataset…'
      : !generator
        ? 'No LIS dataset in this build. Export one with npm run export:lis (see the README).'
        : needsTranslator && translator.status === 'loading'
          ? `Loading the ${sourceLanguage.label} → Italian translator${translator.fraction !== null ? ` ${Math.round(translator.fraction * 100)}%` : '…'} (once, then cached)`
          : needsTranslator && translator.status === 'error'
            ? `Translator not available: ${translator.message}`
            : null;

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
            busy={busy}
            disabled={!ready}
            status={status}
            result={result}
            error={error}
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
          modelName={generator?.name ?? 'No LIS dataset'}
          isMockModel={generator?.isMock ?? false}
          semantic={generator && generator === routerModel ? semantic : null}
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
