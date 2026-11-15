import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import {
  Play,
  Pause,
  RotateCcw,
  Bone,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
} from 'lucide-react';
import type { SignAnimationPayload } from '../types/lis';
import { createProceduralLisAvatar, type ProceduralAvatarHandle } from '../three/proceduralAvatar';
// The VRM/GLB loader (GLTFLoader + three-vrm) loads on demand, when a GLB/VRM avatar is shown.
import type { LoadedVrmAvatar } from '../three/vrmAdapter';
import {
  AvatarNotFoundError,
  fetchAvatarBuffer,
  findBuiltinAvatar,
  type AvatarChoice,
} from '../avatars/avatarCatalog';
import { buildSignAnimationClip, type AvatarRigTarget } from '../utils/clipBuilder';

type CameraPreset = 'front' | 'hands' | 'face';

const PROCEDURAL_AVATAR_INFO = 'Procedural · 54 body bones + 10 facial bones + blendshapes';

interface AvatarViewportProps {
  animation: SignAnimationPayload | null;
  playTrigger: number;
  avatar: AvatarChoice;
  avatarError: string | null;
  onAvatarError: (err: string | null) => void;
  externalTimeSec?: number | null;
  onTimeUpdate?: (timeSec: number) => void;
}

const CAMERA_PRESETS: Record<
  CameraPreset,
  { pos: [number, number, number]; target: [number, number, number]; label: string }
> = {
  front: { pos: [0, 1.36, 1.95], target: [0, 1.20, 0.05], label: 'Front' },
  hands: { pos: [0, 1.28, 1.05], target: [0, 1.24, 0.22], label: 'Hands' },
  face: { pos: [0, 1.66, 0.72], target: [0, 1.62, 0.05], label: 'Face' },
};

function detectSoftwareGl(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (!gl) return true;
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
    const gpu = String(gl.getParameter(param));
    gl.getExtension('WEBGL_lose_context')?.loseContext();
    return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
  } catch {
    return true;
  }
}

export const AvatarViewport: React.FC<AvatarViewportProps> = ({
  animation,
  playTrigger,
  avatar,
  avatarError,
  onAvatarError,
  externalTimeSec = null,
  onTimeUpdate,
}) => {
  const mountRef = useRef<HTMLDivElement | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<0.5 | 1 | 1.5>(1);
  const [skeletonDebug, setSkeletonDebug] = useState(false);
  const [cameraPreset, setCameraPreset] = useState<CameraPreset>('front');
  const [dockCollapsed, setDockCollapsed] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [webglUnavailable, setWebglUnavailable] = useState(false);
  const [avatarInfo, setAvatarInfo] = useState<string>(PROCEDURAL_AVATAR_INFO);
  const [avatarNotice, setAvatarNotice] = useState<string | null>(null);
  const [avatarLoading, setAvatarLoading] = useState(false);

  // Refs bridging React state into the single-mount Three.js loop without re-creating renderer
  const isPlayingRef = useRef(isPlaying);
  const speedRef = useRef(speed);
  const onTimeUpdateRef = useRef(onTimeUpdate);
  isPlayingRef.current = isPlaying;
  speedRef.current = speed;
  onTimeUpdateRef.current = onTimeUpdate;
  const animationRef = useRef(animation);
  animationRef.current = animation;

  const sceneApiRef = useRef<{
    scene: THREE.Scene;
    camera: THREE.PerspectiveCamera;
    controls: OrbitControls;
    avatarGroup: THREE.Group;
    mixer: THREE.AnimationMixer | null;
    activeAction: THREE.AnimationAction | null;
    rigTarget: AvatarRigTarget;
    skeletonHelper: THREE.SkeletonHelper;
    vrmInstance: LoadedVrmAvatar | null;
    proceduralInstance: ProceduralAvatarHandle | null;
    requestRender: () => void;
    startPlaybackLoop: () => void;
    setCameraGoal: (preset: CameraPreset) => void;
    applyClip: (anim: SignAnimationPayload, autoPlay: boolean) => void;
    scrubTo: (t: number) => void;
  } | null>(null);

  // Initialize WebGLRenderer and Scene once
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const isSoftwareGl = detectSoftwareGl();
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: !isSoftwareGl,
        powerPreference: 'high-performance',
        preserveDrawingBuffer: true,
      });
    } catch {
      setWebglUnavailable(true);
      return;
    }

    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    const width = Math.max(320, container.clientWidth);
    const height = Math.max(320, container.clientHeight);
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xe8eef5);
    scene.fog = new THREE.Fog(0xe8eef5, 5.5, 14.0);

    // Mandatory studio environment reflection
    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    const camera = new THREE.PerspectiveCamera(38, width / height, 0.1, 30);
    camera.position.set(...CAMERA_PRESETS.front.pos);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(...CAMERA_PRESETS.front.target);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.08;
    controls.minDistance = 0.45;
    controls.maxDistance = 4.2;
    controls.maxPolarAngle = Math.PI * 0.56;
    controls.update();

    // Warm/cool 3-point studio lighting rig
    const hemiLight = new THREE.HemisphereLight(0xdbeafe, 0x94a3b8, 0.85);
    scene.add(hemiLight);

    const keyLight = new THREE.DirectionalLight(0xfff5e6, 1.85);
    keyLight.position.set(1.8, 3.2, 2.4);
    keyLight.castShadow = !isSoftwareGl;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 8;
    keyLight.shadow.camera.left = -1.2;
    keyLight.shadow.camera.right = 1.2;
    keyLight.shadow.camera.top = 2.0;
    keyLight.shadow.camera.bottom = -0.2;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const rimLight = new THREE.DirectionalLight(0x93c5fd, 0.75);
    rimLight.position.set(-1.6, 2.1, -1.8);
    scene.add(rimLight);

    // Studio floor cyclorama disc + pedestal + contact shadow grounding
    const floorGeo = new THREE.CircleGeometry(6, 48);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0xdbe3ee,
      roughness: 0.92,
      metalness: 0.02,
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.rotation.x = -Math.PI / 2;
    floorMesh.receiveShadow = true;
    scene.add(floorMesh);

    const gridHelper = new THREE.GridHelper(4.4, 16, 0x94a3b8, 0xcbd5e1);
    gridHelper.position.y = 0.005;
    scene.add(gridHelper);

    const contactGeo = new THREE.CircleGeometry(0.32, 32);
    const contactMat = new THREE.MeshBasicMaterial({
      color: 0x0f172a,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
    });
    const contactMesh = new THREE.Mesh(contactGeo, contactMat);
    contactMesh.rotation.x = -Math.PI / 2;
    contactMesh.position.y = 0.015;
    scene.add(contactMesh);

    const avatarGroup = new THREE.Group();
    scene.add(avatarGroup);

    const procedural = createProceduralLisAvatar();
    avatarGroup.add(procedural.root);

    let frameId = 0;
    let disposed = false;
    let renderBusy = false;
    let renderDirty = false;
    let lastTickMs = performance.now();
    let lastHudReportMs = 0;

    let cameraGoalPos: THREE.Vector3 | null = null;
    let cameraGoalTarget: THREE.Vector3 | null = null;

    const afterDraw = (cb: () => void) => {
      if (disposed) return;
      const gl = renderer.getContext() as WebGL2RenderingContext;
      if (gl && typeof gl.fenceSync === 'function') {
        const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
        gl.flush();
        const poll = () => {
          if (disposed) return;
          const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
          if (state === gl.UNSIGNALED) {
            frameId = window.setTimeout(poll, 5);
          } else {
            if (sync) gl.deleteSync(sync);
            frameId = window.setTimeout(cb, 80);
          }
        };
        frameId = window.setTimeout(poll, 5);
      } else {
        frameId = window.setTimeout(cb, 80);
      }
    };

    const advanceCameraTransition = (dt: number): boolean => {
      if (!cameraGoalPos || !cameraGoalTarget) return false;
      const lambda = 9.0;
      camera.position.x = THREE.MathUtils.damp(camera.position.x, cameraGoalPos.x, lambda, dt);
      camera.position.y = THREE.MathUtils.damp(camera.position.y, cameraGoalPos.y, lambda, dt);
      camera.position.z = THREE.MathUtils.damp(camera.position.z, cameraGoalPos.z, lambda, dt);
      controls.target.x = THREE.MathUtils.damp(controls.target.x, cameraGoalTarget.x, lambda, dt);
      controls.target.y = THREE.MathUtils.damp(controls.target.y, cameraGoalTarget.y, lambda, dt);
      controls.target.z = THREE.MathUtils.damp(controls.target.z, cameraGoalTarget.z, lambda, dt);
      if (
        camera.position.distanceTo(cameraGoalPos) < 0.005 &&
        controls.target.distanceTo(cameraGoalTarget) < 0.005
      ) {
        camera.position.copy(cameraGoalPos);
        controls.target.copy(cameraGoalTarget);
        cameraGoalPos = null;
        cameraGoalTarget = null;
        return false;
      }
      return true;
    };

    const requestRender = () => {
      if (disposed || isPlayingRef.current) return;
      if (renderBusy) {
        renderDirty = true;
        return;
      }
      renderBusy = true;
      frameId = requestAnimationFrame(() => {
        if (disposed) return;
        renderDirty = false;
        const now = performance.now();
        const dt = Math.min(0.1, (now - lastTickMs) / 1000);
        lastTickMs = now;
        const camMoving = advanceCameraTransition(dt);
        controls.update();
        renderer.render(scene, camera);
        const done = () => {
          renderBusy = false;
          if (camMoving || renderDirty) requestRender();
        };
        if (isSoftwareGl) afterDraw(done);
        else done();
      });
    };

    const tickPlayback = () => {
      if (disposed || !isPlayingRef.current) return;
      const now = performance.now();
      const dt = Math.min(0.1, Math.max(0.001, (now - lastTickMs) / 1000));
      lastTickMs = now;

      const api = sceneApiRef.current;
      advanceCameraTransition(dt);
      controls.update();

      if (api?.mixer && api.activeAction) {
        api.mixer.update(dt * speedRef.current);
        if (api.vrmInstance?.vrm) {
          api.vrmInstance.vrm.update(dt);
        }
        const t = api.activeAction.time;
        const dur = api.activeAction.getClip().duration;

        if (now - lastHudReportMs >= 100 || t >= dur - 0.001) {
          lastHudReportMs = now;
          setCurrentTime(t);
          onTimeUpdateRef.current?.(t);
        }

        if (!api.activeAction.isRunning() && t >= dur - 0.001) {
          setIsPlaying(false);
          isPlayingRef.current = false;
          setCurrentTime(dur);
          onTimeUpdateRef.current?.(dur);
        }
      }

      renderer.render(scene, camera);
      if (isPlayingRef.current) {
        if (isSoftwareGl) afterDraw(tickPlayback);
        else frameId = requestAnimationFrame(tickPlayback);
      }
    };

    const startPlaybackLoop = () => {
      if (disposed) return;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      renderBusy = false;
      lastTickMs = performance.now();
      if (isSoftwareGl) {
        tickPlayback();
      } else {
        frameId = requestAnimationFrame(tickPlayback);
      }
    };

    controls.addEventListener('change', requestRender);
    controls.addEventListener('start', () => {
      cameraGoalPos = null;
      cameraGoalTarget = null;
    });

    const handleResize = () => {
      if (disposed || !container) return;
      const w = Math.max(320, container.clientWidth);
      const h = Math.max(320, container.clientHeight);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
      requestRender();
    };
    const resizeObs = new ResizeObserver(handleResize);
    resizeObs.observe(container);

    const mixer = new THREE.AnimationMixer(procedural.root);

    sceneApiRef.current = {
      scene,
      camera,
      controls,
      avatarGroup,
      mixer,
      activeAction: null,
      rigTarget: procedural.rigTarget,
      skeletonHelper: procedural.skeletonHelper,
      vrmInstance: null,
      proceduralInstance: procedural,
      requestRender,
      startPlaybackLoop,
      setCameraGoal: (preset) => {
        const cfg = CAMERA_PRESETS[preset];
        cameraGoalPos = new THREE.Vector3(...cfg.pos);
        cameraGoalTarget = new THREE.Vector3(...cfg.target);
        if (!isPlayingRef.current) requestRender();
      },
      applyClip: (anim, autoPlay) => {
        const api = sceneApiRef.current;
        if (!api || !api.mixer) return;
        api.mixer.stopAllAction();
        const clip = buildSignAnimationClip(anim, api.rigTarget);
        const action = api.mixer.clipAction(clip);
        action.reset();
        action.loop = THREE.LoopOnce;
        action.clampWhenFinished = true;
        action.play();
        api.activeAction = action;
        api.mixer.update(0.0001);
        // VRM: copy normalized bone poses to the skinned (raw) bones even when paused.
        api.vrmInstance?.vrm?.update(0);
        setCurrentTime(0);
        if (autoPlay) {
          setIsPlaying(true);
          isPlayingRef.current = true;
          api.startPlaybackLoop();
        } else {
          setIsPlaying(false);
          isPlayingRef.current = false;
          api.requestRender();
        }
      },
      scrubTo: (t) => {
        const api = sceneApiRef.current;
        if (!api?.mixer || !api.activeAction) return;
        const dur = api.activeAction.getClip().duration;
        const clamped = Math.max(0, Math.min(dur, t));
        api.activeAction.paused = false;
        api.activeAction.time = clamped;
        api.mixer.update(0);
        api.vrmInstance?.vrm?.update(0);
        setCurrentTime(clamped);
        api.requestRender();
      },
    };

    if (!isSoftwareGl) {
      renderer.compile(scene, camera);
    }
    requestRender();

    const handleContextLost = (ev: Event) => {
      ev.preventDefault();
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
    };
    renderer.domElement.addEventListener('webglcontextlost', handleContextLost);

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      resizeObs.disconnect();
      controls.dispose();
      floorGeo.dispose();
      floorMat.dispose();
      contactGeo.dispose();
      contactMat.dispose();
      sceneApiRef.current?.proceduralInstance?.dispose();
      sceneApiRef.current?.vrmInstance?.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === container) {
        container.removeChild(renderer.domElement);
      }
      sceneApiRef.current = null;
    };
  }, []);

  // Show the chosen avatar: built-in GLB/VRM, local file or procedural.
  // A missing built-in file falls back to the procedural avatar.
  useEffect(() => {
    const api = sceneApiRef.current;
    if (!api) return;
    let cancelled = false;
    const abort = new AbortController();

    const attach = (
      root: THREE.Object3D,
      rigTarget: AvatarRigTarget,
      skeletonHelper: THREE.SkeletonHelper,
    ) => {
      const helperVisible = api.skeletonHelper.visible;
      api.avatarGroup.clear();
      api.avatarGroup.add(root);
      api.avatarGroup.visible = true;
      api.rigTarget = rigTarget;
      api.skeletonHelper = skeletonHelper;
      api.skeletonHelper.visible = helperVisible;
      api.mixer = new THREE.AnimationMixer(root);
      api.activeAction = null;
      const anim = animationRef.current;
      if (anim) api.applyClip(anim, false);
      else api.requestRender();
    };

    const showProcedural = () => {
      setAvatarInfo(PROCEDURAL_AVATAR_INFO);
      const procedural = api.proceduralInstance;
      if (!procedural) return;
      const alreadyShown = !api.vrmInstance && procedural.root.parent === api.avatarGroup;
      api.vrmInstance?.dispose();
      api.vrmInstance = null;
      if (alreadyShown) {
        api.avatarGroup.visible = true;
        api.requestRender();
        return;
      }
      attach(procedural.root, procedural.rigTarget, procedural.skeletonHelper);
    };

    const showLoaded = (loaded: LoadedVrmAvatar, name: string) => {
      api.vrmInstance?.dispose();
      api.vrmInstance = loaded;
      attach(loaded.root, loaded.rigTarget, loaded.skeletonHelper);
      const autoChannels = loaded.autoFaceRig?.channels.length ?? 0;
      const authoredMorphs = loaded.rigTarget.availableMorphs.size - autoChannels;
      const faceParts: string[] = [];
      if (authoredMorphs > 0) faceParts.push(`${authoredMorphs} morphs`);
      if (loaded.hasFacialBoneRig) faceParts.push('face bone rig');
      if (autoChannels > 0) faceParts.push(`auto face rig (${autoChannels} channels)`);
      const faceLabel = faceParts.length ? faceParts.join(' + ') : 'no face rig';
      const boneLabel =
        loaded.drivenBonesCount > loaded.matchedBonesCount
          ? `${loaded.matchedBonesCount} (${loaded.drivenBonesCount} driven) bones`
          : `${loaded.matchedBonesCount} bones`;
      setAvatarInfo(`${name} · ${loaded.detectedFormat} · ${boneLabel} · ${faceLabel}`);
    };

    const load = async () => {
      onAvatarError(null);
      setAvatarNotice(null);
      let buffer: ArrayBuffer;
      let name: string;
      if (avatar.kind === 'file') {
        name = avatar.file.name;
        setAvatarLoading(true);
        buffer = await avatar.file.arrayBuffer();
      } else {
        const entry = findBuiltinAvatar(avatar.id);
        name = entry.label;
        if (!entry.url) {
          setAvatarLoading(false);
          showProcedural();
          return;
        }
        // Hide the current avatar while the new one downloads (no procedural flash at startup).
        setAvatarLoading(true);
        api.avatarGroup.visible = false;
        api.requestRender();
        buffer = await fetchAvatarBuffer(entry.url, abort.signal);
      }
      const { loadVrmOrGlbFromArrayBuffer } = await import('../three/vrmAdapter');
      const loaded = await loadVrmOrGlbFromArrayBuffer(buffer);
      if (cancelled) {
        loaded.dispose();
        return;
      }
      showLoaded(loaded, name);
    };

    load()
      .catch((err: unknown) => {
        if (cancelled) return;
        showProcedural();
        if (err instanceof AvatarNotFoundError) {
          setAvatarNotice(`${err.url} not found. Showing the procedural avatar.`);
        } else {
          onAvatarError(err instanceof Error ? err.message : 'Invalid VRM/GLB file');
        }
      })
      .finally(() => {
        if (!cancelled) setAvatarLoading(false);
      });

    return () => {
      cancelled = true;
      abort.abort();
    };
  }, [avatar]);

  // Apply animation when class or playTrigger changes
  useEffect(() => {
    if (!animation || !sceneApiRef.current) return;
    sceneApiRef.current.applyClip(animation, playTrigger > 0);
  }, [animation, playTrigger]);

  // Sync external QA scrubber timestamp
  useEffect(() => {
    if (externalTimeSec === null || externalTimeSec === undefined) return;
    sceneApiRef.current?.scrubTo(externalTimeSec);
  }, [externalTimeSec]);

  // Toggle skeleton helper visibility
  useEffect(() => {
    const api = sceneApiRef.current;
    if (!api) return;
    api.skeletonHelper.visible = skeletonDebug;
    if (!isPlayingRef.current) api.requestRender();
  }, [skeletonDebug]);

  const handleTogglePlay = () => {
    const api = sceneApiRef.current;
    if (!api || !api.activeAction) return;
    if (isPlaying) {
      setIsPlaying(false);
      isPlayingRef.current = false;
      api.requestRender();
    } else {
      const dur = api.activeAction.getClip().duration;
      if (api.activeAction.time >= dur - 0.01) {
        api.activeAction.reset().play();
        setCurrentTime(0);
      }
      setIsPlaying(true);
      isPlayingRef.current = true;
      api.startPlaybackLoop();
    }
  };

  const handleReplay = () => {
    if (!animation || !sceneApiRef.current) return;
    sceneApiRef.current.applyClip(animation, true);
  };

  const duration = animation?.duration ?? 1.5;

  return (
    <div
      className="relative w-full h-full min-h-[460px] bg-[#E8EEF5] overflow-hidden select-none"
      data-testid="avatar-viewport"
      data-playing={isPlaying ? 'true' : 'false'}
      data-sign-id={animation?.classId ?? 'none'}
      data-skeleton-debug={skeletonDebug ? 'true' : 'false'}
    >
      {/* Dedicated WebGL mount container */}
      <div ref={mountRef} className="w-full h-full" />

      {/* Minimal top-left overlay label */}
      <div className="absolute top-4 left-4 z-10 pointer-events-none">
        <div className="bg-white/95 border border-slate-200 rounded-lg px-3.5 py-2 shadow-sm">
          <div className="text-xs text-slate-500">
            {avatarLoading ? 'Loading avatar…' : avatarInfo}
          </div>
          <div className="text-sm font-display font-semibold text-slate-900">
            {animation ? animation.phrase : 'Waiting for a phrase'}
          </div>
        </div>
      </div>

      {avatarError ? (
        <div role="alert" className="absolute top-4 right-4 z-10 bg-red-50 border border-red-200 rounded-lg px-3.5 py-2 text-xs text-red-800 flex items-center gap-2 max-w-xs">
          <AlertTriangle className="w-4 h-4 shrink-0 text-red-600" aria-hidden="true" />
          <span>{avatarError}</span>
        </div>
      ) : avatarNotice ? (
        <div role="status" className="absolute top-4 right-4 z-10 bg-amber-50 border border-amber-200 rounded-lg px-3.5 py-2 text-xs text-amber-900 flex items-center gap-2 max-w-xs">
          <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600" aria-hidden="true" />
          <span>{avatarNotice}</span>
        </div>
      ) : null}

      {webglUnavailable && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-100 p-6">
          <div className="max-w-sm text-center bg-white border border-slate-200 rounded-xl p-6">
            <h2 className="font-display text-base font-semibold text-slate-900">
              WebGL is not available
            </h2>
            <p className="mt-2 text-sm text-slate-600">
              Turn on hardware acceleration in the browser to show the 3D avatar.
            </p>
          </div>
        </div>
      )}

      {/* Unified Collapsible Bottom Control Dock (solid/opaque bg, no backdrop-filter over WebGL) */}
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 z-10 w-[min(860px,calc(100%-2rem))] flex flex-col items-center">
        <button
          type="button"
          onClick={() => setDockCollapsed((v) => !v)}
          aria-expanded={!dockCollapsed}
          className="mb-1.5 inline-flex items-center gap-1 px-3 py-1 text-xs font-medium text-slate-700 bg-white/95 hover:bg-white border border-slate-200 rounded-md shadow-sm cursor-pointer"
        >
          {dockCollapsed ? (
            <>
              <ChevronUp className="w-3.5 h-3.5" aria-hidden="true" />
              Show controls
            </>
          ) : (
            <>
              <ChevronDown className="w-3.5 h-3.5" aria-hidden="true" />
              Hide controls
            </>
          )}
        </button>

        {!dockCollapsed && (
          <div className="w-full bg-white/95 border border-slate-200 rounded-xl px-4 py-3 shadow-sm flex flex-wrap items-center justify-between gap-3">
            {/* Transport controls */}
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={handleTogglePlay}
                disabled={!animation}
                aria-label={isPlaying ? 'Pause animation' : 'Play animation'}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 hover:bg-slate-800 disabled:opacity-40 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                {isPlaying ? (
                  <>
                    <Pause className="w-3.5 h-3.5" aria-hidden="true" />
                    Pause
                  </>
                ) : (
                  <>
                    <Play className="w-3.5 h-3.5" aria-hidden="true" />
                    Play
                  </>
                )}
              </button>

              <button
                type="button"
                onClick={handleReplay}
                disabled={!animation}
                aria-label="Replay sign"
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-slate-100 hover:bg-slate-200 disabled:opacity-40 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              >
                <RotateCcw className="w-3.5 h-3.5" aria-hidden="true" />
                Replay
              </button>
            </div>

            {/* Scrubber */}
            <div className="flex items-center gap-2.5 flex-1 min-w-[160px]">
              <input
                type="range"
                min={0}
                max={duration}
                step={0.01}
                value={Math.min(currentTime, duration)}
                disabled={!animation}
                onChange={(e) => {
                  const t = parseFloat(e.target.value);
                  setIsPlaying(false);
                  isPlayingRef.current = false;
                  sceneApiRef.current?.scrubTo(t);
                  onTimeUpdateRef.current?.(t);
                }}
                aria-label="Animation timeline"
                className="w-full accent-blue-600 cursor-pointer"
              />
              <span className="font-mono text-xs text-slate-600 tabular-nums whitespace-nowrap shrink-0">
                {currentTime.toFixed(2)}s / {duration.toFixed(2)}s
              </span>
            </div>

            {/* Speed & Camera & Skeleton Debug */}
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center bg-slate-100 p-0.5 rounded-lg" role="group" aria-label="Playback speed">
                {([0.5, 1, 1.5] as const).map((s) => (
                  <button
                    key={s}
                    type="button"
                    onClick={() => setSpeed(s)}
                    aria-pressed={speed === s}
                    className={`px-2 py-1 text-xs font-mono font-medium rounded-md transition-colors cursor-pointer ${
                      speed === s
                        ? 'bg-white text-slate-900 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {s}x
                  </button>
                ))}
              </div>

              <div className="flex items-center bg-slate-100 p-0.5 rounded-lg" role="group" aria-label="Camera view">
                {(Object.keys(CAMERA_PRESETS) as CameraPreset[]).map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => {
                      setCameraPreset(preset);
                      sceneApiRef.current?.setCameraGoal(preset);
                    }}
                    aria-pressed={cameraPreset === preset}
                    className={`px-2.5 py-1 text-xs font-medium rounded-md transition-colors cursor-pointer ${
                      cameraPreset === preset
                        ? 'bg-white text-slate-900 shadow-sm'
                        : 'text-slate-600 hover:text-slate-900'
                    }`}
                  >
                    {CAMERA_PRESETS[preset].label}
                  </button>
                ))}
              </div>

              <button
                type="button"
                onClick={() => setSkeletonDebug((v) => !v)}
                aria-pressed={skeletonDebug}
                className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium rounded-lg border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                  skeletonDebug
                    ? 'bg-blue-50 border-blue-200 text-blue-800'
                    : 'bg-white border-slate-200 text-slate-700 hover:bg-slate-50'
                }`}
              >
                <Bone className="w-3.5 h-3.5" aria-hidden="true" />
                Skeleton
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
