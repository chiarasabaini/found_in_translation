import React from 'react';

export const ArchitectureView: React.FC = () => {
  return (
    <div className="flex-1 overflow-y-auto bg-slate-50 p-6 lg:p-8">
      <div className="max-w-4xl mx-auto space-y-8">
        <div>
          <h1 className="font-display text-2xl font-semibold text-slate-900">
            Architecture, limits and hardware checks
          </h1>
          <p className="mt-1 text-sm text-slate-600">
            Summary of the engineering choices for the Junction 2026 demo, and the explicit list of parameters to validate on real Smartgloves II data.
          </p>
        </div>

        <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
          <h2 className="font-display text-lg font-semibold text-slate-900">
            01. Architecture and inherent limits
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            With 10 closed classes, deterministic client-side matching (normalization + per-language synonyms from YAML + fuzzy Levenshtein/n-gram similarity) gives &lt;1 ms latency, no backend and no hallucinations. Multilingual embeddings or constrained LLMs become useful only when the dictionary grows past 50–100 heavily paraphrased phrases.
          </p>
          <p className="text-sm text-slate-600 leading-relaxed">
            <strong>Languages:</strong> the UI is in English. The source language (the phrase you type) and the target sign language are set in the top bar. Each dictionary class has its phrase, synonyms and description per source language (<code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">translations</code> in <code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">config/dictionary.yaml</code>). Each sign language has its own dictionary and animations (<code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">src/i18n/languages.ts</code>).
          </p>
          <p className="text-sm text-slate-600 leading-relaxed">
            <strong>Main intelligibility limit:</strong> the Smartgloves II measure finger flexion and hand orientation, but <strong>not the absolute position of the hand relative to the body</strong>. In LIS the same handshape near the chin, the forehead or the chest changes meaning. The synthetic pipeline therefore uses <strong>per-class spatial anchors</strong> in <code className="font-mono text-xs bg-slate-100 px-1 py-0.5 rounded">config/dictionary.yaml</code>, solved with an analytical 2-bone IK (shoulder–elbow–wrist). Real recordings use the camera body tracking instead (section 04).
          </p>
        </section>

        <section className="bg-white border border-slate-200 rounded-xl p-6 space-y-3">
          <h2 className="font-display text-lg font-semibold text-slate-900">
            02. Professional GLB models (rig, bones and blendshapes)
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            If anime-style VRM models do not fit an institutional or hackathon presentation, the app supports realistic <strong>.glb</strong> files with a standard humanoid skeleton. Recommended sources:
          </p>
          <ul className="text-sm text-slate-600 space-y-2 list-disc pl-5">
            <li>
              <strong>Ready Player Me (.glb with ?morphTargets=ARKit):</strong> realistic or semi-realistic avatars in plain clothes, with all 15 phalanges per hand (<code className="font-mono text-xs">RightArm</code>, <code className="font-mono text-xs">RightForeArm</code>, <code className="font-mono text-xs">RightHandIndex1..3</code>) and the 52 ARKit blendshapes on <code className="font-mono text-xs">Wolf3D_Head</code>.
            </li>
            <li>
              <strong>Avaturn (.glb):</strong> photorealistic avatars with a humanoid rig and 52 ARKit blendshapes, exported directly as GLB.
            </li>
            <li>
              <strong>Adobe Mixamo + Blender (.glb):</strong> realistic characters in T-pose/A-pose (<code className="font-mono text-xs">mixamorig:RightArm</code>, <code className="font-mono text-xs">RightHandThumb1..3</code>). Note: base Mixamo models often have no facial morph targets (hands, arms and head animate; lips and eyebrows do not).
            </li>
          </ul>
          <p className="text-sm text-slate-600 leading-relaxed">
            <strong>Minimum GLB rig:</strong> one <code className="font-mono text-xs">SkinnedMesh</code> skeleton in T-pose or A-pose, mapped phalanges (3 bones per finger), scale in meters (~1.7 m, normalized on load anyway) and Mixamo/RPM/VRM-style bone names. The bind pose of arms, hands and every finger bone is aligned to the source rig on load, for GLB and VRM alike; VRM 0.x models are turned to face the camera.
          </p>
        </section>

        <section className="bg-white border border-amber-200 rounded-xl p-6 space-y-3">
          <h2 className="font-display text-lg font-semibold text-amber-950">
            03. Items NOT verified on real hardware data (synthetic pipeline)
          </h2>
          <p className="text-sm text-slate-600">
            The offline pipeline was built and tested on the synthetic generator before the real dataset arrived. Check these points on the first real raw-sensor CSV (with the <strong>QA comparison</strong> view and <code className="font-mono text-xs">config/sensors_and_retargeting.yaml</code>):
          </p>
          <ul className="text-sm text-slate-700 space-y-1.5 list-disc pl-5">
            <li>
              <strong>Quaternion order:</strong> assumed <code className="font-mono text-xs">xyzw</code> in <code className="font-mono text-xs">TRANSFORM[3..6]</code> and <code className="font-mono text-xs">ORIENTATION[0..3]</code>. If the hand looks oddly rotated, set <code className="font-mono text-xs">quaternion_order: &quot;wxyz&quot;</code>.
            </li>
            <li>
              <strong>Sensor axes vs VRM:</strong> the physical mounting of the hub on the back of the glove can differ from the Three.js axes (+Y up, +Z forward). Adjust <code className="font-mono text-xs">retargeting.axis_remap_euler_deg</code> per hand.
            </li>
            <li>
              <strong>Left-hand schema and EM Group 2:</strong> assumed identical to the right hand, with Group 2 (EM-corrected) preferred when <code className="font-mono text-xs">relative_emf_residual &lt;= 0.35</code>.
            </li>
            <li>
              <strong>Face Mesh column names:</strong> mapped to the 52 ARKit blendshapes; update <code className="font-mono text-xs">face_blendshape_map</code> if the extractor uses custom names.
            </li>
          </ul>
        </section>

        <section className="bg-white border border-emerald-200 rounded-xl p-6 space-y-3">
          <h2 className="font-display text-lg font-semibold text-emerald-950">
            04. Real recordings (Recordings view)
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            The <strong>Recordings</strong> view reads a recording folder directly in the browser: <code className="font-mono text-xs">left_hand.csv</code> and <code className="font-mono text-xs">right_hand.csv</code> (hands solved by the Rokoko hand solver, 26 joints with position and quaternion), <code className="font-mono text-xs">face.csv</code> (478 MediaPipe landmarks + 52 ARKit blendshapes) and <code className="font-mono text-xs">body.csv</code> (neck, shoulders, elbows). All files share the <code className="font-mono text-xs">timestamp_ms</code> column. The Python pipeline is not needed.
          </p>
          <ul className="text-sm text-slate-600 space-y-1.5 list-disc pl-5">
            <li>
              <strong>Direction-based retargeting:</strong> every phalanx, the back of the hand and the forearm come from joint positions, so the result does not depend on the solver&apos;s local axes. Fingers use 3 real phalanges each, not a curl estimated from one sensor.
            </li>
            <li>
              <strong>Hand position:</strong> the upper arm comes from the shoulders and elbows in <code className="font-mono text-xs">body.csv</code> (weighted by landmark visibility), the forearm from the glove. Without <code className="font-mono text-xs">body.csv</code>, the elbow is estimated from the forearm elevation.
            </li>
            <li>
              <strong>Glove yaw:</strong> the gloves are referenced to magnetic north. The yaw is estimated by assuming the forearms point forward while signing; it can be corrected by hand (global and per hand).
            </li>
            <li>
              <strong>Axes detected automatically:</strong> the axis signs of <code className="font-mono text-xs">body.csv</code> (left shoulder = +X, forward = +Z) and the camera mirroring for the face. Head and torso are relative to the median pose of the recording.
            </li>
            <li>
              <strong>Frame rate:</strong> the test recording runs at about 15 Hz (the gloves can stream 83 Hz). Clips are resampled to 30 fps with interpolation; very fast finger movements lose detail.
            </li>
            <li>
              <strong>Persistence:</strong> assigned signs stay in the browser (IndexedDB), per sign language. To make them permanent for everyone, download the JSON and copy it to <code className="font-mono text-xs">public/data/animations/&lt;id&gt;.json</code>.
            </li>
          </ul>
        </section>

        <section className="bg-white border border-blue-200 rounded-xl p-6 space-y-3">
          <h2 className="font-display text-lg font-semibold text-blue-950">
            05. Motion model (Interpreter, &ldquo;Motion model&rdquo; mode)
          </h2>
          <p className="text-sm text-slate-600 leading-relaxed">
            Target flow: <strong>phrase → text-to-motion model → recording-format CSV (hands, body, face) → same import and retargeting as real recordings → avatar</strong>. The model output is exactly the format of a glove recording session, so a model trained on the recordings plugs in without changes to the rest of the site.
          </p>
          <ul className="text-sm text-slate-600 space-y-1.5 list-disc pl-5">
            <li>
              <strong>Now:</strong> a mock model in the browser (<code className="font-mono text-xs">src/generation/mockMotionModel.ts</code>) splits the phrase into dictionary signs and synthesizes the CSV files from placeholder handshapes, the dictionary anchors and arm IK. The handshapes are <strong>not validated LIS</strong>.
            </li>
            <li>
              <strong>Real model:</strong> set <code className="font-mono text-xs">VITE_MOTION_MODEL_URL</code>. The site sends <code className="font-mono text-xs">{'{text, sourceLanguage, signLanguage}'}</code> and expects <code className="font-mono text-xs">{'{files: [{name, text}], segments?, unmatched?, model?}'}</code> (README, &ldquo;Motion model API&rdquo;).
            </li>
            <li>
              <strong>Required output:</strong> <code className="font-mono text-xs">left_hand.csv</code> and <code className="font-mono text-xs">right_hand.csv</code> with joint positions; <code className="font-mono text-xs">body.csv</code> is strongly recommended because sign location carries meaning; <code className="font-mono text-xs">face.csv</code> is optional.
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
};
