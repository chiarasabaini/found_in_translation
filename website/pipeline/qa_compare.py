"""QA diagnostic comparison between original cleaned glove clips and exported avatar tracks."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def compute_clip_qa_metrics(anim_payload: dict[str, Any]) -> dict[str, Any]:
    qa = anim_payload.get("qaComparison", {})
    frames = qa.get("frames", [])
    if not frames:
        return {"classId": anim_payload.get("classId"), "status": "empty"}

    wrist_err_sq = 0.0
    g1_g2_curl_sq = 0.0
    curl_samples = 0
    max_emf = 0.0

    for fr in frames:
        for side in ("right", "left"):
            h = fr[side]
            tw = h["target_wrist"]
            aw = h["achieved_wrist"]
            wrist_err_sq += (tw[0] - aw[0]) ** 2 + (tw[1] - aw[1]) ** 2 + (tw[2] - aw[2]) ** 2
            max_emf = max(max_emf, float(h.get("mean_emf_residual", 0.0)))
            c1 = h.get("curls_deg_g1", {})
            c2 = h.get("curls_deg_g2", {})
            for fname in c2:
                if fname in c1:
                    diff = float(c1[fname]) - float(c2[fname])
                    g1_g2_curl_sq += diff * diff
                    curl_samples += 1

    rms_wrist_cm = 100.0 * math.sqrt(wrist_err_sq / max(1, len(frames) * 2))
    rms_curl_deg = math.sqrt(g1_g2_curl_sq / max(1, curl_samples))

    return {
        "classId": anim_payload["classId"],
        "phrase": anim_payload["phrase"],
        "prototypeClipId": anim_payload["prototypeClipId"],
        "frameCount": len(frames),
        "durationSec": anim_payload["duration"],
        "rmsWristReachErrorCm": round(rms_wrist_cm, 3),
        "rmsGroup1VsGroup2CurlDeg": round(rms_curl_deg, 3),
        "maxMeanEmfResidual": round(max_emf, 4),
    }


def generate_qa_report(animations_dir: Path, out_path: Path) -> dict[str, Any]:
    reports = []
    for json_file in sorted(animations_dir.glob("*.json")):
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        reports.append(compute_clip_qa_metrics(payload))
    summary = {"totalClasses": len(reports), "classes": reports}
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
