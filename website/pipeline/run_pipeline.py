"""End-to-end offline LIS pipeline orchestrator.

Steps:
1. Optionally generate synthetic Smartgloves II + Face Mesh dataset.
2. Load and clean each clip per class (reset-frame removal, Group 1 vs 2 comparison,
   dropout SLERP/linear interpolation, 1-Euro smoothing, 30 fps resampling).
3. Select representative prototype clip per class via DTW Medoid (or manual override).
4. Retarget prototype to VRM-compatible bone quaternion tracks + morph target tracks.
5. Export dictionary.json, per-class animations/*.json, and qa_summary.json for the web app.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from pipeline.glove_loader import (
    CleanedFaceClip,
    CleanedHandClip,
    load_and_clean_face_json,
    load_and_clean_glove_csv,
)
from pipeline.prototype_selector import select_class_prototype
from pipeline.qa_compare import generate_qa_report
from pipeline.retargeting import retarget_clip_to_avatar
from pipeline.synthetic_generator import generate_synthetic_dataset
from pipeline.yaml_io import load_yaml


def run_offline_pipeline(
    dictionary_path: Path,
    sensor_cfg_path: Path,
    dataset_dir: Path,
    web_data_dir: Path,
    generate_if_missing: bool = True,
    clips_per_class: int = 4,
    signers: int = 4,
) -> dict[str, Any]:
    dict_cfg = load_yaml(dictionary_path)
    sensor_cfg = load_yaml(sensor_cfg_path)

    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.exists():
        if not generate_if_missing:
            raise FileNotFoundError(f"Dataset manifest not found at {manifest_path}")
        manifest = generate_synthetic_dataset(
            dictionary_path,
            sensor_cfg_path,
            dataset_dir,
            clips_per_class=clips_per_class,
            signers=signers,
        )
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for item in manifest:
        by_class[item["class_id"]].append(item)

    animations_dir = web_data_dir / "animations"
    animations_dir.mkdir(parents=True, exist_ok=True)

    proto_cfg = sensor_cfg.get("prototype_selection", {})
    method = str(proto_cfg.get("method", "medoid_dtw"))
    win_ratio = float(proto_cfg.get("dtw_window_ratio", 0.20))

    exported_dict_classes: list[dict[str, Any]] = []
    for cls_entry in dict_cfg["classes"]:
        cls_id = cls_entry["id"]
        entries = by_class.get(cls_id, [])
        if not entries:
            raise ValueError(f"No clips found for dictionary class '{cls_id}'")

        clip_ids: list[str] = []
        right_clips: list[CleanedHandClip] = []
        left_clips: list[CleanedHandClip] = []
        face_clips: list[CleanedFaceClip] = []

        for entry in entries:
            r_clip = load_and_clean_glove_csv(entry["right_csv"], "right", sensor_cfg)
            l_clip = load_and_clean_glove_csv(
                entry["left_csv"], "left", sensor_cfg, target_times=r_clip.times
            )
            f_clip = load_and_clean_face_json(entry["face_json"], target_times=r_clip.times)
            clip_ids.append(entry["clip_id"])
            right_clips.append(r_clip)
            left_clips.append(l_clip)
            face_clips.append(f_clip)

        proto = select_class_prototype(
            clip_ids=clip_ids,
            right_clips=right_clips,
            left_clips=left_clips,
            method=method,
            window_ratio=win_ratio,
            manual_override_id=cls_entry.get("manual_prototype"),
        )

        p_idx = proto.selected_index
        anim_payload = retarget_clip_to_avatar(
            cls_entry=cls_entry,
            right_clip=right_clips[p_idx],
            left_clip=left_clips[p_idx],
            face_clip=face_clips[p_idx],
            sensor_cfg=sensor_cfg,
            prototype_id=proto.selected_clip_id,
        )
        anim_payload["prototypeSelection"] = {
            "method": proto.method,
            "candidates": clip_ids,
            "meanDtwDistance": proto.mean_distance_to_others,
        }

        out_anim_path = animations_dir / f"{cls_id}.json"
        out_anim_path.write_text(json.dumps(anim_payload), encoding="utf-8")

        exported_dict_classes.append(
            {
                "id": cls_id,
                "phrase": cls_entry["phrase"],
                "category": cls_entry["category"],
                "description": cls_entry["description"],
                "twoHanded": bool(cls_entry.get("two_handed", False)),
                "synonyms": cls_entry.get("synonyms", []),
                # Per-language text (phrase, category, description, synonyms) for other source languages
                "translations": cls_entry.get("translations", {}),
                "anchor": cls_entry["anchor"],
                "prototypeClipId": proto.selected_clip_id,
                "duration": anim_payload["duration"],
                "animationUrl": f"/data/animations/{cls_id}.json",
            }
        )

    dict_export = {
        "metadata": dict_cfg.get("metadata", {}),
        "classes": exported_dict_classes,
    }
    (web_data_dir / "dictionary.json").write_text(
        json.dumps(dict_export, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    qa_summary = generate_qa_report(animations_dir, web_data_dir / "qa_summary.json")
    return {"dictionary": dict_export, "qa": qa_summary}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline LIS Smartgloves II pipeline.")
    parser.add_argument("--dictionary", default="config/dictionary.yaml")
    parser.add_argument("--config", default="config/sensors_and_retargeting.yaml")
    parser.add_argument("--dataset", default="data/synthetic")
    parser.add_argument("--out", default="public/data")
    parser.add_argument("--clips-per-class", type=int, default=4)
    parser.add_argument("--signers", type=int, default=4)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if args.regenerate and (dataset_dir / "dataset_manifest.json").exists():
        (dataset_dir / "dataset_manifest.json").unlink()

    result = run_offline_pipeline(
        Path(args.dictionary),
        Path(args.config),
        dataset_dir,
        Path(args.out),
        generate_if_missing=True,
        clips_per_class=args.clips_per_class,
        signers=args.signers,
    )
    print(
        f"Exported {len(result['dictionary']['classes'])} LIS classes and QA report to {args.out}"
    )


if __name__ == "__main__":
    main()
