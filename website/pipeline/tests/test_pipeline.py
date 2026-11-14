"""Unit tests for the offline LIS motion pipeline."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from pipeline.glove_loader import load_and_clean_face_json, load_and_clean_glove_csv
from pipeline.prototype_selector import dtw_distance, select_class_prototype
from pipeline.retargeting import retarget_clip_to_avatar
from pipeline.signal_processing import (
    enforce_quaternion_continuity,
    interpolate_quat_dropouts,
    quat_dot,
    quat_from_axis_angle,
    quat_norm,
    quat_slerp,
)
from pipeline.synthetic_generator import generate_clip
from pipeline.text_matcher import match_phrase, normalize_italian
from pipeline.yaml_io import load_yaml


class TestSignalProcessing(unittest.TestCase):
    def test_quaternion_continuity_flips_antipodal(self) -> None:
        q0 = quat_from_axis_angle((1.0, 0.0, 0.0), 0.3)
        q1_flipped = (-q0[0], -q0[1], -q0[2], -q0[3])
        cont = enforce_quaternion_continuity([q0, q1_flipped])
        self.assertGreater(quat_dot(cont[0], cont[1]), 0.999)

    def test_slerp_midpoint_and_dropout_interpolation(self) -> None:
        q0 = (0.0, 0.0, 0.0, 1.0)
        q1 = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 2.0)
        mid = quat_slerp(q0, q1, 0.5)
        self.assertAlmostEqual(quat_norm(mid), 1.0, places=5)
        # Expect ~45 degrees around Y
        expected = quat_from_axis_angle((0.0, 1.0, 0.0), math.pi / 4.0)
        self.assertGreater(abs(quat_dot(mid, expected)), 0.999)

        interp = interpolate_quat_dropouts(
            [0.0, 0.5, 1.0],
            [q0, (0.0, 0.0, 0.0, 0.0), q1],
            [True, False, True],
        )
        self.assertGreater(abs(quat_dot(interp[1], expected)), 0.999)


class TestLoaderAndPipeline(unittest.TestCase):
    def test_synthetic_load_clean_retarget(self) -> None:
        dict_cfg = load_yaml("config/dictionary.yaml")
        sensor_cfg = load_yaml("config/sensors_and_retargeting.yaml")
        cls_entry = dict_cfg["classes"][0]

        import random

        with tempfile.TemporaryDirectory() as tmp:
            rng = random.Random(42)
            entries = [
                generate_clip(Path(tmp), cls_entry, s, 0, sensor_cfg, rng)
                for s in range(3)
            ]
            r_clips = [load_and_clean_glove_csv(e["right_csv"], "right", sensor_cfg) for e in entries]
            l_clips = [
                load_and_clean_glove_csv(e["left_csv"], "left", sensor_cfg, target_times=r.times)
                for e, r in zip(entries, r_clips)
            ]
            # Verify reset frame was discarded
            self.assertEqual(r_clips[0].reset_frames_discarded, 1)
            # Verify Group 1 vs Group 2 comparison stats are populated
            self.assertGreaterEqual(len(r_clips[0].comparison_stats), 5)

            proto = select_class_prototype(
                [e["clip_id"] for e in entries], r_clips, l_clips
            )
            self.assertIn(proto.selected_clip_id, [e["clip_id"] for e in entries])

            p_idx = proto.selected_index
            f_clip = load_and_clean_face_json(entries[p_idx]["face_json"], r_clips[p_idx].times)
            anim = retarget_clip_to_avatar(
                cls_entry, r_clips[p_idx], l_clips[p_idx], f_clip, sensor_cfg, proto.selected_clip_id
            )
            self.assertIn("rightIndexProximal", anim["boneTracks"])
            self.assertIn("mouthSmile", anim["morphTracks"])
            self.assertEqual(
                len(anim["boneTracks"]["rightIndexProximal"]),
                len(anim["times"]) * 4,
            )

    def test_dtw_identity_is_zero(self) -> None:
        seq = [[0.1, 0.2], [0.3, 0.4], [0.2, 0.1]]
        dist, path = dtw_distance(seq, seq)
        self.assertAlmostEqual(dist, 0.0, places=6)
        self.assertEqual(len(path), 3)


class TestTextMatcher(unittest.TestCase):
    def test_normalization_and_synonym_match(self) -> None:
        self.assertEqual(normalize_italian("Perché non capisco?!"), "perche non capisco")
        dict_cfg = load_yaml("config/dictionary.yaml")
        res = match_phrase("ti ringrazio molto!", dict_cfg)
        self.assertTrue(res.matched)
        self.assertIsNotNone(res.best)
        self.assertEqual(res.best.class_id, "grazie")

    def test_out_of_dictionary_returns_three_suggestions(self) -> None:
        dict_cfg = load_yaml("config/dictionary.yaml")
        res = match_phrase("vorrei ordinare una pizza margherita", dict_cfg)
        self.assertFalse(res.matched)
        self.assertEqual(len(res.suggestions), 3)


if __name__ == "__main__":
    unittest.main()
