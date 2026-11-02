import json
import tempfile
import unittest
from pathlib import Path

import tests.helpers  # noqa: F401  (puts the repo root on sys.path)
from partseg.config import SegmentConfig
from partseg.errors import InvalidInputError


class ConfigTests(unittest.TestCase):
    def test_defaults_validate(self):
        SegmentConfig().validate()

    def test_presets(self):
        balanced, fast = SegmentConfig.preset("balanced"), SegmentConfig.preset("fast")
        self.assertTrue(balanced.features.enabled)
        self.assertFalse(fast.features.enabled)
        self.assertLess(fast.reprompt.rounds, balanced.reprompt.rounds)
        precise = SegmentConfig.preset("precise")
        self.assertFalse(precise.reprompt.enabled)
        self.assertFalse(precise.features.enabled)
        self.assertTrue(precise.polish.enabled)
        with self.assertRaises(InvalidInputError):
            SegmentConfig.preset("turbo")

    def test_override_sets_and_coerces_values(self):
        cfg = SegmentConfig().override(["polish.stay_cost=0.1", "reprompt.rounds=2", "export.preview=false"])
        self.assertEqual(cfg.polish.stay_cost, 0.1)
        self.assertEqual(cfg.reprompt.rounds, 2)
        self.assertFalse(cfg.export.preview)

    def test_override_does_not_mutate_the_original(self):
        base = SegmentConfig()
        base.override(["polish.stay_cost=0.5"])
        self.assertEqual(base.polish.stay_cost, 0.06)

    def test_unknown_key_is_rejected_with_the_valid_names(self):
        with self.assertRaises(InvalidInputError) as ctx:
            SegmentConfig().override(["polish.stay_cots=0.1"])
        self.assertIn("stay_cost", str(ctx.exception))

    def test_wrong_types_are_rejected(self):
        for bad in ("reprompt.rounds=1.5", "reprompt.enabled=3", "export.up_axis=7"):
            with self.assertRaises(InvalidInputError, msg=bad):
                SegmentConfig().override([bad])

    def test_out_of_range_values_fail_validation(self):
        with self.assertRaises(InvalidInputError):
            SegmentConfig().override(["export.up_axis=w"])
        with self.assertRaises(InvalidInputError):
            SegmentConfig().override(["features.min_frac=0.9"])

    def test_json_round_trip(self):
        cfg = SegmentConfig.preset("fast")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
            again = SegmentConfig.from_json(path)
        self.assertEqual(again.to_dict(), cfg.to_dict())

    def test_bad_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(InvalidInputError):
                SegmentConfig.from_json(path)


if __name__ == "__main__":
    unittest.main()
