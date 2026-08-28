import hashlib
import json
from pathlib import Path
import unittest

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "config"
FRAGMENT_ROOT = CONFIG_ROOT / "mission"


def _digest(parameters):
    payload = json.dumps(
        parameters,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TestMissionConfigComposition(unittest.TestCase):
    def test_fragments_are_unique_and_match_manifest(self):
        manifest = yaml.safe_load(
            (FRAGMENT_ROOT / "manifest.yaml").read_text(encoding="utf-8")
        )
        merged = {}
        for filename in manifest["fragment_order"]:
            payload = yaml.safe_load(
                (FRAGMENT_ROOT / filename).read_text(encoding="utf-8")
            )
            parameters = payload["mission_controller"]["ros__parameters"]
            duplicates = set(merged).intersection(parameters)
            self.assertFalse(duplicates, f"duplicate parameter keys: {duplicates}")
            merged.update(parameters)

        self.assertEqual(len(merged), manifest["parameter_count"])
        self.assertEqual(_digest(merged), manifest["semantic_sha256"])

    def test_compatibility_profiles_only_contain_overrides(self):
        for filename in ("mission.yaml", "mission_rm1.yaml"):
            payload = yaml.safe_load(
                (CONFIG_ROOT / filename).read_text(encoding="utf-8")
            )
            self.assertEqual(
                payload,
                {"mission_controller": {"ros__parameters": {}}},
            )

    def test_launch_loads_fragments_before_the_caller_override(self):
        launch_source = (PACKAGE_ROOT / "launch" / "mission.launch.py").read_text(
            encoding="utf-8"
        )
        parameters_block = launch_source.split("parameters=[", 1)[1].split("],", 1)[0]

        self.assertLess(
            parameters_block.index("*mission_fragments"),
            parameters_block.index("config_file"),
        )


if __name__ == "__main__":
    unittest.main()
