import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import yaml

from mission_runtime.tf_calibration_defaults import (
    apply_and_save,
    save_default_corrections,
)


KEY = "grasp_box_tf_joint123_left_target_correction_pose_box_smallbox_layer1"


class TestTfCalibrationDefaults(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        root.mkdir(parents=True, exist_ok=True)
        (root / "grasp_tf.yaml").write_text(
            "mission_controller:\n  ros__parameters:\n"
            f"    {KEY}:\n" + "    - 0.0\n" * 7,
            encoding="utf-8",
        )
        (root / "manifest.yaml").write_text(
            "fragment_order:\n- grasp_tf.yaml\nparameter_count: 1\n"
            f"semantic_sha256: {'0' * 64}\n",
            encoding="utf-8",
        )
        return root / "grasp_tf.yaml"

    def test_save_updates_exact_field_and_manifest_with_backups(self):
        with TemporaryDirectory() as directory:
            path = self._fixture(Path(directory))
            values = [0.001, 0.002, 0.003, 0.0, 0.0, 0.0, 1.0]
            backup, manifest_backup = save_default_corrections(path, {KEY: values})
            self.assertTrue(backup.is_file())
            self.assertTrue(manifest_backup.is_file())
            data = yaml.safe_load(path.read_text())
            self.assertEqual(data["mission_controller"]["ros__parameters"][KEY], values)
            manifest = yaml.safe_load((path.parent / "manifest.yaml").read_text())
            digest = hashlib.sha256(
                json.dumps(data["mission_controller"]["ros__parameters"],
                           sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True).encode()
            ).hexdigest()
            self.assertEqual(manifest["semantic_sha256"], digest)

    def test_invalid_field_cannot_modify_source(self):
        with TemporaryDirectory() as directory:
            path = self._fixture(Path(directory))
            before = path.read_text()
            with self.assertRaises(ValueError):
                save_default_corrections(path, {"wrong_parameter": [0.0] * 7})
            self.assertEqual(path.read_text(), before)

    def test_live_parameters_roll_back_if_save_fails(self):
        with TemporaryDirectory() as directory:
            path = self._fixture(Path(directory))
            calls = []
            before = [0.0] * 7
            target = [0.1] * 7
            with self.assertRaises(ValueError):
                apply_and_save(
                    config_file=path,
                    results={"wrong_parameter": target},
                    previous_runtime={"wrong_parameter": before},
                    param_set=lambda name, values: calls.append((name, list(values))),
                )
            self.assertEqual(calls, [("wrong_parameter", target),
                                     ("wrong_parameter", before)])

    def test_calibration_updates_source_and_installed_defaults(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = self._fixture(
                workspace / "src" / "mission_controller" / "config" / "mission"
            )
            installed = self._fixture(
                workspace / "install" / "share" / "mission_controller"
                / "config" / "mission"
            )
            values = [0.001, 0.002, 0.003, 0.0, 0.0, 0.0, 1.0]
            calls = []
            backup, _ = apply_and_save(
                config_file=installed,
                results={KEY: values},
                previous_runtime={KEY: [0.0] * 7},
                param_set=lambda name, result: calls.append((name, list(result))),
            )
            self.assertEqual(backup.parent, source.parent)
            self.assertEqual(calls, [(KEY, values)])
            for path in (source, installed):
                data = yaml.safe_load(path.read_text())
                self.assertEqual(data["mission_controller"]["ros__parameters"][KEY], values)
            self.assertEqual(source.read_text(), installed.read_text())
            self.assertEqual(
                (source.parent / "manifest.yaml").read_text(),
                (installed.parent / "manifest.yaml").read_text(),
            )

    def test_installed_save_failure_rolls_back_source_and_runtime(self):
        with TemporaryDirectory() as directory:
            workspace = Path(directory)
            source = self._fixture(
                workspace / "src" / "mission_controller" / "config" / "mission"
            )
            installed = self._fixture(
                workspace / "install" / "share" / "mission_controller"
                / "config" / "mission"
            )
            installed.write_text(installed.read_text().replace(KEY, "other_parameter"))
            original_source = source.read_text()
            original_manifest = (source.parent / "manifest.yaml").read_text()
            calls = []
            target = [0.1] * 7
            before = [0.0] * 7
            with self.assertRaises(ValueError):
                apply_and_save(
                    config_file=source,
                    results={KEY: target},
                    previous_runtime={KEY: before},
                    param_set=lambda name, result: calls.append((name, list(result))),
                )
            self.assertEqual(source.read_text(), original_source)
            self.assertEqual((source.parent / "manifest.yaml").read_text(), original_manifest)
            self.assertEqual(calls, [(KEY, target), (KEY, before)])


if __name__ == "__main__":
    unittest.main()
