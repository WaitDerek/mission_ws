"""Persist a confirmed TF target calibration without rewriting unrelated YAML."""

from __future__ import annotations

import math
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import yaml


def _workspace_config_files(config_file: Path) -> tuple[Path, ...]:
    """Return source and installed fragments for a colcon workspace, if present."""
    config_file = config_file.resolve(strict=True)
    config_root = config_file.parent
    for parent in config_root.parents:
        if parent.name == "install":
            install_root = parent
            source_root = parent.parent / "src" / "mission_controller" / "config" / "mission"
            break
        if parent.name == "src":
            install_root = parent.parent / "install"
            source_root = parent / "mission_controller" / "config" / "mission"
            break
    else:
        return (config_file,)

    installed_roots = (
        install_root / "share" / "mission_controller" / "config" / "mission",
        install_root / "mission_controller" / "share" / "mission_controller" / "config" / "mission",
    )
    if config_root != source_root and config_root not in installed_roots:
        return (config_file,)
    installed_root = next(
        (root for root in installed_roots if (root / config_file.name).is_file()),
        None,
    )
    source_file = source_root / config_file.name
    if (
        not source_file.is_file()
        or not (source_root / "manifest.yaml").is_file()
        or installed_root is None
        or not (installed_root / "manifest.yaml").is_file()
    ):
        raise FileNotFoundError(
            "标定需要同时找到源码与安装配置及各自的 manifest.yaml；"
            f"source={source_root}, install={installed_roots}"
        )
    installed_file = installed_root / config_file.name
    return tuple(dict.fromkeys((source_file.resolve(), installed_file.resolve())))


def default_config_root() -> Path:
    """Locate the source config when available, otherwise the installed config.

    Python entry points are installed under ``lib/pythonX/site-packages``;
    deriving ``config/mission`` relative to that directory points at a path
    that does not exist. Resolve the package share through the ament index,
    then prefer its workspace source so calibration survives the next build.
    """
    candidates: list[Path] = []
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("mission_controller")) / "config" / "mission"
        if (installed / "drag.yaml").is_file():
            candidates.extend(path.parent for path in _workspace_config_files(installed / "drag.yaml"))
        else:
            candidates.append(installed)
    except Exception:
        # The helper is also used by tests and offline calibration utilities
        # that may run without a sourced ROS environment.
        pass

    module_path = Path(__file__).resolve()
    candidates.extend(parent / "config" / "mission" for parent in module_path.parents)
    for candidate in candidates:
        if (candidate / "drag.yaml").is_file() and (candidate / "manifest.yaml").is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "无法定位 mission_controller/config/mission（需要 drag.yaml 和 manifest.yaml）；"
        f"已检查：{searched}"
    )


def _replace_seven_value_sequence(
    source: str, name: str, values: Sequence[float]
) -> str:
    if len(values) != 7 or not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"{name} must contain seven finite values")
    lines = source.splitlines(keepends=True)
    keys = [i for i, line in enumerate(lines) if line.strip() == f"{name}:"]
    if len(keys) != 1:
        raise ValueError(f"配置参数 {name} 在源 YAML 中出现 {len(keys)} 次，要求恰好一次")
    key_index = keys[0]
    indent = re.match(r"^\s*", lines[key_index]).group(0)
    item_pattern = re.compile(rf"^{re.escape(indent)}-\s+([^#\r\n]+)(?:\s*#.*)?(?:\r?\n)?$")
    old = []
    for index in range(key_index + 1, key_index + 8):
        if index >= len(lines) or not (match := item_pattern.match(lines[index])):
            raise ValueError(f"配置参数 {name} 不是七行数值列表；没有写文件")
        old.append(float(match.group(1)))
    if key_index + 8 < len(lines) and item_pattern.match(lines[key_index + 8]):
        raise ValueError(f"配置参数 {name} 超过七个值；没有写文件")
    if not all(math.isfinite(value) for value in old):
        raise ValueError(f"配置参数 {name} 含非有限数；没有写文件")
    newline = "\r\n" if lines[key_index].endswith("\r\n") else "\n"
    for offset, value in enumerate(values, 1):
        lines[key_index + offset] = f"{indent}- {float(value):.9f}{newline}"
    return "".join(lines)


def save_default_corrections(
    config_file: Path, updates: Mapping[str, Sequence[float]]
) -> tuple[Path, Path]:
    """Update one fragment and its semantic manifest, retaining both originals."""
    config_file = config_file.resolve(strict=True)
    if not config_file.is_file() or not updates:
        raise ValueError("源配置文件不存在或没有待写参数")
    before = config_file.read_text(encoding="utf-8")
    after = before
    for name, values in updates.items():
        after = _replace_seven_value_sequence(after, name, values)
    manifest_file = config_file.parent / "manifest.yaml"
    manifest_before = manifest_file.read_text(encoding="utf-8")
    manifest = yaml.safe_load(manifest_before)
    merged: dict = {}
    for filename in manifest["fragment_order"]:
        fragment = config_file.parent / filename
        content = after if fragment.resolve() == config_file else fragment.read_text(encoding="utf-8")
        parameters = yaml.safe_load(content)["mission_controller"]["ros__parameters"]
        if set(merged).intersection(parameters):
            raise ValueError(f"{fragment} 与其他配置片段存在重复参数")
        merged.update(parameters)
    digest = hashlib.sha256(
        json.dumps(merged, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    manifest_after, count_matches = re.subn(
        r"(?m)^parameter_count:[ \t]*\d+[ \t]*$",
        f"parameter_count: {len(merged)}",
        manifest_before,
    )
    manifest_after, hash_matches = re.subn(
        r"(?m)^semantic_sha256:[ \t]*[0-9a-f]{64}[ \t]*$",
        f"semantic_sha256: {digest}",
        manifest_after,
    )
    if count_matches != 1 or hash_matches != 1:
        raise ValueError("manifest.yaml 的计数或摘要字段缺失/重复；没有写文件")
    if after == before and manifest_after == manifest_before:
        return config_file, manifest_file
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    backup = config_file.with_name(f"{config_file.name}.bak_calibration_{stamp}")
    manifest_backup = manifest_file.with_name(f"manifest.yaml.bak_calibration_{stamp}")
    shutil.copy2(config_file, backup)
    shutil.copy2(manifest_file, manifest_backup)
    try:
        _atomic_write(config_file, after)
        _atomic_write(manifest_file, manifest_after)
    except Exception:
        shutil.copy2(backup, config_file)
        shutil.copy2(manifest_backup, manifest_file)
        raise
    return backup, manifest_backup


def _atomic_write(path: Path, content: str) -> None:
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_name, stat.S_IMODE(path.stat().st_mode))
        os.replace(tmp_name, path)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def apply_and_save(
    *,
    config_file: Path,
    results: Mapping[str, Sequence[float]],
    previous_runtime: Mapping[str, Sequence[float]],
    param_set: Callable[[str, Sequence[float]], None],
) -> tuple[Path, Path]:
    """Update live, source and installed defaults; roll all back on failure."""
    config_files = _workspace_config_files(config_file)
    originals = {
        path: (
            path.read_text(encoding="utf-8"),
            (path.parent / "manifest.yaml").read_text(encoding="utf-8"),
        )
        for path in config_files
    }
    applied: list[str] = []
    saved: list[Path] = []
    try:
        for name, values in results.items():
            param_set(name, values)
            applied.append(name)
        first_backup = None
        for path in config_files:
            backup = save_default_corrections(path, results)
            saved.append(path)
            if first_backup is None:
                first_backup = backup
        return first_backup
    except Exception as error:
        rollback_errors = []
        for path in reversed(saved):
            original, manifest_original = originals[path]
            try:
                _atomic_write(path, original)
                _atomic_write(path.parent / "manifest.yaml", manifest_original)
            except Exception as restore_error:
                rollback_errors.append(f"{path}: {restore_error}")
        for name in reversed(applied):
            try:
                param_set(name, previous_runtime[name])
            except Exception as restore_error:
                rollback_errors.append(f"runtime {name}: {restore_error}")
        if rollback_errors:
            raise RuntimeError(
                f"保存标定失败且回滚不完整：{'; '.join(rollback_errors)}"
            ) from error
        raise
