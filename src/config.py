"""Configuration loading, merging and validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "default_config.json"
USER_PATH = ROOT / "config" / "user_config.json"


class ConfigError(ValueError):
    """Raised when a configuration is invalid."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def validate_config(cfg: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems (empty list == valid)."""
    errs: List[str] = []

    def num(key, lo=None, hi=None, section=cfg):
        v = section.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"'{key}' must be a number (got {v!r}).")
            return
        if lo is not None and v < lo:
            errs.append(f"'{key}' must be >= {lo} (got {v}).")
        if hi is not None and v > hi:
            errs.append(f"'{key}' must be <= {hi} (got {v}).")

    num("confidence_threshold", 0.0, 1.0)
    num("persistence_s", 0.0, 3600)
    num("exit_grace_s", 0.0, 60)
    num("dedup_cooldown_s", 0.0, 3600)
    num("frame_stride", 1, 60)
    num("process_width", 160, 4096)
    num("low_conf_min_obs", 1, 1000)
    if cfg.get("detector") not in ("auto", "yolo", "hog"):
        errs.append("'detector' must be one of: auto, yolo, hog.")
    zone = cfg.get("zone")
    if not isinstance(zone, dict) or "polygon" not in zone:
        errs.append("'zone.polygon' is required.")
    else:
        from .zone_manager import ZoneError, Zone
        try:
            Zone(zone.get("name", "Zone"), zone["polygon"])
        except ZoneError as e:
            errs.append(f"zone: {e}")
    ret = cfg.get("retention", {})
    if not isinstance(ret, dict):
        errs.append("'retention' must be an object.")
    else:
        num("max_days", 1, 3650, ret)
        num("max_records", 10, 10_000_000, ret)
    trk = cfg.get("tracker", {})
    if not isinstance(trk, dict):
        errs.append("'tracker' must be an object.")
    else:
        num("iou_threshold", 0.01, 1.0, trk)
        num("max_missed", 0, 1000, trk)
    return errs


def load_config(path: Path | str | None = None, use_user: bool = True) -> Dict[str, Any]:
    """Load defaults, overlay user config, validate. Raises ConfigError."""
    try:
        cfg = json.loads(Path(DEFAULT_PATH).read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"Cannot read default config: {e}") from e
    override_path = Path(path) if path else (USER_PATH if use_user else None)
    if override_path and override_path.exists():
        try:
            cfg = _deep_merge(cfg, json.loads(override_path.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            raise ConfigError(f"Cannot read config {override_path}: {e}") from e
    errs = validate_config(cfg)
    if errs:
        raise ConfigError("Invalid configuration: " + " | ".join(errs))
    return cfg


def save_user_config(cfg: Dict[str, Any], path: Path | str = USER_PATH) -> None:
    errs = validate_config(cfg)
    if errs:
        raise ConfigError("Refusing to save invalid configuration: " + " | ".join(errs))
    Path(path).write_text(json.dumps(cfg, indent=2))
