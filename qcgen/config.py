"""Configuration for the synthetic retail world.

Profiles are code-level defaults; ``config/datasets/*.yaml`` files override any
field. No hidden global state: every generator takes an explicit config and a
seeded numpy Generator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

STAGES: tuple[str, ...] = ("source", "coded", "warehouse", "report")

FAULT_FAMILIES: tuple[str, ...] = (
    "missing_stores",
    "missing_products",
    "entity_merge",
    "new_store_backfill",
    "history_truncation",
    "commodity_remap",
    "coding_error",
    "warehouse_transform_error",
    "recalculation",
    "schema_failure",
    "null_duplicate_storm",
    "expected_event",
)

CONTROL_FAMILIES: tuple[str, ...] = ("market_movement",)

ALL_FAMILIES: tuple[str, ...] = FAULT_FAMILIES + CONTROL_FAMILIES


@dataclass(frozen=True)
class UniverseConfig:
    n_banners: int = 4
    n_states: int = 3
    n_stores: int = 24
    n_products: int = 80
    n_commodities: int = 8
    script_fraction: float = 0.2
    assortment_fraction: float = 0.6

    def __post_init__(self) -> None:
        if self.n_banners < 1:
            raise ValueError("n_banners must be >= 1")
        if self.n_states < 1:
            raise ValueError("n_states must be >= 1")
        if self.n_stores < 1:
            raise ValueError("n_stores must be >= 1")
        if self.n_products < 1:
            raise ValueError("n_products must be >= 1")
        if not 0 <= self.script_fraction <= 1:
            raise ValueError("script_fraction must be in [0, 1]")
        if not 0 < self.assortment_fraction <= 1:
            raise ValueError("assortment_fraction must be in (0, 1]")
        if self.n_commodities < 1 or self.n_commodities > self.n_products:
            raise ValueError("n_commodities must be in [1, n_products]")


@dataclass(frozen=True)
class HistoryConfig:
    n_weeks: int = 104
    start_week: str = "2022-01-01"  # must be a Saturday
    trend_annual: float = 0.04
    noise_sigma: float = 0.08
    promo_fraction: float = 0.05
    promo_lift: float = 1.6
    christmas_boost: float = 1.35
    easter_boost: float = 1.15
    eofy_boost: float = 1.12
    base_units_mean: float = 20.0
    stock_cover_weeks: float = 5.0

    def __post_init__(self) -> None:
        if self.n_weeks < 4:
            raise ValueError("n_weeks must be >= 4")
        if self.noise_sigma < 0:
            raise ValueError("noise_sigma must be >= 0")
        if not 0 <= self.promo_fraction <= 1:
            raise ValueError("promo_fraction must be in [0, 1]")


_UNIVERSE_PROFILES: dict[str, UniverseConfig] = {
    "tiny": UniverseConfig(
        n_banners=2,
        n_states=2,
        n_stores=4,
        n_products=24,
        n_commodities=3,
    ),
    "small": UniverseConfig(
        n_banners=4,
        n_states=3,
        n_stores=24,
        n_products=80,
        n_commodities=8,
    ),
    "full": UniverseConfig(
        n_banners=8,
        n_states=5,
        n_stores=60,
        n_products=300,
        n_commodities=15,
    ),
}

_HISTORY_PROFILES: dict[str, HistoryConfig] = {
    "tiny": HistoryConfig(n_weeks=30, start_week="2024-01-06"),
    "small": HistoryConfig(n_weeks=104, start_week="2022-01-01"),
    "full": HistoryConfig(n_weeks=320, start_week="2020-07-04"),
}


@dataclass(frozen=True)
class SuiteConfig:
    profile: str = "small"
    seed: int = 7
    universe: UniverseConfig = field(default_factory=lambda: _UNIVERSE_PROFILES["small"])
    history: HistoryConfig = field(default_factory=lambda: _HISTORY_PROFILES["small"])
    families: tuple[str, ...] = FAULT_FAMILIES
    controls: tuple[str, ...] = CONTROL_FAMILIES
    stages: tuple[str, ...] = ("report",)

    def __post_init__(self) -> None:
        unknown = set(self.families) | set(self.controls)
        unknown -= set(ALL_FAMILIES)
        if unknown:
            raise ValueError(f"unknown fault families: {sorted(unknown)}")
        bad_stages = set(self.stages) - set(STAGES)
        if bad_stages:
            raise ValueError(f"unknown stages: {sorted(bad_stages)}")
        if not self.stages:
            raise ValueError("at least one stage must be selected")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def profile_universe(profile: str) -> UniverseConfig:
    try:
        return replace(_UNIVERSE_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile!r}") from exc


def profile_history(profile: str) -> HistoryConfig:
    try:
        return replace(_HISTORY_PROFILES[profile])
    except KeyError as exc:
        raise ValueError(f"unknown profile: {profile!r}") from exc


def suite_config(profile: str = "small", **overrides: Any) -> SuiteConfig:
    """Build a suite config from a named profile plus explicit overrides."""
    base = SuiteConfig(
        profile=profile,
        universe=profile_universe(profile),
        history=profile_history(profile),
    )
    return replace(base, **overrides)


def dataset_calendar(profile: str = "small") -> dict[str, Any]:
    """Explicit Australian calendar fields for a synthetic profile.

    Synthetic histories are generated on a Saturday-ending week grid with
    Christmas, Easter and EOFY boosts; declaring the same calendar keeps the
    engine's event windows aligned with the generator instead of assuming a
    geography. The returned mapping is expanded into ``DatasetConfig``.
    """
    import datetime as _dt

    start = _dt.date.fromisoformat(profile_history(profile).start_week)
    anchor = start - _dt.timedelta(days=6)
    return {
        "calendar_anchor_date": anchor.isoformat(),
        "calendar_anchor_week": 1,
        "calendar_events": (
            ("christmas", "12-25", 0, 0),
            ("easter", "easter", 0, 0),
            ("eofy", "06-30", 0, 0),
        ),
    }


def _apply_section(base: Any, data: Mapping[str, Any] | None, name: str) -> Any:
    if data is None:
        return base
    if not isinstance(data, Mapping):
        raise ValueError(f"{name} config must be a mapping")
    known = {f.name for f in fields(base)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {name} config keys: {sorted(unknown)}")
    return replace(base, **data)


def load_suite_config(path: str | Path) -> SuiteConfig:
    """Load a suite config from YAML.

    Only explicitly listed fields override the profile defaults; this keeps the
    YAML small and the code the source of truth for defaults.
    """
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise ValueError("dataset config must be a mapping")
    known = {"dataset", "profile", "seed", "universe", "history", "faults", "stages"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")

    profile = str(raw.get("profile", "small"))
    config = suite_config(profile)
    config = replace(
        config,
        universe=_apply_section(config.universe, raw.get("universe"), "universe"),
        history=_apply_section(config.history, raw.get("history"), "history"),
    )

    if "seed" in raw:
        config = replace(config, seed=int(raw["seed"]))

    faults = raw.get("faults")
    if faults is not None:
        if not isinstance(faults, Mapping):
            raise ValueError("faults config must be a mapping")
        unknown = set(faults) - {"include", "controls"}
        if unknown:
            raise ValueError(f"unknown faults config keys: {sorted(unknown)}")
        if "include" in faults:
            config = replace(config, families=tuple(str(f) for f in faults["include"]))
        if "controls" in faults:
            config = replace(config, controls=tuple(str(f) for f in faults["controls"]))

    stages = raw.get("stages")
    if stages is not None:
        if stages == "all":
            config = replace(config, stages=STAGES)
        elif isinstance(stages, str):
            config = replace(config, stages=tuple(s.strip() for s in stages.split(",") if s.strip()))
        else:
            config = replace(config, stages=tuple(str(s) for s in stages))

    return config
