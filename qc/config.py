"""Engine configuration.

Defaults match the synthetic retail world so Milestone A can be developed and
tested without configuration; YAML files override any field for real datasets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str = "synthetic-retail"
    week_column: str = "week"
    entity_columns: tuple[str, ...] = (
        "store_id",
        "product_id",
        "commodity_id",
        "banner_id",
        "state_id",
    )
    metric_columns: tuple[str, ...] = ("dollar", "units", "scripts", "stock")
    count_entity_map: tuple[tuple[str, str], ...] = (
        ("store_count", "store_id"),
        ("product_count", "product_id"),
    )
    primary_metric: str = "dollar"

    # Stage selection: contracts run on the final report table, analysis runs on
    # the entity-bearing table.
    analysis_stage: str = "warehouse"
    contract_stage: str = "report"
    stage_preference: tuple[str, ...] = ("warehouse", "coded", "source", "report")

    required_columns: tuple[str, ...] = ("week", "dollar", "units")
    entity_key_columns: tuple[str, ...] = ("store_id", "product_id")
    report_grain: tuple[str, ...] = ("banner_id", "state_id")
    # Canonical name -> production column name, e.g. {"product_id": "pfc"}.
    # Applied by a source decorator so the whole engine sees canonical names.
    column_map: tuple[tuple[str, str], ...] = ()
    # Contributions are summed at the first entity level present, highest
    # priority first, to avoid double counting parent and child events.
    explanation_entity_types: tuple[str, ...] = (
        "store",
        "banner",
        "state",
        "product",
        "commodity",
    )

    explained_fraction_threshold: float = 0.9
    reconstruction_score_threshold: float = 0.9
    materiality_ratio: float = 0.001
    materiality_abs: float = 0.0
    broad_recalculation_breadth: float = 0.5
    lineage_materiality_ratio: float = 1e-5
    reconciliation_tolerance: float = 1e-6
    hierarchy_tolerance: float = 1e-5
    hierarchy_columns: tuple[tuple[str, ...], ...] = (
        ("banner_id",),
        ("state_id",),
    )
    hierarchy_total_markers: tuple[str, ...] = (
        "total",
        "all",
        "*",
        "grand total",
        "subtotal",
    )
    price_outlier_k: float = 4.0
    null_fraction_limit: float = 0.01
    duplicate_fraction_limit: float = 0.005
    top_contributors: int = 10
    pipeline_order: tuple[str, ...] = ("source", "coded", "warehouse", "report")

    def column_map_dict(self) -> dict[str, str]:
        """Canonical name -> production column name."""
        return dict(self.column_map)

    def source_rename(self) -> dict[str, str]:
        """Production column name -> canonical name, for the source layer."""
        return {
            production: canonical
            for canonical, production in self.column_map_dict().items()
        }

    # Milestone C: latest-week temporal intelligence.
    temporal_enabled: bool = True
    temporal_entity_columns: tuple[str, ...] = ("banner_id", "commodity_id")
    temporal_min_history: int = 26
    temporal_window: int = 13
    temporal_season: int = 52
    temporal_z_threshold: float = 3.0
    temporal_change_point_threshold: float = 5.0
    temporal_min_relative_residual: float = 0.02
    temporal_lower_percentile: float = 0.01
    temporal_upper_percentile: float = 0.99
    temporal_backtest_origins: int = 26
    forecast_quantiles: tuple[float, ...] = (
        0.01,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        0.99,
    )
    forecaster: str = "baseline"
    chronos_model: str = "amazon/chronos-2"
    sarimax_order: tuple[int, ...] = (1, 0, 1)
    sarimax_seasonal_order: tuple[int, ...] = (1, 0, 1, 52)

    # Milestone D: semantic decisions.
    decision_enabled: bool = True

    # Entity relationship detection (section 12): conservative candidates.
    relationship_correlation_threshold: float = 0.9
    relationship_min_weeks: int = 4
    relationship_ratio_bounds: tuple[float, float] = (0.5, 2.0)

    def analysis_stages(self) -> tuple[str, ...]:
        return (self.analysis_stage, *self.stage_preference)

    def contract_stages(self) -> tuple[str, ...]:
        return (self.contract_stage, *self.stage_preference)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if raw is None:
        return DatasetConfig()
    if not isinstance(raw, Mapping):
        raise ValueError("dataset config must be a mapping")
    config = DatasetConfig()
    known = {field.name for field in fields(config)}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"unknown dataset config keys: {sorted(unknown)}")
    converted: dict[str, Any] = {}
    for key, value in raw.items():
        current = getattr(config, key)
        if key == "column_map":
            if not isinstance(value, Mapping):
                raise ValueError("column_map must be a mapping of canonical: production")
            converted[key] = tuple(
                (str(canonical), str(production))
                for canonical, production in value.items()
            )
        elif isinstance(current, tuple) and isinstance(value, (list, tuple)):
            if current and isinstance(current[0], tuple):
                converted[key] = tuple(tuple(item) for item in value)
            else:
                converted[key] = tuple(value)
        else:
            converted[key] = value
    config = replace(config, **converted)
    canonical_names = [canonical for canonical, _ in config.column_map]
    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError("column_map canonical names must be unique")
    production_names = [production for _, production in config.column_map]
    if len(production_names) != len(set(production_names)):
        raise ValueError("column_map production names must be unique")
    return config
