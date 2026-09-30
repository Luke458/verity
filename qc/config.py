"""Engine configuration.

Defaults match the synthetic retail world so Milestone A can be developed and
tested without configuration; YAML files override any field for real datasets.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
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
    # Effective unit-value decomposition: value measure divided by quantity.
    unit_value_metric: str = "dollar"
    unit_value_quantity: str = "units"

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
    # Columns scanned for aggregate marker values ("TOTAL", "ALL", ...) that
    # would double count if mixed into detail rows. Parent/child reconciliation
    # itself is done by comparing the analysis and report frames per key.
    hierarchy_columns: tuple[tuple[str, ...], ...] = (
        ("banner_id",),
        ("state_id",),
    )
    # Declared intersection views (for example commodity-by-banner). Each is
    # checked against its first key's parent, never added to the national total.
    hierarchy_intersections: tuple[tuple[str, ...], ...] = ()
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

    calendar: str = "sequential_week"
    period_map: tuple[tuple[str, int], ...] = ()
    # Declared business calendar: the first day of ``calendar_anchor_week`` and
    # the annual events (name, rule, lead_weeks, lag_weeks). No anchor means no
    # calendar evidence; the engine never assumes a geography.
    calendar_anchor_date: str = ""
    calendar_anchor_week: int = 1
    calendar_events: tuple[tuple[str, str, int, int], ...] = ()
    snapshot_metrics: tuple[str, ...] = ("stock",)
    required_dimensions: tuple[str, ...] = ()
    temporal_required: bool = True
    # Explicit temporal measures. Empty values derive required measures from
    # the declared required columns and optional measures from the remaining
    # metric columns; snapshot measures are compared within a period and are
    # never summed across time.
    temporal_required_metrics: tuple[str, ...] = ()
    temporal_optional_metrics: tuple[str, ...] = ()
    optional_checks: tuple[str, ...] = ()
    absent_entity_policy: str = "zero"
    stage_keys: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", self.name):
            raise ValueError("name must be a safe dataset identifier")
        if self.calendar not in ("sequential_week", "weekly_date", "mapped"):
            raise ValueError("unsupported calendar")
        if self.calendar == "mapped" and not self.period_map:
            raise ValueError("encoded calendars require period_map")
        for field in fields(self):
            value, default = getattr(self, field.name), field.default
            if isinstance(default, bool) and not isinstance(value, bool):
                raise ValueError(f"{field.name} must be boolean")
            if isinstance(default, int) and not isinstance(default, bool):
                if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                    raise ValueError(f"{field.name} must be a positive integer")
            if isinstance(default, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                    raise ValueError(f"{field.name} must be finite and nonnegative")
                if ("fraction" in field.name or "percentile" in field.name or "breadth" in field.name) and value > 1:
                    raise ValueError(f"{field.name} must be <= 1")
            if isinstance(default, tuple) and not isinstance(value, tuple):
                raise ValueError(f"{field.name} must be a sequence")
        stages = [stage for stage, _ in self.stage_keys]
        if len(set(stages)) != len(stages):
            raise ValueError("stage_keys must have unique stages")
        seen_intersections: set[tuple[str, ...]] = set()
        for keys in self.hierarchy_intersections:
            if len(keys) < 2 or any(not key for key in keys) or len(set(keys)) != len(keys):
                raise ValueError("hierarchy intersections require unique keys")
            if keys in seen_intersections:
                raise ValueError("hierarchy intersections must be unique")
            seen_intersections.add(keys)
        for keys in (self.entity_key_columns, self.report_grain, *(keys for _, keys in self.stage_keys)):
            if any(not isinstance(key, str) or not key for key in keys) or len(set(keys)) != len(keys) or self.week_column in keys:
                raise ValueError("grain keys must be unique column names excluding the period")
        if self.absent_entity_policy not in ("zero", "missing"):
            raise ValueError("absent_entity_policy must be zero or missing")
        if self.primary_metric in self.snapshot_metrics:
            raise ValueError("snapshot measures cannot be the time-aggregated primary metric")
        if self.calendar_anchor_date:
            try:
                _dt.date.fromisoformat(self.calendar_anchor_date)
            except ValueError as error:
                raise ValueError("calendar_anchor_date must be an ISO date") from error
        if self.calendar_events:
            if not self.calendar_anchor_date:
                raise ValueError("calendar events require calendar_anchor_date")
            names = [name for name, _, _, _ in self.calendar_events]
            if any(not name for name in names) or len(set(names)) != len(names):
                raise ValueError("calendar event names must be unique and nonempty")
            for _, rule, lead, lag in self.calendar_events:
                if rule != "easter" and not re.fullmatch(r"\d{2}-\d{2}", str(rule)):
                    raise ValueError("calendar event rule must be easter or MM-DD")
                if type(lead) is not int or type(lag) is not int or lead < 0 or lag < 0:
                    raise ValueError("calendar event lead/lag weeks must be nonnegative")
        if not 0 < self.temporal_interval_alpha < 1:
            raise ValueError("temporal_interval_alpha must be in (0, 1)")
        if not 0 < self.temporal_fdr_q < 1:
            raise ValueError("temporal_fdr_q must be in (0, 1)")
        if self.recurrence_minimum > self.recurrence_window:
            raise ValueError("recurrence_minimum cannot exceed recurrence_window")
        if self.period_map:
            if len(dict(self.period_map)) != len(self.period_map) or any(type(v) is not int for _, v in self.period_map):
                raise ValueError("period_map must have unique labels and integer indices")
            if len({v for _, v in self.period_map}) != len(self.period_map):
                raise ValueError("period_map indices must be unique")
        if self.primary_metric not in self.metric_columns:
            raise ValueError("primary_metric must be declared in metric_columns")
        for declared in (self.temporal_required_metrics, self.temporal_optional_metrics):
            for metric in declared:
                if metric not in self.metric_columns:
                    raise ValueError(
                        f"temporal metric {metric!r} must be declared in metric_columns"
                    )
        if set(self.temporal_required_metrics) & set(self.temporal_optional_metrics):
            raise ValueError(
                "temporal required and optional metrics must be disjoint"
            )
        if self.temporal_required_metrics and (
            self.primary_metric not in self.temporal_required_metrics
        ):
            raise ValueError(
                "primary_metric must be a required temporal metric"
            )
        if self.unit_value_metric and self.unit_value_metric not in self.metric_columns:
            raise ValueError("unit_value_metric must be declared in metric_columns")
        if self.unit_value_quantity and self.unit_value_quantity not in self.metric_columns:
            raise ValueError("unit_value_quantity must be declared in metric_columns")
        if not self.forecast_quantiles or any(not 0 < q < 1 for q in self.forecast_quantiles):
            raise ValueError("forecast quantiles must be in (0, 1)")
        if tuple(sorted(set(self.forecast_quantiles))) != self.forecast_quantiles:
            raise ValueError("forecast quantiles must be sorted and unique")
        if self.temporal_lower_percentile >= self.temporal_upper_percentile:
            raise ValueError("invalid temporal percentile bounds")
        for index in (0, 1):
            names = [item[index] for item in self.column_map]
            if len(names) != len(set(names)):
                raise ValueError("column_map names must be unique")

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
    # Distribution drift over the appended period. Off by default: the gate in
    # config/drift-gate.json passed on synthetic data only, and enabling a
    # detector before it has real-data calibration is how false-positive
    # generators get promoted. The threshold is derived per scope from that
    # scope's own consecutive-week PSI distribution, never from a global
    # constant, because PSI is sample-size dependent.
    distribution_drift_enabled: bool = False
    distribution_drift_bins: int = 10
    distribution_drift_reference_weeks: int = 8
    distribution_drift_abs_floor: float = 0.01
    distribution_drift_quantile: float = 0.95
    # Entity presence reliability. An entity absent from the appended period is
    # only evidence of a coverage regression if it traded in at least this
    # fraction of its own trailing window: on a sparse transactional source a
    # product that trades one week in five is expected to be absent next week.
    entity_presence_threshold: float = 0.5
    entity_presence_window: int = 8
    # Absolute materiality floor per metric/period/scope; never an accumulated
    # historical total.
    temporal_materiality_abs: float = 0.0
    temporal_lower_percentile: float = 0.01
    temporal_upper_percentile: float = 0.99
    temporal_backtest_origins: int = 26
    # The latest origins are reserved exclusively for interval calibration and
    # never reused for model selection.
    temporal_calibration_origins: int = 12
    # Annual-seasonality explanations require at least two full retail years.
    temporal_min_annual_history: int = 104
    # Reported prediction intervals use the held-out residual quantiles.
    temporal_interval_alpha: float = 0.1
    # Sparse leaves fall back to parent expectation x historical child share;
    # the fallback is labelled and cannot support statistical clearance.
    temporal_sparse_fallback: bool = True
    # Verified certificates may clear their specific finding without a new
    # human approval; integrity failures still always require review. A pinned
    # qualification artifact is required before any automatic clearance.
    statistical_clearance_enabled: bool = True
    qualification_path: str = ""
    # Leaf anomaly screening controls false discoveries before escalation.
    temporal_fdr_enabled: bool = True
    temporal_fdr_q: float = 0.05
    # Recurrence escalation: a finding repeated in at least
    # ``recurrence_minimum`` of the last ``recurrence_window`` refreshes whose
    # cumulative impact is material escalates to review.
    recurrence_enabled: bool = True
    recurrence_window: int = 3
    recurrence_minimum: int = 2
    # Registered cumulative materiality budget as a multiple of the largest
    # scoped per-period threshold. Summing each observation's threshold cannot
    # detect a sequence whose members all stay individually below threshold.
    recurrence_budget_ratio: float = 1.0
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

    def required_temporal_metrics(self) -> tuple[str, ...]:
        """Measures whose absence makes the temporal assessment incomplete."""
        if self.temporal_required_metrics:
            return self.temporal_required_metrics
        return tuple(
            metric
            for metric in self.metric_columns
            if metric in self.required_columns
            and metric not in self.snapshot_metrics
        )

    def optional_temporal_metrics(self) -> tuple[str, ...]:
        """Declared measures assessed when present and reported when absent."""
        if self.temporal_optional_metrics:
            return self.temporal_optional_metrics
        required = set(self.required_temporal_metrics())
        return tuple(
            metric for metric in self.metric_columns if metric not in required
        )

    def temporal_metrics(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.required_temporal_metrics(),
                    *self.optional_temporal_metrics(),
                )
            )
        )

    def analysis_stages(self) -> tuple[str, ...]:
        return (self.analysis_stage, *self.stage_preference)

    def contract_stages(self) -> tuple[str, ...]:
        return (self.contract_stage, *self.stage_preference)

    def to_dict(self) -> dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def _calendar_events(value: Any) -> tuple[tuple[str, str, int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("calendar_events must be a list of event declarations")
    events: list[tuple[str, str, int, int]] = []
    for item in value:
        if isinstance(item, Mapping):
            name = item.get("name")
            rule = item.get("rule")
            lead = item.get("lead_weeks", 0)
            lag = item.get("lag_weeks", 0)
        elif isinstance(item, (list, tuple)) and len(item) == 4:
            name, rule, lead, lag = item
        else:
            raise ValueError("calendar events require name, rule and optional lead/lag weeks")
        events.append((str(name), str(rule), int(lead), int(lag)))
    return tuple(events)


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
        if key == "stage_keys":
            if not isinstance(value, Mapping):
                raise ValueError("stage_keys must map stages to key lists")
            converted[key] = tuple((str(stage), tuple(keys)) for stage, keys in value.items())
        elif key == "period_map":
            if not isinstance(value, Mapping):
                raise ValueError("period_map must map labels to sequential week indices")
            converted[key] = tuple((str(label), index) for label, index in value.items())
        elif key == "column_map":
            if not isinstance(value, Mapping):
                raise ValueError("column_map must be a mapping of canonical: production")
            converted[key] = tuple(
                (str(canonical), str(production))
                for canonical, production in value.items()
            )
        elif key == "calendar_events":
            converted[key] = _calendar_events(value)
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
