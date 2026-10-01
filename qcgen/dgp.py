"""Data-generating process for the synthetic retail world.

Produces store x product x week truth with trend, seasonality, moving holidays,
promotions and noise. The truth is generated once; every version snapshot is a
deterministic transform of it, so revision differences are exactly the
injected faults plus the appended week.
"""

from __future__ import annotations

import datetime as _dt

import numpy as np
import pandas as pd

from .config import HistoryConfig
from .universe import Universe


def week_end_dates(start_week: str, n_weeks: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(start_week)
    if start.weekday() != 5:
        raise ValueError(
            f"start_week must be a Saturday, got {start.date()} ({start.day_name()})"
        )
    return pd.date_range(start=start, periods=n_weeks, freq="W-SAT")


def easter_sunday(year: int) -> _dt.date:
    """Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ell = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ell) // 451
    month = (h + ell - 7 * m + 114) // 31
    day = ((h + ell - 7 * m + 114) % 31) + 1
    return _dt.date(year, month, day)


def _week_index_for(dates: pd.DatetimeIndex, when: _dt.date) -> int | None:
    starts = dates - pd.Timedelta(days=6)
    for i in range(len(dates)):
        if starts[i].date() <= when <= dates[i].date():
            return i
    return None


def seasonal_factor(dates: pd.DatetimeIndex) -> np.ndarray:
    woy = dates.isocalendar().week.to_numpy(dtype=np.float64)
    return (
        1.0
        + 0.05 * np.sin(2 * np.pi * woy / 52.18)
        + 0.02 * np.cos(4 * np.pi * woy / 52.18)
    )


def holiday_boosts(dates: pd.DatetimeIndex, config: HistoryConfig) -> np.ndarray:
    boost = np.ones(len(dates))
    for year in range(dates[0].year, dates[-1].year + 1):
        fixed = (
            (_dt.date(year, 12, 25), config.christmas_boost),
            (_dt.date(year, 6, 30), config.eofy_boost),
            (easter_sunday(year), config.easter_boost),
        )
        for when, mult in fixed:
            idx = _week_index_for(dates, when)
            if idx is not None:
                boost[idx] = max(boost[idx], mult)
    return boost


def _realism_factor(
    frame: pd.DataFrame,
    dates: pd.DatetimeIndex,
    products: pd.DataFrame,
    config: HistoryConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Commodity seasonality and market/commodity shocks (all off by default)."""
    factor = np.ones(len(frame))
    commodities = sorted(str(value) for value in products["commodity_id"].unique())
    commodity_index = {value: index for index, value in enumerate(commodities)}
    row_commodity = frame["commodity_id"].astype(str).map(commodity_index).to_numpy()
    row_week = frame["week"].to_numpy(dtype=np.int64) - 1
    if config.commodity_season_amplitude > 0:
        amplitude = rng.uniform(0.0, config.commodity_season_amplitude, len(commodities))
        phase = rng.uniform(0.0, 52.18, len(commodities))
        woy = dates.isocalendar().week.to_numpy(dtype=np.float64)
        season = 1.0 + amplitude[:, None] * np.sin(
            2 * np.pi * (woy[None, :] - phase[:, None]) / 52.18
        )
        factor *= season[row_commodity, row_week]
    if config.market_shock_sigma > 0:
        shocks = rng.lognormal(0.0, config.market_shock_sigma, len(dates))
        factor *= shocks[row_week]
    if config.commodity_shock_sigma > 0:
        shocks = rng.lognormal(
            0.0, config.commodity_shock_sigma, (len(commodities), len(dates))
        )
        factor *= shocks[row_commodity, row_week]
    return factor


def _apply_intermittency(
    frame: pd.DataFrame,
    products: pd.DataFrame,
    config: HistoryConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Drop non-trading store-weeks of intermittently selling products."""
    if config.intermittency <= 0:
        return frame
    product_ids = products["product_id"].astype(str).to_numpy()
    slow = rng.random(len(product_ids)) < config.intermittency
    trade_probability = np.where(slow, rng.uniform(0.25, 0.75, len(product_ids)), 1.0)
    probability = frame["product_id"].astype(str).map(
        dict(zip(product_ids, trade_probability, strict=True))
    ).to_numpy(dtype=np.float64)
    keep = rng.random(len(frame)) < probability
    return frame.loc[keep].reset_index(drop=True)


def generate_history(
    universe: Universe,
    config: HistoryConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    dates = week_end_dates(config.start_week, config.n_weeks)
    n_weeks = config.n_weeks

    week_factor = (seasonal_factor(dates) * holiday_boosts(dates, config)).astype(
        np.float64
    )
    trend = (1.0 + config.trend_annual) ** (np.arange(n_weeks) / 52.0)
    week_factor = week_factor * trend
    week_factor_series = pd.Series(week_factor, index=np.arange(1, n_weeks + 1))

    stores = universe.stores
    products = universe.products

    store_factor = pd.Series(
        rng.lognormal(mean=0.0, sigma=0.30, size=len(stores)),
        index=stores["store_id"],
    )
    base_price = np.where(products["is_script"].to_numpy(), 45.0, 12.0)
    price = pd.Series(
        base_price * rng.lognormal(mean=0.0, sigma=0.35, size=len(products)),
        index=products["product_id"],
    )

    weeks = pd.DataFrame({"week": np.arange(1, n_weeks + 1, dtype=np.int32)})
    frame = universe.assortment.merge(weeks, how="cross")
    frame = frame.merge(
        products[["product_id", "commodity_id", "base_popularity", "is_script"]],
        on="product_id",
        how="left",
    )
    frame = frame.merge(stores[["store_id", "open_week"]], on="store_id", how="left")
    frame["store_factor"] = frame["store_id"].map(store_factor).astype(np.float64)
    frame["price"] = frame["product_id"].map(price).astype(np.float64)
    frame["week_factor"] = frame["week"].map(week_factor_series).astype(np.float64)
    frame = frame.loc[frame["week"] >= frame["open_week"]].reset_index(drop=True)

    frame["week_factor"] = frame["week_factor"] * _realism_factor(
        frame, dates, products, config, rng
    )
    frame = _apply_intermittency(frame, products, config, rng)

    n = len(frame)
    promo = rng.random(n) < config.promo_fraction
    promo_mult = np.where(promo, rng.uniform(1.3, config.promo_lift, n), 1.0)
    noise = rng.lognormal(mean=0.0, sigma=config.noise_sigma, size=n)

    units_f = (
        config.base_units_mean
        * frame["base_popularity"].to_numpy(dtype=np.float64)
        * frame["store_factor"].to_numpy(dtype=np.float64)
        * frame["week_factor"].to_numpy(dtype=np.float64)
        * promo_mult
        * noise
    )
    units = np.rint(units_f).astype(np.int32)
    dollar = np.round(
        units.astype(np.float64)
        * frame["price"].to_numpy(dtype=np.float64)
        * rng.lognormal(mean=0.0, sigma=0.03, size=n),
        2,
    ).astype(np.float32)
    scripts = np.where(frame["is_script"].to_numpy(), units, 0).astype(np.float32)
    stock = (
        units.astype(np.float64)
        * config.stock_cover_weeks
        * rng.lognormal(mean=0.0, sigma=0.15, size=n)
    ).astype(np.float32)

    fact = pd.DataFrame(
        {
            "week": frame["week"].to_numpy(dtype=np.int32),
            "store_id": frame["store_id"].astype(str).to_numpy(),
            "product_id": frame["product_id"].astype(str).to_numpy(),
            "units": units,
            "dollar": dollar,
            "scripts": scripts,
            "stock": stock,
        }
    )
    fact = fact.sort_values(
        ["week", "store_id", "product_id"], kind="stable"
    ).reset_index(drop=True)
    return fact
