"""Entity universe for the synthetic retail world.

Store-level detail is generated always; the report grain (week x banner x
state) is produced by the pipeline stages, so lifecycle and reclassification
faults remain representable at every grain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import UniverseConfig

AU_STATES: tuple[tuple[str, str], ...] = (
    ("NSW", "New South Wales"),
    ("VIC", "Victoria"),
    ("QLD", "Queensland"),
    ("WA", "Western Australia"),
    ("SA", "South Australia"),
    ("TAS", "Tasmania"),
    ("ACT", "Australian Capital Territory"),
    ("NT", "Northern Territory"),
)

COMMODITY_NAMES: tuple[str, ...] = (
    "Analgesics",
    "Vitamins",
    "Cough & Cold",
    "First Aid",
    "Oral Care",
    "Skin Care",
    "Baby Care",
    "Beauty",
    "Fragrance",
    "Household",
    "Confectionery",
    "Snacks",
    "Beverages",
    "Pet Care",
    "Electricals",
)


@dataclass(frozen=True)
class Universe:
    states: pd.DataFrame
    banners: pd.DataFrame
    stores: pd.DataFrame
    commodities: pd.DataFrame
    products: pd.DataFrame
    assortment: pd.DataFrame

    def dimensions(self) -> dict[str, pd.DataFrame]:
        return {
            "states": self.states,
            "banners": self.banners,
            "stores": self.stores,
            "commodities": self.commodities,
            "products": self.products,
        }


def generate_universe(
    config: UniverseConfig,
    rng: np.random.Generator,
    horizon_weeks: int = 104,
) -> Universe:
    if config.n_states > len(AU_STATES):
        raise ValueError(f"n_states cannot exceed {len(AU_STATES)}")
    if config.n_commodities > len(COMMODITY_NAMES):
        raise ValueError(f"n_commodities cannot exceed {len(COMMODITY_NAMES)}")

    state_rows = AU_STATES[: config.n_states]
    states = pd.DataFrame(
        {
            "state_id": [code for code, _ in state_rows],
            "state_name": [name for _, name in state_rows],
        }
    )

    banner_ids = [f"B{i + 1:02d}" for i in range(config.n_banners)]
    home_states = rng.choice(states["state_id"].to_numpy(), size=config.n_banners)
    banners = pd.DataFrame(
        {
            "banner_id": banner_ids,
            "banner_name": [f"Banner {i + 1:02d}" for i in range(config.n_banners)],
            "home_state": home_states,
        }
    )

    store_ids = [f"S{i + 1:03d}" for i in range(config.n_stores)]
    banner_ids_array = banners["banner_id"].to_numpy()
    # Guarantee every banner has at least one store when the universe allows it;
    # a banner with no stores would make some lifecycle and report checks vacuous.
    guaranteed = banner_ids_array[: min(config.n_stores, config.n_banners)]
    remaining = rng.choice(
        banner_ids_array, size=max(0, config.n_stores - len(guaranteed))
    )
    store_banner = rng.permutation(np.concatenate([guaranteed, remaining]))
    banner_state = dict(zip(banners["banner_id"], banners["home_state"], strict=True))
    open_weeks = np.ones(config.n_stores, dtype=np.int32)
    # A small share of stores open part-way through the history.
    n_late = max(1, config.n_stores // 8)
    if horizon_weeks > 4 and n_late < config.n_stores:
        late_idx = rng.choice(config.n_stores, size=n_late, replace=False)
        upper = max(3, min(horizon_weeks, 52))
        open_weeks[late_idx] = rng.integers(2, upper + 1, size=n_late).astype(np.int32)
    stores = pd.DataFrame(
        {
            "store_id": store_ids,
            "store_name": [f"Store {i + 1:03d}" for i in range(config.n_stores)],
            "banner_id": store_banner,
            "state_id": [banner_state[b] for b in store_banner],
            "open_week": open_weeks,
        }
    )

    commodity_ids = [f"C{i + 1:02d}" for i in range(config.n_commodities)]
    commodities = pd.DataFrame(
        {
            "commodity_id": commodity_ids,
            "commodity_name": list(COMMODITY_NAMES[: config.n_commodities]),
        }
    )

    product_ids = [f"P{i + 1:04d}" for i in range(config.n_products)]
    product_commodity = rng.choice(commodity_ids, size=config.n_products)
    is_script = rng.random(config.n_products) < config.script_fraction
    popularity = rng.lognormal(mean=0.0, sigma=0.6, size=config.n_products)
    products = pd.DataFrame(
        {
            "product_id": product_ids,
            "product_name": [f"Product {i + 1:04d}" for i in range(config.n_products)],
            "commodity_id": product_commodity,
            "is_script": is_script,
            "base_popularity": popularity,
        }
    )

    n_carried = max(1, int(round(config.n_products * config.assortment_fraction)))
    weights = popularity / popularity.sum()
    rows: list[dict[str, object]] = []
    for store_id in store_ids:
        idx = rng.choice(config.n_products, size=n_carried, replace=False, p=weights)
        rows.extend(
            {"store_id": store_id, "product_id": product_ids[int(i)]}
            for i in np.sort(idx)
        )
    assortment = pd.DataFrame(rows)

    return Universe(
        states=states,
        banners=banners,
        stores=stores,
        commodities=commodities,
        products=products,
        assortment=assortment,
    )
