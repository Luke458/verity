from __future__ import annotations

import numpy as np

from qcgen.universe import generate_universe


def test_universe_ids_and_relations(tiny_config):
    config = tiny_config.universe
    universe = generate_universe(config, np.random.default_rng(1), horizon_weeks=30)

    assert len(universe.states) == config.n_states
    assert len(universe.banners) == config.n_banners
    assert len(universe.stores) == config.n_stores
    assert len(universe.products) == config.n_products
    assert len(universe.commodities) == config.n_commodities
    assert universe.stores["store_id"].is_unique
    assert universe.products["product_id"].is_unique

    assert set(universe.stores["banner_id"]) <= set(universe.banners["banner_id"])
    assert set(universe.stores["state_id"]) <= set(universe.states["state_id"])
    assert set(universe.products["commodity_id"]) <= set(
        universe.commodities["commodity_id"]
    )
    # Every banner has at least one store when the universe is large enough.
    assert set(universe.banners["banner_id"]) <= set(universe.stores["banner_id"])

    expected_carried = max(
        1, int(round(config.n_products * config.assortment_fraction))
    )
    assert len(universe.assortment) == config.n_stores * expected_carried
    assert set(universe.assortment["product_id"]) <= set(universe.products["product_id"])
    assert set(universe.assortment["store_id"]) <= set(universe.stores["store_id"])


def test_universe_is_deterministic(tiny_config):
    first = generate_universe(tiny_config.universe, np.random.default_rng(5), 30)
    second = generate_universe(tiny_config.universe, np.random.default_rng(5), 30)
    for name in ("states", "banners", "stores", "commodities", "products", "assortment"):
        assert getattr(first, name).equals(getattr(second, name)), name


def test_universe_differs_with_seed(tiny_config):
    first = generate_universe(tiny_config.universe, np.random.default_rng(5), 30)
    second = generate_universe(tiny_config.universe, np.random.default_rng(6), 30)
    assert not first.stores.equals(second.stores)


def test_late_opening_stores_exist(tiny_config):
    universe = generate_universe(tiny_config.universe, np.random.default_rng(2), 30)
    assert (universe.stores["open_week"] >= 1).all()
    assert (universe.stores["open_week"] <= 30).all()
    assert (universe.stores["open_week"] > 1).any()
