"""Synthetic change notices, candidate changes and the oracle's answer.

A *candidate* is one group of lifecycle changes the engine observed in a
refresh (one classification, its entities and weeks), or a merge relationship.
A *notice* is a short operator message. ``true_notice`` describes the fault the
generator injected; the distractors are built to look relevant and explain
nothing. ``gold`` is the set of candidate keys a notice explains (empty for
"none of these").

Only the oracle and the dimension tables are read here, never the engine's
answer: candidates come from the engine's events, the truth from the oracle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from qc.notices import ACCEPTS, Candidate
from qc.notices import candidates as engine_candidates

CHANGE_TYPES = tuple(ACCEPTS)
DISTRACTOR_KINDS = ("wrong_entity", "wrong_change", "wrong_weeks", "other_dataset", "chatter")
OTHER_DATASETS = ("Loyalty panel", "Script volumes (PBS)", "Online orders", "Wholesale shipments")
@dataclass(frozen=True)
class World:
    """What a notice writer would know about the dataset."""

    dataset: str
    latest_week: int
    stores: dict[str, dict[str, Any]]
    products: dict[str, dict[str, Any]]
    commodities: dict[str, str]


@dataclass
class Notice:
    id: str
    text: str
    kind: str  # "true" or a DISTRACTOR_KINDS member
    change_type: str | None
    gold: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Candidates (engine side: qc.notices)
# ---------------------------------------------------------------------------


def candidates(result: Any, world: World) -> list[Candidate]:
    """The engine's observed changes, labelled with store and category names."""
    names = {sid: str(row["store_name"]) for sid, row in world.stores.items()} | dict(world.commodities)
    return engine_candidates(result, names)


def observed_ids(cands: list[Candidate]) -> set[str]:
    return {part for c in cands for e in c.entity_ids for part in e.split("->")}


def gold_for(change_type: str, affected: set[str], cands: list[Candidate]) -> list[str]:
    """Candidates a true notice explains: an accepted classification sharing an entity."""
    return [
        c.key
        for c in cands
        if c.classification in ACCEPTS[change_type]
        and affected & {part for e in c.entity_ids for part in e.split("->")}
    ]


# ---------------------------------------------------------------------------
# Notices (operator side)
# ---------------------------------------------------------------------------


class Writer:
    """Seeded templates with varied phrasing and ID formats."""

    def __init__(self, world: World, rng: np.random.Generator):
        self.world, self.rng = world, rng

    def pick(self, options: list[str] | tuple[str, ...]) -> str:
        return str(options[int(self.rng.integers(len(options)))])

    def store(self, store_id: str) -> str:
        number = int(store_id[1:])
        name = self.world.stores.get(store_id, {}).get("store_name", f"Store {number:03d}")
        return self.pick([store_id, f"store {number}", f"store {store_id}", name, f"{name} ({store_id})"])

    def product(self, product_id: str) -> str:
        number = int(product_id[1:])
        return self.pick([product_id, f"item {product_id}", f"product {number}", f"SKU {product_id}"])

    def commodity(self, commodity_id: str) -> str:
        name = self.world.commodities.get(commodity_id, commodity_id)
        return self.pick([name, name, f"{name} ({commodity_id})", commodity_id])

    def listing(self, items: list[str], render: Any) -> str:
        shown = [render(i) for i in items[:3]]
        text = shown[0] if len(shown) == 1 else ", ".join(shown[:-1]) + " and " + shown[-1]
        if len(items) > 3:
            text += f" (plus {len(items) - 3} more)"
        return text

    def week(self, week: int) -> str:
        return self.pick([f"week {week}", f"wk {week}", f"wk{week}", f"week {week}"])

    def span(self, start: int, end: int) -> str:
        return self.pick([f"weeks {start}-{end}", f"wk {start} to {end}", f"weeks {start} through {end}"])

    def render(self, change_type: str, spec: dict[str, Any]) -> str:
        """One notice for a change, from its spec (entities and weeks)."""
        w = self.world.latest_week
        if change_type == "missing_stores":
            stores = self.listing(spec["stores"], self.store)
            start = spec.get("from_week", w)
            return self.pick([
                f"{stores} closed for refit from {self.week(start)}, reopening in about {int(self.rng.integers(3, 9))} weeks.",
                f"Store closures: {stores} did not trade in {self.week(start)} (flooding in the area).",
                f"Heads up - no sales feed from {stores} since {self.week(start)}; POS migration in progress.",
                f"Temporary closure of {stores} starting {self.week(start)} for renovations.",
            ])
        if change_type == "missing_products":
            products = self.listing(spec["products"], self.product)
            start = spec.get("from_week", w)
            return self.pick([
                f"Discontinued from {self.week(start)}: {products}.",
                f"Supply issue - {products} out of stock nationally since {self.week(start)}.",
                f"Range review delisted {products}, effective {self.week(start)}.",
            ])
        if change_type == "new_store_backfill":
            store = self.store(spec["stores"][0])
            start, end = spec["start"], spec["end"]
            return self.pick([
                f"A new store, {store}, opened; the vendor loaded its history back to {self.week(start)}.",
                f"{store} added to the feed this refresh with {end - start + 1} weeks of back history ({self.span(start, end)}).",
                f"Acquisition: {store} joins the panel, history supplied for {self.span(start, end)}.",
            ])
        if change_type == "history_truncation":
            stores = self.listing(spec["stores"], self.store)
            start, end = spec["start"], spec["end"]
            return self.pick([
                f"Vendor withdrew {self.span(start, end)} for {stores} pending a data audit.",
                f"{stores}: sales for {self.span(start, end)} deleted at source (duplicate load).",
                f"Removing history for {stores} in {self.span(start, end)} - will be resupplied later.",
            ])
        if change_type == "commodity_remap":
            source, target = self.commodity(spec["from"]), self.commodity(spec["to"])
            start = spec.get("from_week", w)
            return self.pick([
                f"Range review: {len(spec['products'])} lines moving from {source} to {target} from {self.week(start)}.",
                f"Category change - products reclassified {source} -> {target} in the product hierarchy.",
                f"Hierarchy update: some {source} items now sit under {target}.",
            ])
        if change_type == "entity_merge":
            source, target = self.store(spec["source"]), self.store(spec["target"])
            return self.pick([
                f"{source} merged into the new store {target}; its history was carried across.",
                f"Rebrand: {source} now trades as {target}, history moved to the new id.",
                f"Consolidation - {source} replaced by {target} in the store master.",
            ])
        raise ValueError(change_type)

    def render_off_weeks(self, change_type: str, spec: dict[str, Any]) -> str | None:
        """The change at weeks that cannot explain this refresh (always stated), or None."""
        w = self.world.latest_week
        if change_type == "missing_stores":
            stores = self.listing(spec["stores"], self.store)
            return self.pick([
                f"{stores} will close for refit from {self.week(w + 6)}.",
                f"Planned closure: {stores}, starting {self.week(w + 6)}.",
            ])
        if change_type == "missing_products":
            products = self.listing(spec["products"], self.product)
            return self.pick([
                f"{products} will be discontinued from {self.week(w + 6)}.",
                f"Range review: {products} to be delisted effective {self.week(w + 6)}.",
            ])
        if change_type in ("new_store_backfill", "history_truncation"):
            length = spec["end"] - spec["start"]
            start = max(1, spec["start"] - 60)
            end = start + length
            if change_type == "new_store_backfill":
                store = self.store(spec["stores"][0])
                return f"{store}: history for {self.span(start, end)} only has been loaded."
            stores = self.listing(spec["stores"], self.store)
            return f"Vendor withdrew {self.span(start, end)} for {stores} pending a data audit."
        if change_type == "commodity_remap":
            source, target = self.commodity(spec["from"]), self.commodity(spec["to"])
            return f"From {self.week(w + 8)}, {source} lines will move to {target}."
        return None  # a merge has no weeks to get wrong

    def chatter(self) -> str:
        return self.pick([
            "Reminder: quarterly re-weighting review is on Thursday.",
            "Month-end close moves to the 3rd this month.",
            "The data portal will be down for maintenance on Saturday night.",
            "New analyst starting Monday - please add them to the QC channel.",
            "Price file for next quarter has been published.",
        ])


def spec_from_case(case: dict[str, Any], world: World) -> tuple[str, dict[str, Any], set[str]] | None:
    """The injected change as (change type, notice spec, affected ids), or None."""
    family, affected, weeks, details = case["family"], case.get("affected", {}), case.get("weeks", []), case.get("details", {})
    if family == "missing_stores":
        return family, {"stores": affected["stores"], "from_week": int(details.get("week", world.latest_week))}, set(affected["stores"])
    if family == "missing_products":
        return family, {"products": affected["products"], "from_week": int(details.get("week", world.latest_week))}, set(affected["products"])
    if family == "new_store_backfill":
        return family, {"stores": affected["stores"], "start": int(details["backfill_start"]), "end": int(details["backfill_end"])}, set(affected["stores"])
    if family == "history_truncation":
        return family, {"stores": affected["stores"], "start": int(min(weeks)), "end": int(max(weeks))}, set(affected["stores"])
    if family == "commodity_remap":
        return family, {"from": details["from_commodity"], "to": details["to_commodity"], "products": affected["products"]}, {details["from_commodity"], details["to_commodity"]}
    if family == "entity_merge":
        return family, {"source": details["source_store"], "target": details["target_store"]}, {details["source_store"], details["target_store"]}
    return None


def _unused(ids: list[str], exclude: set[str], rng: np.random.Generator, n: int) -> list[str]:
    pool = [i for i in ids if i not in exclude]
    n = min(n, len(pool))
    return [pool[int(i)] for i in rng.choice(len(pool), size=n, replace=False)] if n else []


def notices_for(
    case: dict[str, Any], world: World, cands: list[Candidate], rng: np.random.Generator
) -> list[Notice]:
    """One true notice (if the fault has a template) plus one notice of each distractor kind."""
    writer = Writer(world, rng)
    observed = observed_ids(cands)
    stores, products = sorted(world.stores), sorted(world.products)
    injected = spec_from_case(case, world)
    out: list[Notice] = []
    if injected is not None:
        change_type, spec, affected = injected
        out.append(Notice("n-true", writer.render(change_type, spec), "true", change_type, gold_for(change_type, affected, cands)))
        exclude = observed | affected
    else:
        change_type = CHANGE_TYPES[int(rng.integers(len(CHANGE_TYPES)))]
        spec, affected, exclude = {}, set(), set(observed)
    w = world.latest_week

    def fresh_spec(kind: str) -> dict[str, Any]:
        """A spec of change type ``kind`` about entities the engine did not observe."""
        some = _unused(stores, exclude, rng, 2)
        if kind == "missing_stores":
            return {"stores": some, "from_week": w}
        if kind == "missing_products":
            return {"products": _unused(products, exclude, rng, 2), "from_week": w}
        if kind == "new_store_backfill":
            return {"stores": some[:1], "start": w - 11, "end": w}
        if kind == "history_truncation":
            return {"stores": some, "start": w - 6, "end": w - 1}
        if kind == "commodity_remap":
            pair = _unused(sorted(world.commodities), exclude, rng, 2)
            return {"from": pair[0], "to": pair[1], "products": ["x", "y"]}
        return {"source": some[0], "target": f"S{int(rng.integers(950, 999))}"}

    # Right change type, entities this refresh does not touch.
    out.append(Notice("n-wrong-entity", writer.render(change_type, fresh_spec(change_type)), "wrong_entity", change_type))
    # Right entities (when there are any), a different kind of change.
    other = [t for t in CHANGE_TYPES if t != change_type and _compatible(t, change_type)]
    alt = other[int(rng.integers(len(other)))] if other else CHANGE_TYPES[0]
    alt_spec = fresh_spec(alt)
    if injected is not None:
        alt_spec = _retarget(alt, alt_spec, spec, change_type)
    out.append(Notice("n-wrong-change", writer.render(alt, alt_spec), "wrong_change", alt))
    # Right entities and change, but weeks that cannot explain this refresh.
    if injected is not None:
        off = writer.render_off_weeks(change_type, spec)
        if off is not None:
            out.append(Notice("n-wrong-weeks", off, "wrong_weeks", change_type))
        tagged = writer.render(change_type, spec)
    else:
        tagged = writer.render(change_type, fresh_spec(change_type))
    # Another dataset's notice about the same kind of change.
    other_dataset = OTHER_DATASETS[int(rng.integers(len(OTHER_DATASETS)))]
    out.append(Notice("n-other-dataset", f"[{other_dataset}] {tagged}", "other_dataset", change_type))
    out.append(Notice("n-chatter", writer.chatter(), "chatter", None))
    return out


def _compatible(a: str, b: str) -> bool:
    """Change types about the same entity type (stores vs products/categories)."""
    store_types = {"missing_stores", "new_store_backfill", "history_truncation", "entity_merge"}
    return (a in store_types) == (b in store_types)


def _retarget(alt: str, alt_spec: dict[str, Any], spec: dict[str, Any], change_type: str) -> dict[str, Any]:
    """Point a different change type at the injected change's own entities."""
    stores = spec.get("stores") or [s for s in (spec.get("source"), spec.get("target")) if s]
    if alt in ("missing_stores", "history_truncation") and stores:
        return {**alt_spec, "stores": stores}
    if alt == "new_store_backfill" and stores:
        return {**alt_spec, "stores": stores[:1]}
    if alt == "entity_merge" and stores:
        return {**alt_spec, "source": stores[0]}
    if alt == "missing_products" and spec.get("products"):
        return {**alt_spec, "products": spec["products"]}
    if alt == "commodity_remap" and spec.get("products") and change_type == "missing_products":
        return alt_spec
    return alt_spec


def world_from_dims(dims: dict[str, Any], dataset: str, latest_week: int) -> World:
    return World(
        dataset=dataset,
        latest_week=int(latest_week),
        stores={str(r["store_id"]): r for r in dims["stores"].to_dict("records")},
        products={str(r["product_id"]): r for r in dims["products"].to_dict("records")},
        commodities={str(r["commodity_id"]): str(r["commodity_name"]) for r in dims["commodities"].to_dict("records")},
    )
