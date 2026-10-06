"""Deterministic what-if simulations of feasible actions, driven by the AUTOBUS metadata.

A simulation:
  1. loads one row per unit entity (e.g. subscription) with every attribute used by the
     prediction models, KPIs, actions and segment filters, joining related entities,
  2. scores the registered models of that unit to obtain the baseline predictions,
  3. for each scenario (baseline, every action level, and combinations across different
     actions) applies the action levels to the segment rows only, rescores the models
     whose inputs changed, and evaluates the KPIs,
  4. checks each scenario against the goal: an objective KPI target and the revenue
     constraint (total revenue KPI must not decline by more than a given amount).
"""

import itertools
import json
from functools import lru_cache
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from agent.autobus_db import get_attributes, get_feasible_actions, get_kpis, load_analysis_frame
from agent.kpi import evaluate_kpi, referenced_attributes
from agent.model_runtime import PredictionModel, get_models

MAX_SCENARIOS = 500

# KPI units that are fractions, whose targets may be expressed in percentage points.
FRACTION_UNITS = {"probability", "rate", "share", "fraction"}


class SegmentFilter(BaseModel):
    attribute: str = Field(description="entity.attribute name from the simulation catalog")
    op: Literal["top_share", "bottom_share", "gt", "ge", "lt", "le", "eq", "ne", "between", "in"] = Field(
        description="top_share/bottom_share: the given fraction of rows with the highest/lowest values "
                    "(value 0.10 = top 10%); gt/ge/lt/le/eq/ne compare with value; "
                    "between uses values [low, high] inclusive; in uses values"
    )
    value: Optional[Union[float, str]] = Field(description="Value for top_share, bottom_share and comparisons")
    values: Optional[List[Union[float, str]]] = Field(description="Values for between and in")


class GoalSpec(BaseModel):
    objective_kpi: str = Field(description="kpi_name of the KPI the goal wants to change")
    objective_scope: Literal["segment", "population"] = Field(
        description="Evaluate the objective KPI over the segment or over all rows of the unit entity"
    )
    direction: Literal["decrease", "increase"]
    target_change: float = Field(description="Required size of the change, positive, in target_mode units")
    target_mode: Literal["absolute", "percentage_points", "relative_percent"] = Field(
        description="absolute: in KPI units; percentage_points: for fraction KPIs, 1.0 = 0.01; "
                    "relative_percent: percent of the baseline value"
    )
    max_revenue_decline: Optional[float] = Field(
        description="Largest allowed decline of the total revenue KPI over the population; null for no constraint"
    )
    revenue_kpi: Optional[str] = Field(
        description="kpi_name of the revenue KPI; null to use the KPI with category 'revenue'"
    )


def resolve_attribute(name: str) -> str:
    """Accept 'entity.attribute' or an unambiguous attribute name (case-insensitive)."""
    attributes = [f"{a['entity_name']}.{a['attribute_name']}" for a in get_attributes()]
    lowered = name.strip().lower()
    matches = [a for a in attributes if a.lower() == lowered] or [
        a for a in attributes if a.split(".", 1)[1].lower() == lowered.rsplit(".", 1)[-1]
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Attribute {name!r} is {'ambiguous' if matches else 'unknown'}; "
            f"use one of {matches or attributes}"
        )
    return matches[0]


def _kpi_by_name(kpis: Dict[str, Dict[str, Any]], name: str) -> Dict[str, Any]:
    if name not in kpis:
        raise ValueError(f"Unknown KPI {name!r}; choose from {sorted(kpis)}")
    kpi = kpis[name]
    used = referenced_attributes(kpi["expression"])
    if kpi["attributes"] and used != set(kpi["attributes"]):
        raise ValueError(
            f"KPI {name!r} expression uses {sorted(used)} but kpi_attribute lists {kpi['attributes']}"
        )
    return kpi


def _revenue_kpi(kpis: Dict[str, Dict[str, Any]], name: Optional[str]) -> Dict[str, Any]:
    if name:
        return _kpi_by_name(kpis, name)
    revenue = [k for k in kpis.values() if k["category"] == "revenue"]
    if len(revenue) != 1:
        raise ValueError(
            f"Expected one KPI with category 'revenue', found {[k['kpi_name'] for k in revenue]}; "
            "pass revenue_kpi explicitly"
        )
    return _kpi_by_name(kpis, revenue[0]["kpi_name"])


def _segment_mask(frame: pd.DataFrame, filters: List[SegmentFilter]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for f in filters:
        column = frame[f.attribute]
        if f.op in ("top_share", "bottom_share"):
            share = float(f.value) if f.value is not None else None
            if share is None or not 0 < share < 1:
                raise ValueError(f"{f.op} needs a value between 0 and 1, e.g. 0.10 for 10%; got {f.value}")
            if f.op == "top_share":
                mask &= column.gt(column.quantile(1 - share))
            else:
                mask &= column.lt(column.quantile(share))
        elif f.op == "between":
            if not f.values or len(f.values) != 2:
                raise ValueError("between needs values [low, high]")
            mask &= column.between(*f.values)
        elif f.op == "in":
            if not f.values:
                raise ValueError("in needs a non-empty values list")
            mask &= column.isin(f.values)
        else:
            if f.value is None:
                raise ValueError(f"{f.op} needs a value")
            mask &= getattr(column, f.op)(f.value)
    return mask


def _describe_filter(f: SegmentFilter, frame: pd.DataFrame) -> str:
    column = frame[f.attribute]
    if f.op == "top_share":
        return f"top {float(f.value):.0%} by {f.attribute} ({f.attribute} > {column.quantile(1 - float(f.value)):g})"
    if f.op == "bottom_share":
        return f"bottom {float(f.value):.0%} by {f.attribute} ({f.attribute} < {column.quantile(float(f.value)):g})"
    if f.op in ("between", "in"):
        return f"{f.attribute} {f.op} {f.values}"
    symbols = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "=", "ne": "!="}
    return f"{f.attribute} {symbols[f.op]} {f.value}"


def _apply_action(values: pd.Series, action_type: str, level_value: float) -> pd.Series:
    if action_type == "scale":
        return values * level_value
    if action_type == "add":
        return values + level_value
    if action_type == "set":
        return pd.Series(level_value, index=values.index, dtype=float)
    raise ValueError(f"Unsupported action_type {action_type!r}")


@lru_cache(maxsize=8)
def _baseline_frame(unit_entity: str, attributes: tuple, model_ids: tuple) -> pd.DataFrame:
    """Unit rows with the requested attributes and baseline model predictions (cached per process)."""
    models = get_models()
    frame = load_analysis_frame(unit_entity, attributes)
    for model_id in model_ids:
        model = models[model_id]
        frame[model.output_attribute] = model.predict(frame)
    return frame


def _target_units(improvement: float, baseline: float, kpi: Dict[str, Any], mode: str) -> float:
    if mode == "absolute":
        return improvement
    if mode == "percentage_points":
        if kpi["unit"] not in FRACTION_UNITS:
            raise ValueError(
                f"percentage_points targets need a fraction KPI (unit in {sorted(FRACTION_UNITS)}); "
                f"{kpi['kpi_name']!r} has unit {kpi['unit']!r}"
            )
        return improvement * 100
    if baseline == 0:
        raise ValueError(f"relative_percent target is undefined: baseline of {kpi['kpi_name']!r} is 0")
    return improvement / abs(baseline) * 100


def simulate(
    segment_filters: List[SegmentFilter],
    goal: GoalSpec,
    action_ids: Optional[List[int]] = None,
    include_combinations: bool = True,
) -> Dict[str, Any]:
    """Simulate the feasible actions on the segment and evaluate them against the goal."""
    key = json.dumps({
        "filters": [f.model_dump() for f in segment_filters],
        "goal": goal.model_dump(),
        "actions": sorted(action_ids) if action_ids else None,
        "combinations": include_combinations,
    }, sort_keys=True)
    return _simulate(key)


@lru_cache(maxsize=32)
def _simulate(key: str) -> Dict[str, Any]:
    spec = json.loads(key)
    goal = GoalSpec(**spec["goal"])
    filters = [
        SegmentFilter(**{**f, "attribute": resolve_attribute(f["attribute"])}) for f in spec["filters"]
    ]

    kpis = {k["kpi_name"]: k for k in get_kpis()}
    objective_kpi = _kpi_by_name(kpis, goal.objective_kpi)
    revenue_kpi = _revenue_kpi(kpis, goal.revenue_kpi) if goal.max_revenue_decline is not None else None
    unit = objective_kpi["unit_entity_name"]
    if revenue_kpi and revenue_kpi["unit_entity_name"] != unit:
        raise ValueError(
            f"Objective KPI is over {unit!r} but revenue KPI {revenue_kpi['kpi_name']!r} is over "
            f"{revenue_kpi['unit_entity_name']!r}"
        )
    unit_kpis = {n: k for n, k in kpis.items() if k["unit_entity_name"] == unit}

    # Models that score the unit entity; their outputs are computed, not read from the database.
    models: Dict[int, PredictionModel] = {
        m_id: m for m_id, m in get_models().items() if m.unit_entity == unit
    }
    outputs = {m.output_attribute for m in models.values()}

    action_ids = spec["actions"]
    all_actions = {a["action_id"]: a for a in get_feasible_actions()}
    if action_ids:
        unknown = sorted(set(action_ids) - set(all_actions))
        if unknown:
            raise ValueError(f"Unknown action_ids {unknown}; choose from {sorted(all_actions)}")
    actions = [all_actions[a] for a in (action_ids or sorted(all_actions))]
    for action in actions:
        action["attribute"] = f"{action['entity_name']}.{action['attribute_name']}"
        if not action["levels"]:
            raise ValueError(f"Action {action['action_id']} has no levels in action_level")

    needed = set()
    for model in models.values():
        needed |= set(model.input_attributes)
    for kpi in unit_kpis.values():
        needed |= referenced_attributes(kpi["expression"])
    needed |= {f.attribute for f in filters} | {a["attribute"] for a in actions}
    needed -= outputs

    frame = _baseline_frame(unit, tuple(sorted(needed)), tuple(sorted(models)))
    mask = _segment_mask(frame, filters).to_numpy()
    if not mask.any():
        raise ValueError("The segment is empty; relax the segment filters")
    objective_rows = mask if goal.objective_scope == "segment" else np.ones(len(frame), dtype=bool)

    # Scenarios: every combination of at most one level per action (None = action not applied).
    level_choices = [[None] + [(a, level) for level in a["levels"]] for a in actions]
    if spec["combinations"]:
        combos = list(itertools.product(*level_choices))
    else:
        combos = [tuple(None for _ in actions)] + [
            tuple(choice if j == i else None for j in range(len(actions)))
            for i, choices in enumerate(level_choices) for choice in choices[1:]
        ]
    if len(combos) > MAX_SCENARIOS:
        raise ValueError(
            f"{len(combos)} scenarios exceed the limit of {MAX_SCENARIOS}; "
            "pass fewer action_ids or set include_combinations to false"
        )

    def evaluate(scenario_frame: pd.DataFrame) -> Dict[str, float]:
        values = {
            name: evaluate_kpi(kpi, scenario_frame.loc[objective_rows])
            for name, kpi in unit_kpis.items()
        }
        if revenue_kpi:
            values["__revenue__"] = evaluate_kpi(revenue_kpi, scenario_frame)
        return values

    scenarios = []
    for scenario_id, combo in enumerate(combos):
        applied = [c for c in combo if c is not None]
        scenario_frame = frame
        if applied:
            scenario_frame = frame.copy()
            changed = set()
            for action, level in applied:
                column = action["attribute"]
                scenario_frame[column] = scenario_frame[column].astype(float)
                scenario_frame.loc[mask, column] = _apply_action(
                    scenario_frame.loc[mask, column], action["action_type"], level["value"]
                )
                changed.add(column)
            segment_rows = scenario_frame.loc[mask]
            for model in models.values():
                if changed & set(model.input_attributes):
                    scenario_frame.loc[mask, model.output_attribute] = model.predict(segment_rows)
        scenarios.append({
            "scenario_id": scenario_id,
            "label": " + ".join(level["label"] for _, level in applied) or "baseline (no action)",
            "actions": [
                {"action_id": a["action_id"], "level_id": level["level_id"], "value": level["value"]}
                for a, level in applied
            ],
            "kpis": evaluate(scenario_frame),
        })

    baseline = scenarios[0]["kpis"]
    sign = -1 if goal.direction == "decrease" else 1
    for scenario in scenarios:
        kpi_values = scenario.pop("kpis")
        objective_value = kpi_values[goal.objective_kpi]
        improvement = 0.0 + _target_units(
            sign * (objective_value - baseline[goal.objective_kpi]),
            baseline[goal.objective_kpi], objective_kpi, goal.target_mode,
        )
        scenario.update({
            "objective_value": objective_value,
            "objective_change": objective_value - baseline[goal.objective_kpi],
            "objective_improvement": improvement,
            "meets_objective_target": improvement >= goal.target_change,
            "objective_shortfall": max(0.0, goal.target_change - improvement),
        })
        if revenue_kpi:
            revenue_change = kpi_values["__revenue__"] - baseline["__revenue__"]
            scenario.update({
                "revenue_value": kpi_values["__revenue__"],
                "revenue_change": revenue_change,
                "meets_revenue_constraint": -revenue_change <= goal.max_revenue_decline,
                "revenue_overrun": max(0.0, -revenue_change - goal.max_revenue_decline),
            })
        else:
            scenario.update({"revenue_change": 0.0, "meets_revenue_constraint": True})
        scenario["other_kpis"] = {
            name: value for name, value in kpi_values.items()
            if name not in (goal.objective_kpi, "__revenue__")
        }

    feasible = [s for s in scenarios if s["meets_objective_target"] and s["meets_revenue_constraint"]]
    within_budget = [s for s in scenarios if s["meets_revenue_constraint"]]
    meeting_target = [s for s in scenarios if s["meets_objective_target"]]

    def pick(candidates, key_fn):
        best = max(candidates, key=key_fn, default=None)
        return {"scenario_id": best["scenario_id"], "label": best["label"]} if best else None

    def cheapest(s):
        return (s["revenue_change"], s["objective_improvement"])

    def most_improved(s):
        return (s["objective_improvement"], s["revenue_change"])

    return {
        "unit_entity": unit,
        "segment": {
            "filters": [_describe_filter(f, frame) for f in filters],
            "size": int(mask.sum()),
            "share_of_population": float(mask.mean()),
            "population_size": len(frame),
        },
        "goal": goal.model_dump(),
        "objective_kpi": {k: objective_kpi[k] for k in ("kpi_name", "description", "aggregation", "expression", "unit")}
                         | {"scope": goal.objective_scope},
        "revenue_kpi": (
            {k: revenue_kpi[k] for k in ("kpi_name", "description", "aggregation", "expression", "unit")}
            | {"scope": "population"} if revenue_kpi else None
        ),
        "models": [
            {"model_id": m.model_id, "short_desc": m.meta["short_desc"], "prediction": m.output_attribute}
            for m in models.values()
        ],
        "actions": [
            {k: a[k] for k in ("action_id", "description", "attribute", "action_type", "levels")} for a in actions
        ],
        "scenarios": scenarios,
        "evaluation": {
            "verdict": "feasible" if feasible else "infeasible",
            "feasible_scenario_ids": [s["scenario_id"] for s in feasible],
            # Feasible scenario with the smallest revenue loss.
            "recommended": pick(feasible, cheapest),
            # Largest objective improvement within the revenue constraint.
            "best_within_budget": pick(within_budget, most_improved),
            # Smallest revenue loss among scenarios reaching the objective target.
            "cheapest_meeting_target": pick(meeting_target, cheapest),
            "largest_objective_improvement": pick(scenarios, most_improved),
        },
    }
