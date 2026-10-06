"""Deterministic what-if simulations of feasible actions with the churn model.

Churn is measured as the segment's mean prior-corrected churn probability.
Revenue is the expected monthly revenue of all subscriptions,
sum(monthly_cost * (1 - churn probability)); subscriptions outside the
segment keep their baseline contribution.
"""

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from agent.autobus_db import get_feasible_actions, load_subscription_data
from agent.churn_model import get_churn_model

DISCOUNT_ENTITY = "subscription"
DISCOUNT_ATTRIBUTE = "monthly_cost"


def allowed_discount_rates() -> List[float]:
    """Discount rates allowed by the feasible action on subscription.monthly_cost."""
    for action in get_feasible_actions():
        if (action["entity_name"], action["attribute_name"]) == (DISCOUNT_ENTITY, DISCOUNT_ATTRIBUTE):
            rates = [int(p) / 100 for p in re.findall(r"(\d+)\s*%", action["description"] or "")]
            if rates:
                return sorted(set(rates))
    raise LookupError(f"No feasible action with discount levels on {DISCOUNT_ENTITY}.{DISCOUNT_ATTRIBUTE}")


@lru_cache(maxsize=1)
def _scored_subscriptions() -> pd.DataFrame:
    """Subscriptions from the database scored with the churn model (cached per process)."""
    model = get_churn_model()
    df = load_subscription_data()
    df["churn_probability"] = model.predict_churn_probability(model.to_model_features(df))
    df["expected_revenue"] = df["monthly_cost"] * (1 - df["churn_probability"])
    return df


SEGMENT_ATTRIBUTES = ("frequency", "breadth", "tenure_days", "monthly_cost")


def normalize_segment_attribute(attribute: str) -> str:
    """Accept 'frequency', 'Frequency' or 'content_consumption_statistics.frequency'."""
    name = attribute.strip().lower().rsplit(".", 1)[-1]
    if name not in SEGMENT_ATTRIBUTES:
        raise ValueError(f"Unknown segment attribute {attribute!r}; choose from {list(SEGMENT_ATTRIBUTES)}")
    return name


def define_segment(df: pd.DataFrame, attribute: str, top_share: float) -> Dict[str, Any]:
    """Subscriptions with the highest attribute values: strictly above the (1 - top_share) quantile."""
    if not 0 < top_share < 1:
        raise ValueError(
            f"top_share must be a fraction between 0 and 1, e.g. 0.10 for the top 10%; got {top_share}"
        )
    cutoff = float(df[attribute].quantile(1 - top_share))
    mask = df[attribute].gt(cutoff)
    return {"mask": mask, "cutoff": cutoff, "size": int(mask.sum()), "share": float(mask.mean())}


@lru_cache(maxsize=32)
def _simulate(segment_attribute: str, top_share: float, discount_rates: tuple) -> Dict[str, Any]:
    model = get_churn_model()
    df = _scored_subscriptions()
    segment = define_segment(df, segment_attribute, top_share)
    mask = segment["mask"]

    total_baseline_revenue = float(df["expected_revenue"].sum())
    outside_revenue = float(df.loc[~mask, "expected_revenue"].sum())
    segment_features = model.to_model_features(df.loc[mask])
    baseline_mean = float(df.loc[mask, "churn_probability"].mean())

    scenarios = []
    for rate in (0.0, *discount_rates):
        if rate == 0:
            cost = df.loc[mask, "monthly_cost"].to_numpy()
            probability = df.loc[mask, "churn_probability"].to_numpy()
        else:
            features = segment_features.copy()
            features["monthly_cost"] *= 1 - rate
            cost = features["monthly_cost"].to_numpy()
            probability = model.predict_churn_probability(features)

        mean_probability = float(probability.mean())
        expected_revenue = outside_revenue + float(np.sum(cost * (1 - probability)))
        scenarios.append({
            "discount_rate": rate,
            "segment_mean_churn_probability": mean_probability,
            "churn_change_pp": (mean_probability - baseline_mean) * 100,
            "churn_change_relative_pct": (mean_probability / baseline_mean - 1) * 100,
            "segment_thresholded_churn_rate": float(np.mean(probability >= model.decision_threshold)),
            "expected_monthly_revenue": expected_revenue,
            "revenue_change": expected_revenue - total_baseline_revenue,
        })

    return {
        "action": f"Uniform discount on {DISCOUNT_ENTITY}.{DISCOUNT_ATTRIBUTE} for every subscription in the segment",
        "segment": {
            "definition": f"top {top_share:.0%} by {segment_attribute}: {segment_attribute} > {segment['cutoff']:g}",
            "size": segment["size"],
            "share_of_subscriptions": segment["share"],
        },
        "total_subscriptions": len(df),
        "baseline_expected_monthly_revenue": total_baseline_revenue,
        "churn_metric": "segment mean prior-corrected churn probability",
        "revenue_metric": "expected monthly revenue of all subscriptions, sum(monthly_cost * (1 - p))",
        "scenarios": scenarios,
    }


def simulate_uniform_discount(
    segment_attribute: str = "frequency",
    top_share: float = 0.10,
    discount_rates: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Rescore the segment for each discount rate; the first scenario is the 0% baseline."""
    allowed = allowed_discount_rates()
    rates = allowed if discount_rates is None else sorted({round(float(r), 4) for r in discount_rates})
    invalid = [r for r in rates if r not in allowed]
    if invalid:
        raise ValueError(f"Discount rates {invalid} are outside the action space {allowed}")
    return _simulate(normalize_segment_attribute(segment_attribute), round(float(top_share), 6), tuple(rates))


def evaluate_goal(
    simulation: Dict[str, Any],
    target_churn_change_pp: float,
    max_revenue_decline: float,
) -> Dict[str, Any]:
    """Check each scenario against an absolute churn reduction (pp) and a revenue-decline cap ($)."""
    evaluated = []
    for scenario in simulation["scenarios"]:
        churn_reduction_pp = -scenario["churn_change_pp"]
        revenue_decline = -scenario["revenue_change"]
        evaluated.append({
            **scenario,
            "meets_churn_target": churn_reduction_pp >= target_churn_change_pp,
            "meets_revenue_constraint": revenue_decline <= max_revenue_decline,
            "churn_shortfall_pp": max(0.0, target_churn_change_pp - churn_reduction_pp),
            "revenue_overrun": max(0.0, revenue_decline - max_revenue_decline),
        })

    feasible = [s for s in evaluated if s["meets_churn_target"] and s["meets_revenue_constraint"]]
    within_budget = [s for s in evaluated if s["meets_revenue_constraint"]]
    meeting_target = [s for s in evaluated if s["meets_churn_target"]]

    return {
        "goal": {
            "target_churn_reduction_pp": target_churn_change_pp,
            "max_revenue_decline": max_revenue_decline,
        },
        "verdict": "feasible" if feasible else "infeasible",
        "feasible_scenarios": feasible,
        # Most churn reduction while staying within the revenue cap.
        "best_within_budget": min(within_budget, key=lambda s: s["churn_change_pp"], default=None),
        # Smallest revenue loss among scenarios that reach the churn target.
        "cheapest_meeting_target": max(meeting_target, key=lambda s: s["revenue_change"], default=None),
        "largest_churn_reduction": min(evaluated, key=lambda s: s["churn_change_pp"]),
        "scenarios": evaluated,
    }
