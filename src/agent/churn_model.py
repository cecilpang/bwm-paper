import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from agent.autobus_db import PROJECT_DIR

MODEL_META_PATH = PROJECT_DIR / "saved-models" / "churn_model_lr_meta.json"

# Database attribute (see predictive_feature) -> feature name the saved pipeline expects.
DB_TO_MODEL_FEATURE = {
    "frequency": "frequency",
    "breadth": "breadth",
    "tenure_days": "tenure",
    "monthly_cost": "monthly_cost",
}


class ChurnModel:
    """Saved churn pipeline with prior-corrected probabilities (see notebooks/Inference.ipynb)."""

    def __init__(self, model_meta_path: Path = MODEL_META_PATH):
        self.meta: Dict[str, Any] = json.loads(Path(model_meta_path).read_text(encoding="utf-8"))
        self.pipe = joblib.load(Path(model_meta_path).parent / self.meta["model_file"])
        self.features = self.meta["features"]
        self.positive_class_index = list(self.pipe.classes_).index(self.meta["positive_class"])
        self.beta = self.meta["probability_correction"]["beta"]
        self.decision_threshold = self.meta["probability_correction"]["corrected_decision_threshold"]

        missing = set(self.features) - set(DB_TO_MODEL_FEATURE.values())
        if missing:
            raise ValueError(f"No database attribute mapped to model features: {sorted(missing)}")

    def to_model_features(self, db_frame: pd.DataFrame) -> pd.DataFrame:
        """Rename database columns to the model's feature names, in the model's order."""
        return db_frame.rename(columns=DB_TO_MODEL_FEATURE)[self.features]

    def predict_churn_probability(self, feature_frame: pd.DataFrame) -> np.ndarray:
        """Prior-corrected churn probability for frames with the model's feature names."""
        raw = self.pipe.predict_proba(feature_frame[self.features])[:, self.positive_class_index]
        return self.beta * raw / (self.beta * raw + 1 - raw)


@lru_cache(maxsize=1)
def get_churn_model() -> ChurnModel:
    return ChurnModel()
