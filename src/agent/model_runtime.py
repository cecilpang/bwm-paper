"""Load and run any registered prediction model from its AUTOBUS metadata."""

from functools import lru_cache
from typing import Any, Callable, Dict, List

import joblib
import numpy as np
import pandas as pd

from agent.autobus_db import PROJECT_DIR, get_prediction_models


def _prior_correction(probability: np.ndarray, beta: float) -> np.ndarray:
    """Undo undersampling: map probabilities back to the original class prior."""
    return beta * probability / (beta * probability + 1 - probability)


# output_transform "type" -> function(raw output, **parameters)
OUTPUT_TRANSFORMS: Dict[str, Callable[..., np.ndarray]] = {
    "prior_correction": _prior_correction,
}

# artifact_format -> loader(path)
ARTIFACT_LOADERS: Dict[str, Callable[[Any], Any]] = {
    "sklearn_joblib": joblib.load,
}


class PredictionModel:
    """A registered prediction model: maps entity.attribute columns to model inputs and
    returns its post-processed prediction."""

    def __init__(self, meta: Dict[str, Any]):
        self.meta = meta
        self.model_id = meta["model_id"]
        self.unit_entity = meta["unit_entity_name"]
        self.prediction_kind = meta["prediction_kind"]
        self.decision_threshold = meta["decision_threshold"]
        # The column this model's predictions populate, e.g. "subscription_predictions.churn_probability".
        self.output_attribute = f"{meta['prediction_entity_name']}.{meta['prediction_attribute_name']}"
        # entity.attribute -> model input name, in model input order.
        self.inputs = {
            f"{f['entity_name']}.{f['attribute_name']}": f["model_input_name"]
            for f in meta["predictive_features"]
        }
        if not self.inputs:
            raise ValueError(f"Model {self.model_id} has no predictive features")

        loader = ARTIFACT_LOADERS.get(meta["artifact_format"])
        if loader is None:
            raise ValueError(f"Unsupported artifact_format {meta['artifact_format']!r}")
        self.artifact = loader(PROJECT_DIR / meta["artifact_path"])

        transform = dict(meta["output_transform"] or {})
        transform_type = transform.pop("type", None)
        if transform_type and transform_type not in OUTPUT_TRANSFORMS:
            raise ValueError(f"Unsupported output_transform type {transform_type!r}")
        self._transform = (
            (lambda output: OUTPUT_TRANSFORMS[transform_type](output, **transform))
            if transform_type else (lambda output: output)
        )

        if self.prediction_kind == "probability":
            classes = [str(c) for c in self.artifact.classes_]
            if str(meta["positive_class"]) not in classes:
                raise ValueError(f"positive_class {meta['positive_class']!r} not in model classes {classes}")
            self._positive_index = classes.index(str(meta["positive_class"]))

        expected = getattr(self.artifact, "feature_names_in_", None)
        if expected is not None and list(expected) != list(self.inputs.values()):
            raise ValueError(
                f"Model {self.model_id} expects inputs {list(expected)}, "
                f"metadata maps {list(self.inputs.values())}"
            )

    @property
    def input_attributes(self) -> List[str]:
        return list(self.inputs)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Prediction for each row of a frame with entity.attribute columns."""
        model_input = frame[self.input_attributes].rename(columns=self.inputs)
        if self.prediction_kind == "probability":
            raw = self.artifact.predict_proba(model_input)[:, self._positive_index]
        else:
            raw = self.artifact.predict(model_input)
        return self._transform(np.asarray(raw))


@lru_cache(maxsize=1)
def get_models() -> Dict[int, PredictionModel]:
    """All registered models of the organization, loaded once per process."""
    return {meta["model_id"]: PredictionModel(meta) for meta in get_prediction_models()}
