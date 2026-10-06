import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
AUTOBUS_DB_PATH = Path(os.getenv("AUTOBUS_DB_PATH", PROJECT_DIR / "database" / "autobus.db"))
DEFAULT_ORG_ID = int(os.getenv("AUTOBUS_ORG_ID", "1"))


def _connect() -> sqlite3.Connection:
    if not AUTOBUS_DB_PATH.exists():
        raise FileNotFoundError(
            f"AUTOBUS database not found at {AUTOBUS_DB_PATH}; build it with database/build_database.py"
        )
    conn = sqlite3.connect(f"file:{AUTOBUS_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def get_prediction_models(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """Prediction models registered for the organization, with their predictive features."""
    with closing(_connect()) as conn:
        models = conn.execute(
            """
            SELECT pm.model_id, pm.short_desc, pm.description, mt.model_type,
                   pm.prediction_entity_name, pm.prediction_attribute_name,
                   a.description AS prediction_attribute_description
            FROM prediction_model pm
            JOIN model_type mt ON mt.model_type_id = pm.model_type_id
            LEFT JOIN attribute a
              ON a.org_id = pm.org_id
             AND a.entity_name = pm.prediction_entity_name
             AND a.attribute_name = pm.prediction_attribute_name
            WHERE pm.org_id = ?
            ORDER BY pm.model_id
            """,
            (org_id,),
        ).fetchall()
        features = conn.execute(
            """
            SELECT model_id, entity_name, attribute_name, description
            FROM predictive_feature
            WHERE org_id = ?
            ORDER BY model_id, entity_name, attribute_name
            """,
            (org_id,),
        ).fetchall()

    result = []
    for model in models:
        model_info = dict(model)
        model_info["predictive_features"] = [
            {k: feature[k] for k in ("entity_name", "attribute_name", "description")}
            for feature in features
            if feature["model_id"] == model["model_id"]
        ]
        result.append(model_info)
    return result


def get_feasible_actions(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """Feasible actions for the organization and the attributes they act on."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT fa.action_id, fa.entity_name, fa.attribute_name, fa.description,
                   a.description AS attribute_description
            FROM feasible_action fa
            LEFT JOIN attribute a
              ON a.org_id = fa.org_id
             AND a.entity_name = fa.entity_name
             AND a.attribute_name = fa.attribute_name
            WHERE fa.org_id = ?
            ORDER BY fa.action_id
            """,
            (org_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_business_rules(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """Business rules for the organization and the entities/attributes they govern."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT br.business_rule_id, br.description,
                   rel.entity_name, rel.attribute_name
            FROM business_rule br
            LEFT JOIN rule_entity_link rel
              ON rel.org_id = br.org_id
             AND rel.business_rule_id = br.business_rule_id
            WHERE br.org_id = ?
            ORDER BY br.business_rule_id
            """,
            (org_id,),
        ).fetchall()

    rules: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        rule = rules.setdefault(
            row["business_rule_id"],
            {"business_rule_id": row["business_rule_id"], "description": row["description"], "governs": []},
        )
        if row["entity_name"]:
            rule["governs"].append({"entity_name": row["entity_name"], "attribute_name": row["attribute_name"]})
    return list(rules.values())


def get_schema(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """Tables of the organization's entities with their columns and descriptions."""
    with closing(_connect()) as conn:
        entities = conn.execute(
            "SELECT entity_name, description FROM entity WHERE org_id = ? ORDER BY entity_name",
            (org_id,),
        ).fetchall()
        attributes = conn.execute(
            "SELECT entity_name, attribute_name, description FROM attribute WHERE org_id = ?",
            (org_id,),
        ).fetchall()
        descriptions = {(a["entity_name"], a["attribute_name"]): a["description"] for a in attributes}

        result = []
        for entity in entities:
            columns = conn.execute(
                "SELECT name, type FROM pragma_table_info(?)", (entity["entity_name"],)
            ).fetchall()
            result.append({
                "table": entity["entity_name"],
                "description": entity["description"],
                "columns": [
                    {
                        "name": column["name"],
                        "type": column["type"],
                        "description": descriptions.get((entity["entity_name"], column["name"])),
                    }
                    for column in columns
                ],
            })
    return result


def run_select_query(sql: str, max_rows: int = 200) -> Dict[str, Any]:
    """Run a single read-only SELECT/WITH statement and return at most max_rows rows."""
    statement = sql.strip().rstrip(";").strip()
    if not re.match(r"(?is)^(select|with)\b", statement):
        raise ValueError("Only a single SELECT or WITH statement is allowed")
    if ";" in statement:
        raise ValueError("Only a single SQL statement is allowed")

    with closing(_connect()) as conn:
        cursor = conn.execute(statement)
        rows = cursor.fetchmany(max_rows + 1)
    return {
        "columns": [d[0] for d in cursor.description] if cursor.description else [],
        "rows": [dict(row) for row in rows[:max_rows]],
        "truncated": len(rows) > max_rows,
    }


def load_subscription_data() -> pd.DataFrame:
    """Subscriptions joined with their content consumption statistics."""
    with closing(_connect()) as conn:
        return pd.read_sql_query(
            """
            SELECT s.subs_id, s.monthly_cost, s.tenure_days,
                   c.frequency, c.breadth, c.recency
            FROM subscription s
            JOIN content_consumption_statistics c ON c.subs_id = s.subs_id
            ORDER BY s.subs_id
            """,
            conn,
        )
