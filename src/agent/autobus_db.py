import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any, Dict, Iterable, List

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
    """Prediction models registered for the organization, with their runtime metadata and
    predictive features in model input order."""
    with closing(_connect()) as conn:
        models = conn.execute(
            """
            SELECT pm.model_id, pm.short_desc, pm.description, mt.model_type,
                   pm.prediction_entity_name, pm.prediction_attribute_name,
                   a.description AS prediction_attribute_description,
                   pm.unit_entity_name, pm.artifact_path, pm.artifact_format,
                   pm.prediction_kind, pm.positive_class, pm.output_transform,
                   pm.decision_threshold
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
            SELECT model_id, entity_name, attribute_name, description,
                   model_input_name, input_order
            FROM predictive_feature
            WHERE org_id = ?
            ORDER BY model_id, input_order
            """,
            (org_id,),
        ).fetchall()

    result = []
    for model in models:
        model_info = dict(model)
        model_info["output_transform"] = (
            json.loads(model["output_transform"]) if model["output_transform"] else None
        )
        model_info["predictive_features"] = [
            {k: feature[k] for k in ("entity_name", "attribute_name", "description", "model_input_name")}
            for feature in features
            if feature["model_id"] == model["model_id"]
        ]
        result.append(model_info)
    return result


def get_feasible_actions(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """Feasible actions for the organization, the attributes they act on and their allowed levels."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT fa.action_id, fa.entity_name, fa.attribute_name, fa.description,
                   fa.action_type, a.description AS attribute_description
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
        levels = conn.execute(
            """
            SELECT action_id, level_id, value, label
            FROM action_level
            WHERE org_id = ?
            ORDER BY action_id, level_id
            """,
            (org_id,),
        ).fetchall()

    result = []
    for row in rows:
        action = dict(row)
        action["levels"] = [
            {k: level[k] for k in ("level_id", "value", "label")}
            for level in levels
            if level["action_id"] == row["action_id"]
        ]
        result.append(action)
    return result


def get_kpis(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """KPIs (named aggregate metrics) of the organization with the attributes they use."""
    with closing(_connect()) as conn:
        kpis = conn.execute(
            """
            SELECT kpi_id, kpi_name, description, unit_entity_name, aggregation,
                   expression, unit, category
            FROM kpi
            WHERE org_id = ?
            ORDER BY kpi_id
            """,
            (org_id,),
        ).fetchall()
        attributes = conn.execute(
            "SELECT kpi_id, entity_name, attribute_name FROM kpi_attribute WHERE org_id = ?",
            (org_id,),
        ).fetchall()

    result = []
    for kpi in kpis:
        kpi_info = dict(kpi)
        kpi_info["attributes"] = sorted(
            f"{a['entity_name']}.{a['attribute_name']}" for a in attributes if a["kpi_id"] == kpi["kpi_id"]
        )
        result.append(kpi_info)
    return result


def get_attributes(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """All attributes of the organization's entities with their data type and key flag."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT entity_name, attribute_name, description, data_type, is_key
            FROM attribute
            WHERE org_id = ?
            ORDER BY entity_name, attribute_name
            """,
            (org_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_relationships(org_id: int = DEFAULT_ORG_ID) -> List[Dict[str, Any]]:
    """Relationships between the organization's entities."""
    with closing(_connect()) as conn:
        rows = conn.execute(
            """
            SELECT from_entity, from_attribute, to_entity, to_attribute, cardinality
            FROM entity_relationship
            WHERE org_id = ?
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


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _join_paths(unit_entity: str, org_id: int) -> Dict[str, List[Dict[str, Any]]]:
    """
    Join path from the unit entity to every entity reachable without multiplying rows,
    i.e. each unit row matches at most one row of the joined entity.
    """
    edges = []
    for r in get_relationships(org_id):
        # (entity joined from, its column, entity joined to, its column)
        if r["cardinality"] in ("1:1", "N:1"):
            edges.append((r["from_entity"], r["from_attribute"], r["to_entity"], r["to_attribute"]))
        if r["cardinality"] in ("1:1", "1:N"):
            edges.append((r["to_entity"], r["to_attribute"], r["from_entity"], r["from_attribute"]))

    paths: Dict[str, List[Dict[str, Any]]] = {unit_entity: []}
    frontier = [unit_entity]
    while frontier:
        current = frontier.pop(0)
        for source, source_column, target, target_column in edges:
            if source == current and target not in paths:
                paths[target] = paths[current] + [{
                    "source": source, "source_column": source_column,
                    "target": target, "target_column": target_column,
                }]
                frontier.append(target)
    return paths


def load_analysis_frame(
    unit_entity: str, attributes: Iterable[str], org_id: int = DEFAULT_ORG_ID
) -> pd.DataFrame:
    """
    One row per unit entity row with the requested "entity.attribute" columns, joining
    related entities along entity_relationship.
    """
    known = {(a["entity_name"], a["attribute_name"]) for a in get_attributes(org_id)}
    requested = []
    for name in dict.fromkeys(attributes):
        entity, _, attribute = name.partition(".")
        if (entity, attribute) not in known:
            raise ValueError(f"Unknown attribute {name!r}; use entity.attribute names from the metadata")
        requested.append((entity, attribute))

    paths = _join_paths(unit_entity, org_id)
    aliases = {unit_entity: "t0"}
    joins = []
    for entity in dict.fromkeys(e for e, _ in requested):
        if entity not in paths:
            raise ValueError(
                f"Entity {entity!r} cannot be joined one-to-one to {unit_entity!r}; "
                "check entity_relationship"
            )
        for step in paths[entity]:
            if step["target"] in aliases:
                continue
            aliases[step["target"]] = f"t{len(aliases)}"
            joins.append(
                f"LEFT JOIN {_quote(step['target'])} {aliases[step['target']]} "
                f"ON {aliases[step['target']]}.{_quote(step['target_column'])} = "
                f"{aliases[step['source']]}.{_quote(step['source_column'])}"
            )

    columns = ", ".join(
        f"{aliases[e]}.{_quote(a)} AS {_quote(f'{e}.{a}')}" for e, a in requested
    )
    sql = f"SELECT {columns} FROM {_quote(unit_entity)} t0 " + " ".join(joins)
    with closing(_connect()) as conn:
        return pd.read_sql_query(sql, conn)
