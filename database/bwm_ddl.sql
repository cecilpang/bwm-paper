-- BWM schema (SQLite), generated from the ERD page of autobus_ddl.csv, plus the
-- simulation metadata (attribute types/keys, entity_relationship, model runtime
-- columns, feature input mapping, action_type, action_level, kpi, kpi_attribute)
-- that is not yet in the ERD.
-- All tables except model_type are partitioned by organization (org_id);
-- model_type is global.
-- SQLite only enforces foreign keys when enabled on each connection:
PRAGMA foreign_keys = ON;

CREATE TABLE organization (
  org_id      INTEGER PRIMARY KEY,
  org_name    TEXT,
  description TEXT
);

CREATE TABLE model_type (
  model_type_id INTEGER PRIMARY KEY,
  model_type    TEXT
);

CREATE TABLE entity (
  org_id      INTEGER NOT NULL,
  entity_name TEXT    NOT NULL,
  description TEXT,
  PRIMARY KEY (org_id, entity_name),
  FOREIGN KEY (org_id)
      REFERENCES organization (org_id)
);

CREATE TABLE attribute (
  org_id         INTEGER NOT NULL,
  entity_name    TEXT    NOT NULL,
  attribute_name TEXT    NOT NULL,
  description    TEXT,
  data_type      TEXT    NOT NULL DEFAULT 'numeric'
                 CHECK (data_type IN ('numeric', 'categorical', 'date', 'id')),
  is_key         INTEGER NOT NULL DEFAULT 0 CHECK (is_key IN (0, 1)),
  PRIMARY KEY (org_id, entity_name, attribute_name),
  FOREIGN KEY (org_id, entity_name)
      REFERENCES entity (org_id, entity_name)
);

-- How entities join: from_entity.from_attribute references to_entity.to_attribute.
-- Only 1:1 and N:1 relationships can be joined without aggregation.
CREATE TABLE entity_relationship (
  org_id         INTEGER NOT NULL,
  from_entity    TEXT    NOT NULL,
  from_attribute TEXT    NOT NULL,
  to_entity      TEXT    NOT NULL,
  to_attribute   TEXT    NOT NULL,
  cardinality    TEXT    NOT NULL CHECK (cardinality IN ('1:1', 'N:1', '1:N')),
  PRIMARY KEY (org_id, from_entity, from_attribute, to_entity, to_attribute),
  FOREIGN KEY (org_id, from_entity, from_attribute)
      REFERENCES attribute (org_id, entity_name, attribute_name),
  FOREIGN KEY (org_id, to_entity, to_attribute)
      REFERENCES attribute (org_id, entity_name, attribute_name)
);

CREATE TABLE prediction_model (
  org_id                    INTEGER NOT NULL,
  model_id                  INTEGER NOT NULL,
  short_desc                TEXT,
  description               TEXT,
  model_type_id             INTEGER NOT NULL,
  prediction_entity_name    TEXT    NOT NULL,
  prediction_attribute_name TEXT    NOT NULL,
  -- Entity whose rows the model scores (one prediction per row).
  unit_entity_name          TEXT    NOT NULL,
  -- Model artifact, relative to the project directory.
  artifact_path             TEXT    NOT NULL,
  artifact_format           TEXT    NOT NULL CHECK (artifact_format IN ('sklearn_joblib')),
  prediction_kind           TEXT    NOT NULL CHECK (prediction_kind IN ('probability', 'value', 'class')),
  -- Class whose probability is predicted (prediction_kind = 'probability').
  positive_class            TEXT,
  -- JSON post-processing of the raw output, e.g. {"type": "prior_correction", "beta": 0.07}.
  output_transform          TEXT,
  -- Threshold on the post-processed output that classifies a row as positive.
  decision_threshold        REAL,
  PRIMARY KEY (org_id, model_id),
  FOREIGN KEY (model_type_id)
      REFERENCES model_type (model_type_id),
  FOREIGN KEY (org_id, prediction_entity_name, prediction_attribute_name)
      REFERENCES attribute (org_id, entity_name, attribute_name),
  FOREIGN KEY (org_id, unit_entity_name)
      REFERENCES entity (org_id, entity_name)
);

CREATE TABLE predictive_feature (
  org_id         INTEGER NOT NULL,
  model_id       INTEGER NOT NULL,
  entity_name    TEXT    NOT NULL,
  attribute_name TEXT    NOT NULL,
  description    TEXT,
  -- Input column name the model artifact expects, and its position in the input.
  model_input_name TEXT  NOT NULL,
  input_order    INTEGER NOT NULL,
  PRIMARY KEY (org_id, model_id, entity_name, attribute_name),
  UNIQUE (org_id, model_id, model_input_name),
  UNIQUE (org_id, model_id, input_order),
  FOREIGN KEY (org_id, model_id)
      REFERENCES prediction_model (org_id, model_id),
  FOREIGN KEY (org_id, entity_name, attribute_name)
      REFERENCES attribute (org_id, entity_name, attribute_name)
);

CREATE TABLE feasible_action (
  org_id         INTEGER NOT NULL,
  action_id      INTEGER NOT NULL,
  entity_name    TEXT    NOT NULL,
  attribute_name TEXT    NOT NULL,
  description    TEXT,
  -- How a level's value changes the attribute: scale (x * value), add (x + value), set (value).
  action_type    TEXT    NOT NULL CHECK (action_type IN ('scale', 'add', 'set')),
  PRIMARY KEY (org_id, action_id),
  FOREIGN KEY (org_id, entity_name, attribute_name)
      REFERENCES attribute (org_id, entity_name, attribute_name)
);

-- Allowed parameter values of a feasible action, e.g. scale 0.95 = "5% discount".
CREATE TABLE action_level (
  org_id   INTEGER NOT NULL,
  action_id INTEGER NOT NULL,
  level_id INTEGER NOT NULL,
  value    REAL    NOT NULL,
  label    TEXT    NOT NULL,
  PRIMARY KEY (org_id, action_id, level_id),
  FOREIGN KEY (org_id, action_id)
      REFERENCES feasible_action (org_id, action_id)
);

-- Named metrics: aggregation(expression) over the rows of unit_entity_name. The expression
-- uses entity.attribute names (including model predictions), numbers, arithmetic,
-- comparisons and the functions abs, log, exp, minimum and maximum.
CREATE TABLE kpi (
  org_id      INTEGER NOT NULL,
  kpi_id      INTEGER NOT NULL,
  kpi_name    TEXT    NOT NULL,
  description TEXT,
  -- Entity whose rows are aggregated; the expression may use related entities.
  unit_entity_name TEXT NOT NULL,
  aggregation TEXT    NOT NULL CHECK (aggregation IN ('sum', 'mean', 'count', 'min', 'max')),
  expression  TEXT    NOT NULL,
  -- 'probability' or 'rate' KPIs are fractions (targets may be given in percentage points).
  unit        TEXT,
  category    TEXT    NOT NULL CHECK (category IN ('outcome', 'revenue', 'volume')),
  PRIMARY KEY (org_id, kpi_id),
  UNIQUE (org_id, kpi_name),
  FOREIGN KEY (org_id, unit_entity_name)
      REFERENCES entity (org_id, entity_name)
);

-- Attributes a KPI expression uses (lineage; checked against the expression).
CREATE TABLE kpi_attribute (
  org_id         INTEGER NOT NULL,
  kpi_id         INTEGER NOT NULL,
  entity_name    TEXT    NOT NULL,
  attribute_name TEXT    NOT NULL,
  PRIMARY KEY (org_id, kpi_id, entity_name, attribute_name),
  FOREIGN KEY (org_id, kpi_id)
      REFERENCES kpi (org_id, kpi_id),
  FOREIGN KEY (org_id, entity_name, attribute_name)
      REFERENCES attribute (org_id, entity_name, attribute_name)
);

CREATE TABLE business_rule (
  org_id           INTEGER NOT NULL,
  business_rule_id INTEGER NOT NULL,
  description      TEXT,
  PRIMARY KEY (org_id, business_rule_id),
  FOREIGN KEY (org_id)
      REFERENCES organization (org_id)
);

-- Links a business rule to the entities (and optionally the attributes) it governs.
CREATE TABLE rule_entity_link (
  org_id           INTEGER NOT NULL,
  business_rule_id INTEGER NOT NULL,
  entity_name      TEXT    NOT NULL,
  attribute_name   TEXT,
  FOREIGN KEY (org_id, business_rule_id)
      REFERENCES business_rule (org_id, business_rule_id),
  FOREIGN KEY (org_id, entity_name)
      REFERENCES entity (org_id, entity_name),
  FOREIGN KEY (org_id, entity_name, attribute_name)
      REFERENCES attribute (org_id, entity_name, attribute_name)
);
