-- BWM case study schema (SQLite), generated from the "Case Study ERD" page of
-- bwm_case_study_ddl.csv, plus the rows that register it in the AUTOBUS
-- metadata tables. Run after autobus_ddl.sql against the same database.
-- SQLite only enforces foreign keys when enabled on each connection:
PRAGMA foreign_keys = ON;

BEGIN TRANSACTION;

-- ---------------------------------------------------------------------------
-- Case study tables
-- ---------------------------------------------------------------------------

CREATE TABLE subscription (
  subs_id      INTEGER PRIMARY KEY,
  monthly_cost REAL,
  tenure_days  INTEGER
);

-- One row per subscription (1:1 with subscription).
CREATE TABLE content_consumption_statistics (
  subs_id   INTEGER PRIMARY KEY,
  frequency REAL,
  breadth   REAL,
  recency   INTEGER,
  FOREIGN KEY (subs_id)
      REFERENCES subscription (subs_id)
);

-- Output of the churn prediction model (1:1 with subscription).
CREATE TABLE subscription_predictions (
  subs_id           INTEGER PRIMARY KEY,
  churn_probability REAL,
  FOREIGN KEY (subs_id)
      REFERENCES subscription (subs_id)
);

-- ---------------------------------------------------------------------------
-- AUTOBUS metadata for the case study
-- ---------------------------------------------------------------------------

INSERT INTO organization (org_id, org_name, description) VALUES
  (1, 'BWM Case Study', 'Subscription churn case study');

INSERT INTO model_type (model_type_id, model_type) VALUES
  (1, 'Logistic Regression');

INSERT INTO entity (org_id, entity_name, description) VALUES
  (1, 'subscription',                   'A customer subscription'),
  (1, 'content_consumption_statistics', 'Content consumption statistics per subscription'),
  (1, 'subscription_predictions',       'Model predictions per subscription');

INSERT INTO attribute (org_id, entity_name, attribute_name, description, data_type, is_key) VALUES
  (1, 'subscription',                   'subs_id',           'Subscription identifier',                            'id',      1),
  (1, 'subscription',                   'monthly_cost',      'Monthly cost of the subscription',                   'numeric', 0),
  (1, 'subscription',                   'tenure_days',       'Days since the subscription started',                'numeric', 0),
  (1, 'content_consumption_statistics', 'subs_id',           'Subscription identifier',                            'id',      1),
  (1, 'content_consumption_statistics', 'frequency',         'How often the subscriber consumes content',          'numeric', 0),
  (1, 'content_consumption_statistics', 'breadth',           'How widely the subscriber consumes content',         'numeric', 0),
  (1, 'content_consumption_statistics', 'recency',           'Days since the subscriber last consumed content',    'numeric', 0),
  (1, 'subscription_predictions',       'subs_id',           'Subscription identifier',                            'id',      1),
  (1, 'subscription_predictions',       'churn_probability', 'Predicted probability that the subscription churns', 'numeric', 0);

INSERT INTO entity_relationship (org_id, from_entity, from_attribute, to_entity, to_attribute, cardinality) VALUES
  (1, 'content_consumption_statistics', 'subs_id', 'subscription', 'subs_id', '1:1'),
  (1, 'subscription_predictions',       'subs_id', 'subscription', 'subs_id', '1:1');

-- output_transform and decision_threshold come from probability_correction in
-- saved-models/churn_model_lr_meta.json.
INSERT INTO prediction_model (org_id, model_id, short_desc, description, model_type_id,
                              prediction_entity_name, prediction_attribute_name,
                              unit_entity_name, artifact_path, artifact_format, prediction_kind,
                              positive_class, output_transform, decision_threshold) VALUES
  (1, 1, 'Churn Prediction Model',
   'Logistic regression churn model (saved-models/churn_model_lr.joblib), prior-corrected probabilities',
   1, 'subscription_predictions', 'churn_probability',
   'subscription', 'saved-models/churn_model_lr.joblib', 'sklearn_joblib', 'probability',
   '1', '{"type": "prior_correction", "beta": 0.07429096461828189}', 0.08508649592084618);

-- Features of the trained model (saved-models/churn_model_lr_meta.json):
-- frequency, breadth, tenure, monthly_cost.
INSERT INTO predictive_feature (org_id, model_id, entity_name, attribute_name, description,
                                model_input_name, input_order) VALUES
  (1, 1, 'content_consumption_statistics', 'frequency',    'Model feature: frequency',                               'frequency',    1),
  (1, 1, 'content_consumption_statistics', 'breadth',      'Model feature: breadth',                                 'breadth',      2),
  (1, 1, 'subscription',                   'tenure_days',  'Model feature: tenure',                                  'tenure',       3),
  (1, 1, 'subscription',                   'monthly_cost', 'Model feature: monthly_cost (total_cost * 30 / tenure)', 'monthly_cost', 4);

-- Feasible actions 
INSERT INTO feasible_action (org_id, action_id, entity_name, attribute_name, description, action_type) VALUES
  (1, 1, 'subscription', 'monthly_cost', 'Reduce monthly cost by 5%, 10%, 15%, 20% or 25%', 'scale');

INSERT INTO action_level (org_id, action_id, level_id, value, label) VALUES
  (1, 1, 1, 0.95, '5% discount on monthly cost'),
  (1, 1, 2, 0.90, '10% discount on monthly cost'),
  (1, 1, 3, 0.85, '15% discount on monthly cost'),
  (1, 1, 4, 0.80, '20% discount on monthly cost'),
  (1, 1, 5, 0.75, '25% discount on monthly cost');

-- KPIs
INSERT INTO kpi (org_id, kpi_id, kpi_name, description, unit_entity_name, aggregation, expression, unit, category) VALUES
  (1, 1, 'mean_churn_probability', 'Churn rate: mean predicted churn probability of the subscriptions',
   'subscription', 'mean', 'subscription_predictions.churn_probability', 'probability', 'outcome'),
  (1, 2, 'predicted_churn_rate', 'Share of subscriptions classified as churning by the decision threshold',
   'subscription', 'mean', 'subscription_predictions.churn_probability >= 0.08508649592084618', 'rate', 'outcome'),
  (1, 3, 'expected_monthly_revenue', 'Monthly revenue expected to be retained: monthly cost times retention probability',
   'subscription', 'sum', 'subscription.monthly_cost * (1 - subscription_predictions.churn_probability)', 'USD', 'revenue'),
  (1, 4, 'subscription_count', 'Number of subscriptions',
   'subscription', 'count', 'subscription.subs_id', 'subscriptions', 'volume');

INSERT INTO kpi_attribute (org_id, kpi_id, entity_name, attribute_name) VALUES
  (1, 1, 'subscription_predictions', 'churn_probability'),
  (1, 2, 'subscription_predictions', 'churn_probability'),
  (1, 3, 'subscription',             'monthly_cost'),
  (1, 3, 'subscription_predictions', 'churn_probability'),
  (1, 4, 'subscription',             'subs_id');

COMMIT;
