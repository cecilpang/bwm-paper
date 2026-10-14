"""Load the inference dataset into the BWM case study tables.

Fills subscription, content_consumption_statistics and subscription_predictions
(see bwm_case_study_ddl.sql) from the inference CSV. churn_probability is scored
with the saved churn model and prior-corrected exactly as in
notebooks/Inference.ipynb. The dataset has no recency, so that column is NULL.

Existing rows in the three tables are replaced, so the load can be re-run.

Usage:
    python database/load_case_study_data.py path/to/bwm.db [--csv CSV]
"""

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CSV = PROJECT_DIR / 'datasets' / 'case_study_data.csv'
DEFAULT_MODEL_META = PROJECT_DIR / 'saved-models' / 'churn_model_lr_meta.json'


def read_inference_data(csv_path):
    df = pd.read_csv(csv_path)
    duplicated = df['subs_id'].duplicated(keep=False)
    if duplicated.any():
        raise ValueError(
            f'{duplicated.sum():,} rows share a subs_id; '
            'subs_id must be unique to be the case study primary key'
        )
    return df


def predict_churn_probability(df, model_meta_path):
    model_meta = json.loads(Path(model_meta_path).read_text(encoding='utf-8'))
    pipe = joblib.load(Path(model_meta_path).parent / model_meta['model_file'])
    positive_class_index = list(pipe.classes_).index(model_meta['positive_class'])
    raw_probability = pipe.predict_proba(df[model_meta['features']])[:, positive_class_index]
    beta = model_meta['probability_correction']['beta']
    return beta * raw_probability / (beta * raw_probability + 1 - raw_probability)


def load_case_study_data(conn, csv_path=DEFAULT_CSV, model_meta_path=DEFAULT_MODEL_META):
    """Replace the case study rows with the inference data; return the row count."""
    df = read_inference_data(csv_path)
    churn_probability = predict_churn_probability(df, model_meta_path)

    # .tolist() converts numpy scalars to Python types that sqlite3 can bind.
    subs_id = df['subs_id'].tolist()
    conn.execute('PRAGMA foreign_keys = ON')
    with conn:
        conn.execute('DELETE FROM subscription_predictions')
        conn.execute('DELETE FROM content_consumption_statistics')
        conn.execute('DELETE FROM subscription')
        conn.executemany(
            'INSERT INTO subscription (subs_id, monthly_cost, tenure_days) '
            'VALUES (?, ?, ?)',
            zip(subs_id, df['monthly_cost'].tolist(), df['tenure'].tolist()),
        )
        conn.executemany(
            'INSERT INTO content_consumption_statistics '
            '(subs_id, frequency, breadth, recency) VALUES (?, ?, ?, NULL)',
            zip(subs_id, df['frequency'].tolist(), df['breadth'].tolist()),
        )
        conn.executemany(
            'INSERT INTO subscription_predictions (subs_id, churn_probability) '
            'VALUES (?, ?)',
            zip(subs_id, churn_probability.tolist()),
        )
    return len(df)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('db', type=Path, help='SQLite database with the case study tables')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV, help='inference dataset')
    parser.add_argument('--model-meta', type=Path, default=DEFAULT_MODEL_META,
                        help='saved model metadata JSON')
    args = parser.parse_args()

    if not args.db.exists():
        parser.error(f'{args.db} does not exist; build it with build_database.py')
    with sqlite3.connect(args.db) as conn:
        row_count = load_case_study_data(conn, args.csv, args.model_meta)
    print(f'Loaded {row_count:,} subscriptions into {args.db}')


if __name__ == '__main__':
    main()
