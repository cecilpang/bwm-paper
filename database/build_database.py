"""Build the BWM SQLite database with the BWM case study.

Steps:
  1. bwm_ddl.sql          - BWM metadata tables
  2. bwm_case_study_ddl.sql   - case study tables + their BWM metadata rows
  3. load_case_study_data.py  - case study rows from the inference dataset

Usage:
    python database/build_database.py [--db database/bwm.db] [--csv CSV] [--force]
"""

import argparse
import sqlite3
from pathlib import Path

from load_case_study_data import DEFAULT_CSV, DEFAULT_MODEL_META, load_case_study_data

DATABASE_DIR = Path(__file__).resolve().parent
DDL_SCRIPTS = [
    DATABASE_DIR / 'bwm_ddl.sql',
    DATABASE_DIR / 'bwm_case_study_ddl.sql',
]
TABLES = [
    'organization', 'model_type', 'entity', 'attribute', 'entity_relationship',
    'prediction_model', 'predictive_feature', 'feasible_action', 'action_level',
    'kpi', 'kpi_attribute', 'business_rule', 'rule_entity_link',
    'subscription', 'content_consumption_statistics', 'subscription_predictions',
]


def build_database(db_path, csv_path=DEFAULT_CSV, model_meta_path=DEFAULT_MODEL_META):
    with sqlite3.connect(db_path) as conn:
        for ddl_script in DDL_SCRIPTS:
            conn.executescript(ddl_script.read_text(encoding='utf-8'))
        load_case_study_data(conn, csv_path, model_meta_path)
        violations = conn.execute('PRAGMA foreign_key_check').fetchall()
        if violations:
            raise RuntimeError(f'Foreign key violations: {violations[:10]}')
        return {
            table: conn.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]
            for table in TABLES
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db', type=Path, default=DATABASE_DIR / 'bwm.db',
                        help='database file to create')
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV, help='inference dataset')
    parser.add_argument('--model-meta', type=Path, default=DEFAULT_MODEL_META,
                        help='saved model metadata JSON')
    parser.add_argument('--force', action='store_true',
                        help='overwrite the database file if it exists')
    args = parser.parse_args()

    if args.db.exists():
        if not args.force:
            parser.error(f'{args.db} already exists; pass --force to rebuild it')
        args.db.unlink()

    row_counts = build_database(args.db, args.csv, args.model_meta)
    print(f'Built {args.db}')
    for table, row_count in row_counts.items():
        print(f'  {table:<32} {row_count:>10,}')


if __name__ == '__main__':
    main()
