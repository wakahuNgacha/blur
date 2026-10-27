import sqlite3
import json
from datetime import datetime, timezone

DB_NAME = "faces.db"


def create_database():
    """Create the SQLite database and faces table."""
    with sqlite3.connect(DB_NAME) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS faces (
                face_number INTEGER PRIMARY KEY AUTOINCREMENT,
                person_label TEXT,
                embedding TEXT NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()

    print(f"Database '{DB_NAME}' is ready.")

create_database()
