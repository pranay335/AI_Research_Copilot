import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Path to memory.db
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DB_FILE = DATA_DIR / "memory.db"


def initialize_database():
    """
    Create the SQLite database if it does not exist,
    and create the research_history table.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS research_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                topics TEXT,
                articles TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        conn.commit()


def save_research(query, topics, articles):
    """
    Save one research interaction.
    topics and articles can be stored as JSON text.
    created_at uses the current timestamp.
    """
    topics_json = json.dumps(topics)
    articles_json = json.dumps(articles)
    created_at = datetime.utcnow().isoformat()
    
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO research_history (query, topics, articles, created_at)
            VALUES (?, ?, ?, ?)
        ''', (query, topics_json, articles_json, created_at))
        conn.commit()


def _dict_factory(cursor, row):
    """
    Helper to return dictionary rows from sqlite3 instead of tuples.
    """
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d


def _parse_row(row):
    """
    Parse the JSON fields in a row dict.
    """
    try:
        row['topics'] = json.loads(row['topics']) if row['topics'] else []
    except json.JSONDecodeError:
        row['topics'] = []
        
    try:
        row['articles'] = json.loads(row['articles']) if row['articles'] else []
    except json.JSONDecodeError:
        row['articles'] = []
        
    return row


def get_research_history(limit=10):
    """
    Return the most recent research records.
    Returns structured Python data.
    """
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = _dict_factory
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM research_history
            ORDER BY created_at DESC
            LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        
    return [_parse_row(row) for row in rows]


def get_research_by_query(query, limit=10):
    """
    Search previous research queries using a simple case-insensitive LIKE search.
    Returns structured Python data.
    """
    with sqlite3.connect(DB_FILE) as conn:
        conn.row_factory = _dict_factory
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM research_history
            WHERE query LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (f"%{query}%", limit))
        rows = cursor.fetchall()
        
    return [_parse_row(row) for row in rows]
