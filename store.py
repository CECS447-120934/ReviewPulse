"""SQLite label cache: each (model, sentence) pair is sent to the LLM only once.

The cache stores labels, never reviewer names or IDs (sentences are already scrubbed by
preprocess.py before they reach here). The default file data/labels.sqlite is git-ignored.
"""
import hashlib
import json
import sqlite3
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent / "data" / "labels.sqlite"


def _key(sentence: str) -> str:
    return hashlib.sha1(sentence.strip().encode("utf-8")).hexdigest()


class LabelCache:
    def __init__(self, path: str | Path = DEFAULT_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.execute(
            "CREATE TABLE IF NOT EXISTS labels ("
            " model TEXT NOT NULL, sentence_hash TEXT NOT NULL, sentence TEXT NOT NULL,"
            " label_json TEXT NOT NULL, PRIMARY KEY (model, sentence_hash))"
        )
        self.conn.commit()

    def get(self, model: str, sentence: str) -> dict | None:
        row = self.conn.execute(
            "SELECT label_json FROM labels WHERE model = ? AND sentence_hash = ?", (model, _key(sentence))
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, model: str, sentence: str, label: dict) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO labels (model, sentence_hash, sentence, label_json) VALUES (?, ?, ?, ?)",
            (model, _key(sentence), sentence, json.dumps(label)),
        )
        self.conn.commit()

    def __len__(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
