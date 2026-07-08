import sqlite3
import json
import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

class DocumentStore:
    """Persistent key-value store for parent section text.
    
    Used at query time to expand retrieved leaf chunks to their
    full parent section before passing to the LLM.
    """
    
    def __init__(self, db_path: str = "data/doc_store.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS parents (
                    parent_id TEXT PRIMARY KEY,
                    book_name TEXT NOT NULL,
                    chapter TEXT,
                    section TEXT,
                    pages TEXT,
                    text TEXT NOT NULL,
                    token_count INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            # Create index if it doesn't exist
            conn.execute('CREATE INDEX IF NOT EXISTS idx_parents_book ON parents(book_name)')

    def store_parent(self, parent_id: str, data: dict) -> None:
        """Store a parent section in the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO parents 
                (parent_id, book_name, chapter, section, pages, text, token_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                parent_id,
                data.get('book_name', ''),
                data.get('chapter', ''),
                data.get('section', ''),
                json.dumps(data.get('pages', [])),
                data.get('text', ''),
                data.get('token_count', 0)
            ))

    def get_parent(self, parent_id: str) -> Optional[Dict]:
        """Retrieve a single parent section by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                'SELECT * FROM parents WHERE parent_id = ?', 
                (parent_id,)
            ).fetchone()
            
            if not row:
                return None
                
            return {
                'parent_id': row['parent_id'],
                'book_name': row['book_name'],
                'chapter': row['chapter'],
                'section': row['section'],
                'pages': json.loads(row['pages']) if row['pages'] else [],
                'text': row['text'],
                'token_count': row['token_count']
            }

    def get_parents_batch(self, parent_ids: List[str]) -> List[Dict]:
        """Retrieve multiple parent sections by ID."""
        if not parent_ids:
            return []
            
        placeholders = ','.join('?' * len(parent_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f'SELECT * FROM parents WHERE parent_id IN ({placeholders})',
                parent_ids
            ).fetchall()
            
            results = []
            for row in rows:
                results.append({
                    'parent_id': row['parent_id'],
                    'book_name': row['book_name'],
                    'chapter': row['chapter'],
                    'section': row['section'],
                    'pages': json.loads(row['pages']) if row['pages'] else [],
                    'text': row['text'],
                    'token_count': row['token_count']
                })
            return results

    def clear_book(self, book_name: str) -> None:
        """Delete all parent sections for a specific book."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM parents WHERE book_name = ?', (book_name,))

    def clear(self) -> None:
        """Delete all parent sections from the database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM parents')

    def stats(self) -> dict:
        """Return basic statistics about the document store."""
        with sqlite3.connect(self.db_path) as conn:
            total_parents = conn.execute('SELECT COUNT(*) FROM parents').fetchone()[0]
            books = [row[0] for row in conn.execute('SELECT DISTINCT book_name FROM parents').fetchall()]
            
            return {
                "total_parents": total_parents,
                "books": books
            }
