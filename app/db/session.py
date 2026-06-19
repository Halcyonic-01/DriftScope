"""
app/db/session.py

Database session management.

SQLAlchemy works in two layers:
  - Engine: the actual connection to PostgreSQL (created once at startup)
  - Session: a temporary "workspace" for reading/writing data (created per request)

We use a context manager pattern so sessions are always properly closed,
even if an error occurs mid-request.
"""

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# Create the engine once — this is the persistent connection pool to PostgreSQL
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,   # checks connection health before using it
    echo=False,           # set True to see all SQL queries in logs (useful for debugging)
)

# Session factory — call SessionLocal() to get a new session
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,   # we control when to commit (save) changes
    autoflush=False,    # we control when to flush (send) changes to DB
)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """
    Context manager that yields a database session and ensures it's
    always closed afterwards — even if an exception is raised.

    Usage:
        with get_db() as db:
            cases = db.query(GoldenCase).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db_session() -> Generator[Session, None, None]:
    """
    FastAPI dependency — inject a DB session into route handlers.

    Usage in a route:
        @app.get("/cases")
        def list_cases(db: Session = Depends(get_db_session)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
