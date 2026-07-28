"""
tests/conftest.py

Shared fixtures for the integration and e2e test suites.

Unit tests (tests/test_*.py) don't use anything here — they mock their
dependencies and never touch a database. Integration and e2e tests
(tests/integration/, tests/e2e/) run against a real PostgreSQL database
and need the fixtures below:

  - db_engine:   session-scoped engine; skips the whole run if no DB
                 is reachable, and makes sure all tables exist.
  - db_session:  a Session bound to a SAVEPOINT-wrapped connection, so
                 every test's changes are rolled back afterwards —
                 including changes made via nested db.commit() calls
                 inside route handlers. Safe to point at a shared dev
                 database; nothing written by tests is ever persisted.
  - client:      a FastAPI TestClient wired to use db_session instead
                 of a real per-request session, running the app's
                 lifespan (startup/shutdown) for realistic e2e coverage.

By default this targets the same database as the app itself
(DATABASE_URL from .env). Point TEST_DATABASE_URL at a separate
database (e.g. in CI) to isolate test runs entirely.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.base import Base
from app.db.models import golden_case, eval_result, centroid_history  # noqa: F401  (registers tables)
from app.db.session import get_db_session
from app.main import app


def _test_database_url() -> str:
    return os.environ.get("TEST_DATABASE_URL", settings.database_url)


@pytest.fixture(scope="session")
def db_engine():
    url = _test_database_url()
    engine = create_engine(url, pool_pre_ping=True)

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment dependent
        engine.dispose()
        pytest.skip(f"No reachable database at {url!r} for integration/e2e tests: {exc}")

    Base.metadata.create_all(bind=engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(db_engine):
    """
    A Session bound to a connection-level transaction with a SAVEPOINT.

    Route handlers call db.commit() themselves (e.g. POST /cases). Without
    the SAVEPOINT trick that would end our outer transaction and persist
    data to the real database. Instead we restart a SAVEPOINT every time
    one ends, so the outer transaction (rolled back at teardown) is the
    only thing that ever actually lands.

    See: https://docs.sqlalchemy.org/en/20/orm/session_transaction.html
         #joining-a-session-into-an-external-transaction-such-as-for-test-suites
    """
    connection = db_engine.connect()
    outer_transaction = connection.begin()
    session = Session(bind=connection)

    nested = connection.begin_nested()

    @event.listens_for(session, "after_transaction_end")
    def _restart_savepoint(sess, transaction):
        nonlocal nested
        if not nested.is_active:
            nested = connection.begin_nested()

    try:
        yield session
    finally:
        session.close()
        if outer_transaction.is_active:
            outer_transaction.rollback()
        connection.close()


@pytest.fixture()
def client(db_session):
    """FastAPI TestClient with DB access routed through the isolated test session."""

    def _override_get_db_session():
        yield db_session

    app.dependency_overrides[get_db_session] = _override_get_db_session
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_db_session, None)
