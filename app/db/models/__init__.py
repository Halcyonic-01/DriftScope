"""
app/db/models/__init__.py

Import all models here so Alembic can discover them automatically
when generating migrations.
"""

from app.db.models.golden_case import GoldenCase
from app.db.models.eval_result import EvalResult

__all__ = ["GoldenCase", "EvalResult"]
