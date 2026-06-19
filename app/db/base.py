"""
app/db/base.py

SQLAlchemy base — all models inherit from here.
This is what connects Python classes to actual database tables.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
