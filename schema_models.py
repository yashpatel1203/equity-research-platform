"""
Module 1a — Database schema
Creates all tables in SQLite using SQLAlchemy.
Run this FIRST before fetching any data.
"""

from sqlalchemy import (
    create_engine, text,
    Column, String, Float, Date, Integer,
    UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase
from dotenv import load_dotenv
import os

load_dotenv()
DB_PATH = os.getenv("DB_PATH", "equity_data.db")


class Base(DeclarativeBase):
    pass


class StockPrice(Base):
    __tablename__ = "stock_prices"
    id        = Column(Integer, primary_key=True, autoincrement=True)
    ticker    = Column(String(10), nullable=False)
    date      = Column(Date, nullable=False)
    open      = Column(Float)
    high      = Column(Float)
    low       = Column(Float)
    close     = Column(Float)
    adj_close = Column(Float)
    volume    = Column(Float)
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_ticker_date"),)


class StockFeatures(Base):
    __tablename__ = "stock_features"
    id                = Column(Integer, primary_key=True, autoincrement=True)
    ticker            = Column(String(10), nullable=False)
    date              = Column(Date, nullable=False)
    daily_return      = Column(Float)
    log_return        = Column(Float)
    return_5d         = Column(Float)
    return_21d        = Column(Float)
    volatility_21d    = Column(Float)
    volatility_63d    = Column(Float)
    sma_20            = Column(Float)
    sma_50            = Column(Float)
    ema_12            = Column(Float)
    ema_26            = Column(Float)
    macd              = Column(Float)
    macd_signal       = Column(Float)
    macd_hist         = Column(Float)
    rsi_14            = Column(Float)
    roc_10            = Column(Float)
    volume_sma_20     = Column(Float)
    volume_ratio      = Column(Float)
    beta_63d          = Column(Float)
    target_direction  = Column(Integer)
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_feat_ticker_date"),)


class MacroData(Base):
    __tablename__ = "macro_data"
    id          = Column(Integer, primary_key=True, autoincrement=True)
    series_name = Column(String(50), nullable=False)
    date        = Column(Date, nullable=False)
    value       = Column(Float)
    __table_args__ = (UniqueConstraint("series_name", "date", name="uq_macro_series_date"),)


class BenchmarkPrice(Base):
    __tablename__ = "benchmark_prices"
    id           = Column(Integer, primary_key=True, autoincrement=True)
    date         = Column(Date, nullable=False, unique=True)
    close        = Column(Float)
    adj_close    = Column(Float)
    daily_return = Column(Float)


def create_database():
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    Base.metadata.create_all(engine)
    print(f"Database ready: {DB_PATH}")
    return engine


if __name__ == "__main__":
    create_database()
