from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    data_raw_dir: Path = ROOT_DIR / "data" / "raw"
    data_processed_dir: Path = ROOT_DIR / "data" / "processed"
    data_interim_dir: Path = ROOT_DIR / "data" / "interim"
    models_dir: Path = ROOT_DIR / "models"
    reports_dir: Path = ROOT_DIR / "reports"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["json", "console"] = "console"

    yfinance_timeout: int = 30
    yfinance_max_retries: int = 3

    default_symbols: list[str] = [
        "BTC-USD",
        "ETH-USD",
        "^NSEI",
        "RELIANCE.NS",
    ]
    default_start_date: str = "2020-01-01"


settings = Settings()
