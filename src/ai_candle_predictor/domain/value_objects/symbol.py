import re

from pydantic.dataclasses import dataclass

SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.^=-]+$")


@dataclass(frozen=True)
class Symbol:
    value: str

    def __post_init__(self) -> None:
        if not self.value or not self.value.strip():
            raise ValueError("symbol must not be empty")
        if not SYMBOL_PATTERN.match(self.value):
            raise ValueError(f"symbol must match {SYMBOL_PATTERN.pattern!r}, got {self.value!r}")

    @property
    def base(self) -> str | None:
        parts = self.value.split("-")
        return parts[0] if len(parts) == 2 else None

    @property
    def quote(self) -> str | None:
        parts = self.value.split("-")
        return parts[1] if len(parts) == 2 else None

    @property
    def is_crypto(self) -> bool:
        return self.quote is not None and self.quote in {"USD", "USDT", "BTC", "ETH"}

    @property
    def is_index(self) -> bool:
        return self.value.startswith("^")

    def __str__(self) -> str:
        return self.value
