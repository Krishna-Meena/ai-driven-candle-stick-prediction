from __future__ import annotations

from pathlib import Path

import pytest

from ai_candle_predictor.infrastructure.models.model_registry import (
    ModelRegistry,
    RegistryEntry,
)


@pytest.fixture
def entry_btc_lr() -> RegistryEntry:
    return RegistryEntry(
        symbol="BTC-USD",
        model_type="LR",
        label="baseline_C1.0",
        filename="BTC_USD_baseline_C1.0.joblib",
        accuracy=0.72,
        precision=0.71,
        recall=0.73,
        f1=0.72,
        roc_auc=0.74,
        support=500,
    )


@pytest.fixture
def entry_btc_rf() -> RegistryEntry:
    return RegistryEntry(
        symbol="BTC-USD",
        model_type="RF",
        label="rf_n300_d10",
        filename="BTC_USD_rf_n300_d10.joblib",
        accuracy=0.68,
        precision=0.66,
        recall=0.70,
        f1=0.68,
        roc_auc=0.69,
        support=500,
    )


@pytest.fixture
def entry_eth_lr() -> RegistryEntry:
    return RegistryEntry(
        symbol="ETH-USD",
        model_type="LR",
        label="baseline_C1.0",
        filename="ETH_USD_baseline_C1.0.joblib",
        accuracy=0.65,
        precision=0.63,
        recall=0.67,
        f1=0.65,
        roc_auc=0.66,
        support=400,
    )


class TestRegistryEntry:
    def test_create(self, entry_btc_lr: RegistryEntry) -> None:
        assert entry_btc_lr.symbol == "BTC-USD"
        assert entry_btc_lr.model_type == "LR"
        assert entry_btc_lr.accuracy == 0.72

    def test_training_date_defaults_to_iso(self, entry_btc_lr: RegistryEntry) -> None:
        assert "T" in entry_btc_lr.training_date

    def test_to_dict_roundtrip(self, entry_btc_lr: RegistryEntry) -> None:
        d = {
            "symbol": "BTC-USD",
            "model_type": "LR",
            "label": "baseline_C1.0",
            "filename": "BTC_USD_baseline_C1.0.joblib",
            "accuracy": 0.72,
            "precision": 0.71,
            "recall": 0.73,
            "f1": 0.72,
            "roc_auc": 0.74,
            "support": 500,
            "training_date": entry_btc_lr.training_date,
        }
        restored = RegistryEntry(**d)  # type: ignore[arg-type]
        assert restored == entry_btc_lr


class TestModelRegistry:
    def test_empty_registry(self, tmp_path: Path) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        assert reg.list_models() == []

    def test_register_and_list(self, tmp_path: Path, entry_btc_lr: RegistryEntry) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        reg.register(entry_btc_lr)
        models = reg.list_models()
        assert len(models) == 1
        assert models[0] == entry_btc_lr

    def test_register_multiple(
        self,
        tmp_path: Path,
        entry_btc_lr: RegistryEntry,
        entry_btc_rf: RegistryEntry,
        entry_eth_lr: RegistryEntry,
    ) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        reg.register(entry_btc_lr)
        reg.register(entry_btc_rf)
        reg.register(entry_eth_lr)
        assert len(reg.list_models()) == 3

    def test_list_by_symbol(
        self,
        tmp_path: Path,
        entry_btc_lr: RegistryEntry,
        entry_btc_rf: RegistryEntry,
        entry_eth_lr: RegistryEntry,
    ) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        reg.register(entry_btc_lr)
        reg.register(entry_btc_rf)
        reg.register(entry_eth_lr)
        btc_models = reg.list_by_symbol("BTC-USD")
        assert len(btc_models) == 2
        eth_models = reg.list_by_symbol("ETH-USD")
        assert len(eth_models) == 1

    def test_get_latest(
        self,
        tmp_path: Path,
        entry_btc_lr: RegistryEntry,
        entry_btc_rf: RegistryEntry,
    ) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        reg.register(entry_btc_lr)
        reg.register(entry_btc_rf)
        latest = reg.get_latest("BTC-USD", "RF")
        assert latest is not None
        assert latest.model_type == "RF"
        assert latest.label == "rf_n300_d10"

    def test_get_latest_none(self, tmp_path: Path) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        assert reg.get_latest("BTC-USD", "XGB") is None

    def test_persistence_across_instances(
        self,
        tmp_path: Path,
        entry_btc_lr: RegistryEntry,
    ) -> None:
        path = tmp_path / "registry.json"
        reg1 = ModelRegistry(registry_path=path)
        reg1.register(entry_btc_lr)
        reg2 = ModelRegistry(registry_path=path)
        assert len(reg2.list_models()) == 1
        assert reg2.list_models()[0] == entry_btc_lr

    def test_corrupted_file_resets(self, tmp_path: Path) -> None:
        path = tmp_path / "registry.json"
        path.write_text("{invalid json}", encoding="utf-8")
        reg = ModelRegistry(registry_path=path)
        assert reg.list_models() == []

    def test_register_updates_existing_instance(
        self,
        tmp_path: Path,
        entry_btc_lr: RegistryEntry,
        entry_btc_rf: RegistryEntry,
    ) -> None:
        reg = ModelRegistry(registry_path=tmp_path / "registry.json")
        reg.register(entry_btc_lr)
        reg.register(entry_btc_rf)
        assert len(reg.list_models()) == 2
        latest = reg.get_latest("BTC-USD", "LR")
        assert latest is not None
        assert latest.label == "baseline_C1.0"
