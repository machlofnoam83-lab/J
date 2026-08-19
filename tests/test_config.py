import pytest

from j_ai.config import load_config


def test_load_config(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model_name: example/model\nmax_length: 256\n", encoding="utf-8")
    config = load_config(path)
    assert config.model_name == "example/model"
    assert config.max_length == 256


def test_unknown_config_key(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("surprise: true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
