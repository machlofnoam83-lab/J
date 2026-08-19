from j_ai.model import ensure_model_downloaded


def test_local_model_does_not_need_network(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    assert ensure_model_downloaded(str(model_dir)) == str(model_dir)
