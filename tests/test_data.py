import json

import pytest

from j_ai.data import deduplicate, read_jsonl, validate_record
from j_ai.prepare import chunk_text


def test_read_and_deduplicate(tmp_path):
    record = {"messages": [{"role": "user", "content": "היי"}, {"role": "assistant", "content": "שלום"}]}
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join([json.dumps(record), json.dumps(record)]), encoding="utf-8")
    assert len(read_jsonl(path)) == 2
    assert deduplicate(read_jsonl(path)) == [record]


def test_rejects_chat_without_answer():
    with pytest.raises(ValueError, match="no assistant"):
        validate_record({"messages": [{"role": "user", "content": "hi"}]})


def test_chunks_plain_text():
    chunks = chunk_text("first paragraph\n\nsecond paragraph", 20)
    assert chunks == [{"text": "first paragraph"}, {"text": "second paragraph"}]
