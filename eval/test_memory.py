from __future__ import annotations

import numpy as np

from models.prototype_memory import PrototypeMemory


def test_memory_updates_ranks_and_persists(tmp_path) -> None:
    state_path = tmp_path / "prototypes.json"
    memory = PrototypeMemory(state_path)
    memory.upsert("red", np.asarray([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0]]))
    memory.upsert("blue", np.asarray([[0.0, 0.0, 1.0]]))
    memory.save()

    restored = PrototypeMemory(state_path)
    matches = restored.match(np.asarray([1.0, 0.0, 0.0]), top_k=2)

    assert [match.label for match in matches] == ["red", "blue"]
    assert restored.classes()[1]["examples"] == 2


def test_memory_rejects_embedding_dimension_change() -> None:
    memory = PrototypeMemory()
    memory.upsert("first", np.asarray([[1.0, 0.0]]))

    try:
        memory.upsert("second", np.asarray([[1.0, 0.0, 0.0]]))
    except ValueError as exc:
        assert "dimension" in str(exc)
    else:
        raise AssertionError("Expected dimension mismatch to fail")
