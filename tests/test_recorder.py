import pytest

from mindreader import recorder
from mindreader.recorder import CONDITIONS, RecordSession, gate
from tests.fakes import Recognizer, Store, RawWriter, frames


def test_gate_rejects_short_takes():
    assert gate([]) is not None
    assert "short" in gate(frames(seconds=0.4)).lower()


def test_gate_rejects_still_lips():
    assert "barely moved" in gate(frames(motion=0.01))


def test_gate_passes_real_mouthing():
    assert gate(frames(seconds=2.0, motion=0.3)) is None


def test_session_requires_known_condition():
    with pytest.raises(ValueError):
        RecordSession(["hi"], "shouting", store=Store())
    assert CONDITIONS == ("silent", "whisper", "voiced")


def test_take_is_saved_with_condition_meta_and_raw_video():
    store, raw = Store(), RawWriter()
    s = RecordSession(["the quick brown fox", "jumps over"], "silent", store=store, raw=raw, camera="FaceTime", take=2)
    snap = s.handle(frames(seconds=2.0, fps=30))
    assert snap["status"] == "saved" and snap["saved"] == 1 and snap["idx"] == 1
    text, meta = store.added[0]
    assert text == "the quick brown fox"
    assert meta["condition"] == "silent" and meta["take"] == 2 and meta["camera"] == "FaceTime"
    assert meta["seconds"] == pytest.approx(59 / 30, abs=0.01) and meta["fps"] == pytest.approx(30, abs=0.5)
    assert raw.written == ["item1"]
    assert snap["prompt"] == "Jumps over." and snap["next"] is None


def test_rejected_take_keeps_the_prompt():
    store = Store()
    s = RecordSession(["hello there"], "silent", store=store)
    snap = s.handle(frames(seconds=0.3))
    assert snap["status"] == "retry" and snap["idx"] == 0 and store.added == []
    s2 = RecordSession(["hello there"], "silent", store=Store(lose_face=True))
    snap = s2.handle(frames())
    assert snap["status"] == "retry" and "face" in snap["message"].lower()


def test_read_back_scores_and_accuracy():
    rec = Recognizer(reads=["HELLO THERE", "GOOD BYE NOW"])
    s = RecordSession(["hello there", "good bye"], "whisper", store=Store(), recognizer=rec)
    s.handle(frames())
    assert s.last["read"] == "hello there" and s.last["errors"] == 0
    assert s.accuracy() == 1.0
    s.handle(frames())
    assert s.last["errors"] == 1 and s.last["words"] == 2  # one inserted word
    assert s.accuracy() == pytest.approx(1 - 1 / 4)
    assert s.done and rec.calls == 2


def test_read_back_failure_does_not_lose_the_take():
    class Boom:
        def recognize_clip(self, clip, rescore=True):
            raise RuntimeError("gpu on fire")
    store = Store()
    s = RecordSession(["hello"], "voiced", store=store, recognizer=Boom())
    snap = s.handle(frames())
    assert snap["status"] == "saved" and len(store.added) == 1
    assert "read-back failed" in s.last["read"] and s.accuracy() is None


def test_redo_deletes_last_take_and_rewinds():
    store = Store()
    s = RecordSession(["a b", "c d"], "silent", store=store, recognizer=Recognizer(["A B", "C D"]))
    s.handle(frames())
    s.handle(frames())
    assert s.done
    s.redo()
    assert store.deleted == ["item2"] and s.idx == 1 and len(s.scores) == 1 and s.last is None
    s.skip()
    assert s.done


def test_counts_and_condition_of(fresh_data):
    import numpy as np
    from silentdict.dataset import add_item, load_items
    clip = np.zeros((10, 96, 96), np.uint8)
    for i in range(6):
        add_item(clip, f"sentence {i}", "mindreader", meta={"condition": "silent"})
    add_item(clip, "engine made this", "calibration")
    c = recorder.counts()
    assert c["silent"] == {"train": 5, "val": 1}   # every 5th item is held out
    assert c["calibration"] == {"train": 1}
    assert recorder.next_take("sentence 1", "silent") == 2
    assert recorder.next_take("sentence 1", "voiced") == 1
    assert [recorder.condition_of(it) for it in load_items()][-1] == "calibration"


def test_prompts_for_other_conditions_reuse_silent_sentences(fresh_data):
    import numpy as np
    from silentdict.dataset import add_item
    clip = np.zeros((10, 96, 96), np.uint8)
    add_item(clip, "zeta first", "mindreader", meta={"condition": "silent"})
    add_item(clip, "zeta second", "mindreader", meta={"condition": "silent"})
    add_item(clip, "zeta first", "mindreader", meta={"condition": "voiced"})
    assert recorder.prompts_for_condition(10, "voiced") == ["ZETA SECOND"]
    assert "ZETA FIRST" in recorder.prompts_for_condition(10, "whisper")
    fresh = recorder.prompts_for_condition(5, "silent")
    assert len(fresh) == 5 and "ZETA FIRST" not in [p.upper() for p in fresh]
