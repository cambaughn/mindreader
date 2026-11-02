import pytest

from mindreader import evaluate as E


PAIRS = [("the cat sat", "the cat sat"), ("a dog ran", "a dog ran far"), ("hello world", "goodbye world"),
         ("one two three", "one two three")]


def test_exact_match():
    assert E.exact_match(PAIRS) == 0.5
    assert E.exact_match([]) == 0.0


def test_chance_is_worse_than_real_when_the_model_reads_lips():
    from silentdict.text import corpus_wer
    real = corpus_wer(PAIRS)
    chance = E.chance_wer(PAIRS)
    assert chance is not None and chance > real
    assert E.chance_wer(PAIRS[:1]) is None
    assert E.chance_wer(PAIRS, seed=1) == pytest.approx(E.chance_wer(PAIRS, seed=1))


def test_summarize_groups_by_condition():
    conditions = ["silent", "silent", "voiced", "voiced"]
    rep = E.summarize({"base": PAIRS, "personal": [(r, r) for r, _ in PAIRS]}, conditions)
    assert set(rep["conditions"]) == {"silent", "voiced", "all"}
    assert rep["conditions"]["silent"]["n"] == 2
    assert rep["conditions"]["all"]["models"]["personal"]["wer"] == 0.0
    assert rep["conditions"]["voiced"]["models"]["base"]["exact"] == 0.5
    table = E.format_table(rep)
    assert table.splitlines()[0].startswith("| condition | n | base WER")
    assert table.splitlines()[-1].startswith("| all | 4 |")


def test_train_filters_items_by_condition(monkeypatch):
    from silentdict import train as T
    from silentdict.dataset import Item
    items = [Item("1", "A", "train", "mindreader", 0.0, 10, {"condition": "silent"}),
             Item("2", "B", "train", "mindreader", 0.0, 10, {"condition": "voiced"}),
             Item("3", "C", "val", "calibration", 0.0, 10, {})]
    original = lambda split=None: [it for it in items if split is None or it.split == split]
    monkeypatch.setattr(T, "load_items", original)
    seen = {}

    def fake_run(log=print, progress=None, **kw):
        seen["train"] = [it.id for it in T.load_items("train")]
        seen["all"] = [it.id for it in T.load_items()]
        return {"ok": True}
    monkeypatch.setattr(T, "run", fake_run)
    assert E.train(condition="silent", log=lambda *_: None) == {"ok": True}
    assert seen == {"train": ["1"], "all": ["1"]}
    assert T.load_items is original  # restored
    E.train(log=lambda *_: None)
    assert seen["all"] == ["1", "2", "3"]
