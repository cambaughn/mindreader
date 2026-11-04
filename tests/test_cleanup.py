from mindreader import cleanup


def test_messages_show_top_guess_or_list_with_context():
    cands = ["PLEASE REVIEW THE MOON REQUEST", "PLEASE REVIEW THE PULL REQUEST"]
    best = cleanup.build_messages(cands, ["GitHub"], mode="best")
    assert best[0]["role"] == "system" and "GitHub" in best[1]["content"]
    assert "Please review the moon request." in best[1]["content"] and "pull" not in best[1]["content"]
    nbest = cleanup.build_messages(cands, [], mode="nbest")
    assert "1. Please review the moon request." in nbest[1]["content"]
    assert "2. Please review the pull request." in nbest[1]["content"]


def test_first_sentence_strips_quotes_markers_and_extra_lines():
    assert cleanup.first_sentence('"Please review the pull request."\n\nExplanation: ...') == "Please review the pull request."
    assert cleanup.first_sentence("1. Sounds good, let's do that.") == "Sounds good, let's do that."
    assert cleanup.first_sentence("   \n") == ""


def test_rewrite_uses_the_model_reply_and_falls_back_to_top_candidate():
    c = cleanup.Cleaner.__new__(cleanup.Cleaner)
    c.context, c.max_tokens = ["GitHub"], 60
    seen = {}
    c.generate = lambda messages: seen.setdefault("m", messages) and "Please review the pull request when you have time."
    assert c.rewrite(["PLEASE REVIEW THE MOON REQUEST WITH YOU EVERY TIME"]) == "Please review the pull request when you have time."
    assert "GitHub" in seen["m"][1]["content"]
    c.generate = lambda messages: "\n"
    assert c.rewrite(["FALLBACK GUESS"]) == "FALLBACK GUESS"
    assert c.rewrite([]) == ""
