from syntony.decompose import SentenceDecomposer


def test_basic_split():
    d = SentenceDecomposer()
    claims = d.decompose("The sky is blue. Water is wet. Fire is hot.")
    assert claims == ["The sky is blue.", "Water is wet.", "Fire is hot."]


def test_empty_and_whitespace():
    d = SentenceDecomposer()
    assert d.decompose("") == []
    assert d.decompose("   \n  ") == []


def test_abbreviations_do_not_split():
    d = SentenceDecomposer()
    claims = d.decompose("The dose was 5mg, e.g. twice daily. It was safe.")
    assert claims == ["The dose was 5mg, e.g. twice daily.", "It was safe."]


def test_question_and_exclamation_boundaries():
    d = SentenceDecomposer()
    claims = d.decompose("Is it safe? Yes it is! That is good.")
    assert claims == ["Is it safe?", "Yes it is!", "That is good."]


def test_min_chars_filters_fragments():
    d = SentenceDecomposer(min_chars=5)
    claims = d.decompose("Ok. This is a real sentence.")
    assert claims == ["This is a real sentence."]


def test_trailing_text_without_period():
    d = SentenceDecomposer()
    claims = d.decompose("First sentence. Second without end")
    assert claims == ["First sentence.", "Second without end"]
