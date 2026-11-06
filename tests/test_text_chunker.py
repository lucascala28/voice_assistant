from text_chunker import Chunker, ThinkFilter, next_chunk, clean_for_speech


# Stream text token by token through a Chunker and return all spoken chunks.
def run(tokens):
    c = Chunker()
    out = []
    for t in tokens:
        out += c.feed(t)
    return out + c.flush()


def test_period_newline_token():
    assert run(["Hello there", ".\n", "Next one", "."]) == ["Hello there.", "Next one."]


def test_exclaim_double_newline_token():
    assert run(["Great", "!\n\n", "Done", "!"]) == ["Great!", "Done!"]


def test_quote_after_period():
    assert run(['He said "go', '."', " Then left", "."]) == ['He said "go."', "Then left."]


def test_markdown_after_period():
    assert run(["**Bold sentence.**", " Another", "."]) == ["Bold sentence.", "Another."]


def test_period_space_token():
    assert run(["One", ". ", "Two", ". ", "Three"]) == ["One.", "Two.", "Three"]


def test_decimal_not_split_across_tokens():
    assert run(["It is 72", ".", "5", " degrees", "."]) == ["It is 72.5 degrees."]


def test_decimal_in_one_token():
    chunk, rest = next_chunk("It is 72.5 degrees and rising. More", final=False)
    assert chunk == "It is 72.5 degrees and rising."
    assert rest == "More"


def test_trailing_period_waits_for_lookahead():
    assert next_chunk("Hello.", final=False) == ("", "Hello.")
    assert next_chunk("Hello.", final=True) == ("Hello.", "")


def test_abbreviations_do_not_split():
    assert run(["Ask Dr. Smith about it", ". ", "Ok", "."]) == ["Ask Dr. Smith about it.", "Ok."]
    assert run(["Use fruit, e.g. apples", ". ", "Bye", "."]) == ["Use fruit, e.g. apples.", "Bye."]
    assert run(["Mr. Jones is here", ".\n"]) == ["Mr. Jones is here."]


def test_ellipsis_and_combo_terminators():
    assert run(["Wait... what", "?! ", "Fine", "."]) == ["Wait...", "what?!", "Fine."]


def test_first_chunk_early_cut_at_comma():
    text = "Well, the weather today is quite nice, and it is warm outside. Enjoy"
    chunks = run([text])
    assert chunks[0].endswith("nice,")          # cut at first comma past ~25 chars
    assert chunks[1].endswith("outside.")
    assert chunks[2] == "Enjoy"


def test_early_cut_only_for_first_chunk():
    c = Chunker()
    c.feed("First sentence is short.")
    out = c.feed(" The second one has a comma, but no end yet and keeps going on and on ")
    assert out == ["First sentence is short."]   # no early cut after the first chunk


def test_first_chunk_early_cut_at_newline():
    chunks = run(["Here is a list of items for you\n", "- one\n", "- two\n"])
    assert chunks[0] == "Here is a list of items for you"
    assert chunks[1:] == ["one", "two"]


def test_no_early_cut_before_minimum_length():
    assert next_chunk("Hi, there friend and more", first=True) == ("", "Hi, there friend and more")


def test_remainder_flushed_at_end():
    c = Chunker()
    assert c.feed("No terminator here") == []
    assert c.flush() == ["No terminator here"]
    assert c.flush() == []


def test_markdown_stripped():
    assert clean_for_speech("## Title\n**bold** and `code`") == "Title bold and code"
    assert run(["- item one", ".\n", "- item two", ".\n"]) == ["item one.", "item two."]


def test_numbered_list_marker_not_a_sentence():
    assert run(["1. First item", ".\n", "2. Second item", ".\n"]) == ["1. First item.", "2. Second item."]


def test_markdown_only_chunk_is_dropped():
    assert run(["***\n", "Real text", "."]) == ["Real text."]


def test_line_breaks_in_one_burst_all_split():
    assert run(["Here you go.\n- apples\n- pears\n- milk\n"]) == ["Here you go.", "apples", "pears", "milk"]


def test_think_filter_split_tags():
    f = ThinkFilter()
    out = "".join(f.feed(t) for t in ["a <thi", "nk>hidden</", "think> b", " <not a tag"]) + f.flush()
    assert out == "a  b <not a tag"


def test_think_filter_unclosed_block_dropped():
    f = ThinkFilter()
    assert f.feed("ok <think>never ends") + f.flush() == "ok "


def test_think_filter_stray_closing_tag_drops_earlier_text():
    f = ThinkFilter()
    assert f.feed("reasoning without opener</think>Answer.") + f.flush() == "Answer."
    g = ThinkFilter()
    # known limit: text already emitted in an earlier token cannot be recalled, but the tag itself is removed
    assert "".join(g.feed(t) for t in ["hidden</thi", "nk>Visible"]) + g.flush() == "hiddenVisible"
