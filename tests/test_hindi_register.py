"""C4b: workplace-Hindi register instruction and name-aware script gate."""

import pytest

from turnguard.data import teacher as t


def test_hindi_requests_carry_the_register_instruction():
    assert t.system_for("hi").startswith(t.DEFAULT_SYSTEM)
    assert t.HINDI_STYLE in t.system_for("hi")
    assert "मीटिंग" in t.system_for("hi") and "पारगमन" in t.system_for("hi")


@pytest.mark.parametrize("language", ("en", "hinglish"))
def test_other_languages_keep_the_unchanged_system_message(language):
    # Same text as before C4b, so their cache keys are untouched.
    assert t.system_for(language) == t.DEFAULT_SYSTEM
    assert t.HINDI_STYLE not in t.system_for(language)


def test_latin_train_name_no_longer_fails_the_devanagari_ratio():
    source = "Tell me the Deccan Queen Express time."
    hindi = "Deccan Queen Express का time बताएं।"
    with pytest.raises(ValueError, match="DEVANAGARI"):
        t.check_script(hindi, "hi", [])
    t.check_script(hindi, "hi", [], exempt=t.latin_names(source))


def test_threshold_still_rejects_hindi_written_in_latin_letters():
    source = "Search for the Deccan Queen timetable."
    with pytest.raises(ValueError, match="requires 35%"):
        t.check_script(
            "Deccan Queen ka timetable dhoondo aur batao.",
            "hi",
            [],
            exempt=t.latin_names(source),
        )


def test_request_uses_language_specific_system_message(monkeypatch, tmp_path):
    systems = []
    original = t.canonical

    def spy(value):
        if isinstance(value, dict) and "messages" in value:
            systems.append(value["messages"][0]["content"])
        return original(value)

    monkeypatch.setattr(t, "canonical", spy)
    client = t.Teacher(
        cache_dir=tmp_path,
        model="m",
        api_base="b",
        api_key="",
        sampling={"max_tokens": 50},
        template_version="v",
        mode="stub",
    )
    for language in ("hi", "hinglish"):
        client.generate({"gen": "note", "about": "a team note"}, language, [])
    assert systems == [t.system_for("hi"), t.DEFAULT_SYSTEM]


def test_code_mixed_workplace_hindi_passes_the_script_gate():
    t.check_script(
        "फिर पढ़ें और मुझे पहले action item का tracking number बताएं।",
        "hi",
        ["action_items.txt"],
    )


def test_bare_short_numbers_in_assets_are_not_keep_tokens():
    tokens = t.keep_tokens({}, {}, {"destroy_ref": "0 days", "file": "q3_payroll.xlsx"})
    assert "0" not in tokens and "q3_payroll.xlsx" in tokens
    assert "2026" in t.keep_tokens({}, {}, {"ref": "since 2026"})
