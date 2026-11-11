"""Tests for fine_tune_prep — validation, stats, split, conversion.
All stdlib, no network, no model."""

import json

import pytest

import fine_tune_prep as ftp


def ex(*roles_content):
    return {"messages": [{"role": r, "content": c} for r, c in roles_content]}


GOOD = ex(("system", "be nice"), ("user", "hi"), ("assistant", "hello"))
VALID_PATH = None  # set in fixture


@pytest.fixture
def good_file(tmp_path):
    p = tmp_path / "good.jsonl"
    p.write_text(json.dumps(GOOD) + "\n", encoding="utf-8")
    return p


def test_example_errors_clean():
    assert ftp.example_errors(GOOD) == []


def test_example_errors_each_failure():
    assert ftp.example_errors("nope") == ["example is not a JSON object"]
    assert ftp.example_errors({}) == ["missing a non-empty 'messages' list"]
    errs = ftp.example_errors({"messages": [{"role": "wizard", "content": "x"}]})
    assert any("role 'wizard'" in e for e in errs)
    errs = ftp.example_errors({"messages": [{"role": "user", "content": "   "}]})
    assert any("empty or non-text" in e for e in errs)


def test_validate_ok_exit_zero(good_file, capsys):
    assert ftp.cmd_validate(type("A", (), {"path": str(good_file)})) == 0
    assert "1/1 examples pass" in capsys.readouterr().out


def test_validate_reports_line_numbers(tmp_path, capsys):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        json.dumps(GOOD) + "\n"
        "{not json\n"
        + json.dumps({"messages": [{"role": "wizard", "content": "x"}]}) + "\n",
        encoding="utf-8",
    )
    rc = ftp.cmd_validate(type("A", (), {"path": str(p)}))
    assert rc == 1
    out = capsys.readouterr().out
    assert "line 2: invalid JSON" in out
    assert "line 3: message 0: role 'wizard'" in out
    assert "1/3 examples pass (2 bad)" in out


def test_stats_counts(good_file, capsys):
    ftp.cmd_stats(type("A", (), {"path": str(good_file)}))
    out = capsys.readouterr().out
    assert "valid examples      1" in out
    assert "system messages     1" in out
    assert "duplicate prompts   0" in out


def test_stats_flags_duplicates(tmp_path):
    p = tmp_path / "d.jsonl"
    a = ex(("user", "same question"), ("assistant", "answer one"))
    b = ex(("user", "same question"), ("assistant", "answer two"))
    p.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n", encoding="utf-8")
    import io, contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ftp.cmd_stats(type("A", (), {"path": str(p)}))
    assert "duplicate prompts   1" in buf.getvalue()


def test_split_deterministic_and_sized(tmp_path):
    src = tmp_path / "src.jsonl"
    lines = [json.dumps(ex(("user", f"q{i}"), ("assistant", f"a{i}"))) for i in range(10)]
    src.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args = lambda tr, vl: type("A", (), {"source": str(src), "train": tr, "val": vl, "ratio": 0.7, "seed": 7})

    ftp.cmd_split(args(str(tmp_path / "t1.jsonl"), str(tmp_path / "v1.jsonl")))
    t1 = (tmp_path / "t1.jsonl").read_text().splitlines()
    v1 = (tmp_path / "v1.jsonl").read_text().splitlines()
    assert len(t1) == 7 and len(v1) == 3
    assert set(t1) | set(v1) == set(lines)          # nothing lost or duplicated

    ftp.cmd_split(args(str(tmp_path / "t2.jsonl"), str(tmp_path / "v2.jsonl")))
    assert (tmp_path / "t2.jsonl").read_text() == (tmp_path / "t1.jsonl").read_text()  # same seed


def test_convert_csv_with_quoted_commas(tmp_path, capsys):
    src = tmp_path / "qa.csv"
    src.write_text('q,a\n"what is 2+2, really","four, obviously"\n', encoding="utf-8")
    dest = tmp_path / "out.jsonl"
    ftp.cmd_convert(type("A", (), {"source": str(src), "dest": str(dest)}))
    row = json.loads(dest.read_text().splitlines()[0])
    assert row["messages"][0]["role"] == "user"
    assert row["messages"][0]["content"] == "what is 2+2, really"
    assert row["messages"][1]["content"] == "four, obviously"


def test_convert_with_system_column(tmp_path):
    src = tmp_path / "qa.csv"
    src.write_text("q,a,system\nhi,yo,be brief\n", encoding="utf-8")
    dest = tmp_path / "out.jsonl"
    ftp.cmd_convert(type("A", (), {"source": str(src), "dest": str(dest)}))
    msgs = json.loads(dest.read_text())["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]


def test_main_dispatches(tmp_path, capsys):
    good = tmp_path / "g.jsonl"
    good.write_text(json.dumps(GOOD) + "\n", encoding="utf-8")
    assert ftp.main(["validate", str(good)]) == 0
    assert ftp.main(["stats", str(good)]) == 0
