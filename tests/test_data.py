# tests/test_data.py

import pytest
import pandas as pd
import io
import json
import zipfile
from explorer.data import load_corpus, compute_term_freq

def test_load_corpus_csv_stringio():
    csv_data = """id,text,category
1,"First text.",A
2,"Second text.",B
"""
    buf = io.StringIO(csv_data)
    buf.name = "dummy.csv"
    df = load_corpus(buf)
    assert isinstance(df, pd.DataFrame)
    assert list(df["text"]) == ["First text.", "Second text."]

def test_load_corpus_json_stringio():
    records = [
        {"id": 1, "text": "JSON text one", "cat": "X"},
        {"id": 2, "text": "JSON text two", "cat": "Y"}
    ]
    json_str = json.dumps(records)
    buf = io.StringIO(json_str)
    buf.name = "dummy.json"
    df = load_corpus(buf)
    assert isinstance(df, pd.DataFrame)
    assert list(df["text"]) == ["JSON text one", "JSON text two"]

def test_load_corpus_path(tmp_path):
    file = tmp_path / "on_disk.csv"
    file.write_text("text\nDisk one\nnDisk two")
    df = load_corpus(str(file))
    assert list(df["text"]) == ["Disk one", "nDisk two"]

def test_load_corpus_unsupported_extension():
    buf = io.StringIO("hello")
    buf.name = "data.xml"
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_corpus(buf)

def test_load_corpus_missing_text_column():
    buf = io.StringIO("a,b\n1,2")
    buf.name = "no_text.csv"
    with pytest.raises(ValueError, match="must contain a 'text' column"):
        load_corpus(buf)

def test_load_corpus_zip_stringio(tmp_path):
    # create an in-memory ZIP with two txt files
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w") as z:
        z.writestr("a.txt", "Alpha content")
        z.writestr("b.txt", "Beta content")
    mem_zip.name = "dummy.zip"
    mem_zip.seek(0)
    df = load_corpus(mem_zip)
    assert isinstance(df, pd.DataFrame)
    assert set(df["text"]) == {"Alpha content", "Beta content"}
    assert list(df["source"]) == ["a.txt", "b.txt"]

def test_load_corpus_zip_no_txt(tmp_path):
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, mode="w") as z:
        z.writestr("a.md", "Alpha")
    mem_zip.name = "no_txt.zip"
    mem_zip.seek(0)
    with pytest.raises(ValueError, match="No .txt files found"):
        load_corpus(mem_zip)

def test_compute_term_freq_simple():
    df = pd.DataFrame({
        "text": [
            "apple banana apple",
            "banana banana orange"
        ]
    })
    freq = compute_term_freq(df, top_n=2)
    # Expect banana (3), apple (2)
    assert list(freq["term"]) == ["banana", "apple"]
    assert list(freq["count"]) == [3, 2]

def test_compute_term_freq_missing_column():
    df = pd.DataFrame({"no_text": ["foo"]})
    with pytest.raises(ValueError, match="must contain a 'text' column"):
        compute_term_freq(df)

