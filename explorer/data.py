# explorer/data.py

import pandas as pd
import os
import io
import zipfile
from sklearn.feature_extraction.text import CountVectorizer

def load_corpus(path_or_buffer):
    """
    Load text data from a file path or buffer (like a file upload).

    Detects file type by extension (.csv, .json, .zip) and loads data into a pandas
    DataFrame, expecting a 'text' column (for CSV/JSON) or constructing one (for ZIP).

    Args:
        path_or_buffer (str or file-like object): Path to the file or a file-like
                                                  object with a .name attribute.

    Returns:
        pandas.DataFrame: A DataFrame containing the corpus data, with a 'text'
                          column (and 'source' for ZIP).

    Raises:
        TypeError: If input is not a str or file-like with .name.
        ValueError: If unsupported extension, missing 'text' column, or read errors.
        FileNotFoundError: If path does not exist (from pandas/zipfile).
    """
    # Determine "filename" for extension detection
    try:
        name = path_or_buffer.name
    except AttributeError:
        if isinstance(path_or_buffer, str):
            name = path_or_buffer
        else:
            raise TypeError("path_or_buffer must be a string path or a file-like with .name")
    if not name:
        raise ValueError("Could not determine filename or type from input")

    _, ext = os.path.splitext(name.lower())
    if ext == ".csv":
        try:
            df = pd.read_csv(path_or_buffer)
        except Exception as e:
            raise ValueError(f"Error reading CSV {name}: {e}")
    elif ext == ".json":
        try:
            df = pd.read_json(path_or_buffer)
        except Exception as e:
            raise ValueError(f"Error reading JSON {name}: {e}")
    elif ext == ".zip":
        # handle zip of .txt files
        # path_or_buffer may be path or file-like
        try:
            z = zipfile.ZipFile(path_or_buffer)
        except Exception as e:
            raise ValueError(f"Error opening ZIP {name}: {e}")

        records = []
        for member in z.namelist():
            if member.lower().endswith(".txt"):
                try:
                    content = z.read(member).decode("utf-8").strip()
                except Exception as e:
                    # skip unreadable files
                    continue
                records.append({"source": member, "text": content})
        if not records:
            raise ValueError(f"No .txt files found in ZIP {name}")
        df = pd.DataFrame(records)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Only .csv, .json, .zip supported.")

    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column.")

    return df


def compute_term_freq(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Compute the top N term frequencies across all documents in df['text'].

    Args:
        df (pd.DataFrame): DataFrame with at least a 'text' column.
        top_n (int): Number of top terms to return.

    Returns:
        pd.DataFrame: DataFrame with columns ['term', 'count'], sorted descending.
    """
    if "text" not in df.columns:
        raise ValueError("DataFrame must contain a 'text' column for term frequency.")

    texts = df["text"].astype(str).tolist()
    vec = CountVectorizer(stop_words="english")
    X = vec.fit_transform(texts)
    sums = X.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    freq_df = pd.DataFrame({"term": terms, "count": sums})
    freq_df = freq_df.sort_values(by="count", ascending=False).head(top_n).reset_index(drop=True)
    return freq_df
