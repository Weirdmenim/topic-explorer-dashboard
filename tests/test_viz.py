# tests/test_viz.py

import pytest
import pandas as pd
from matplotlib.figure import Figure
from explorer.viz import plot_term_frequency, plot_wordcloud

@pytest.fixture
def sample_freq_df():
    # Simple DataFrame with term counts
    return pd.DataFrame({
        "term": ["apple", "banana", "cherry"],
        "count": [5, 3, 1]
    })

def test_plot_term_frequency_returns_figure(sample_freq_df):
    fig = plot_term_frequency(sample_freq_df, top_n=3)
    assert isinstance(fig, Figure)
    # Optionally, check axes labels
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Term"
    assert ax.get_ylabel() == "Count"
    assert "Top 3 Term Frequencies" in ax.get_title()

def test_plot_wordcloud_returns_figure(sample_freq_df):
    fig = plot_wordcloud(sample_freq_df)
    assert isinstance(fig, Figure)
    ax = fig.axes[0]
    # WordCloud images have no axes ticks
    assert not ax.get_xticks().any()
    assert not ax.get_yticks().any()
