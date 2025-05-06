# explorer/viz.py
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd


def plot_term_frequency(freq_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """
    Plot a bar chart of term frequencies.

    Args:
        freq_df (pd.DataFrame): DataFrame with 'term' and 'count' columns.
        top_n (int): Number of top terms to plot (uses freq_df head).

    Returns:
        matplotlib.figure.Figure: The bar chart figure.
    """
    fig, ax = plt.subplots()
    data = freq_df.head(top_n)
    ax.bar(data['term'], data['count'])
    ax.set_xlabel('Term')
    ax.set_ylabel('Count')
    ax.set_title(f'Top {top_n} Term Frequencies')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig


def plot_wordcloud(freq_df: pd.DataFrame) -> plt.Figure:
    """
    Generate a word cloud from term frequencies.

    Args:
        freq_df (pd.DataFrame): DataFrame with 'term' and 'count' columns.

    Returns:
        matplotlib.figure.Figure: The word cloud figure.
    """
    # Create dictionary for WordCloud
    frequencies = dict(zip(freq_df['term'], freq_df['count']))
    wc = WordCloud(width=800, height=400)
    wc.generate_from_frequencies(frequencies)
    fig, ax = plt.subplots()
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    # explicitly remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    return fig
