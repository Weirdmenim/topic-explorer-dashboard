# explorer/viz.py
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import seaborn as sns

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def create_term_frequency_chart(freq_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """
    Create an interactive bar chart of term frequencies using Plotly.
    
    Args:
        freq_df: DataFrame with 'term' and 'count' columns
        top_n: Number of top terms to display
        
    Returns:
        Plotly figure object
    """
    if freq_df.empty:
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    data = freq_df.head(top_n).copy()
    
    # Create bar chart
    fig = px.bar(
        data,
        x='count',
        y='term',
        orientation='h',
        title=f'Top {min(top_n, len(data))} Term Frequencies',
        labels={'count': 'Frequency', 'term': 'Terms'},
        color='count',
        color_continuous_scale='viridis'
    )
    
    # Update layout
    fig.update_layout(
        height=max(400, len(data) * 25),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title="Frequency",
        yaxis_title="Terms",
        title_x=0.5,
        font=dict(size=12)
    )
    
    # Add value labels on bars
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )
    
    return fig

def create_wordcloud(freq_df: pd.DataFrame, width: int = 800, height: int = 400) -> plt.Figure:
    """
    Generate a word cloud from term frequencies.
    
    Args:
        freq_df: DataFrame with 'term' and 'count' columns
        width: Width of the word cloud
        height: Height of the word cloud
        
    Returns:
        Matplotlib figure object
    """
    if freq_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, color='gray')
        ax.axis('off')
        return fig
    
    # Create frequency dictionary
    frequencies = dict(zip(freq_df['term'], freq_df['count']))
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap='viridis',
        max_words=100,
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(frequencies)
    
    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title('Word Cloud', fontsize=16, pad=20)
    
    plt.tight_layout()
    return fig

def create_ngram_chart(ngram_df: pd.DataFrame, ngram_range: Tuple[int, int], top_n: int = 20) -> go.Figure:
    """
    Create a chart for n-gram frequencies.
    
    Args:
        ngram_df: DataFrame with n-gram data
        ngram_range: Tuple indicating n-gram range
        top_n: Number of top n-grams to display
        
    Returns:
        Plotly figure object
    """
    if ngram_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No n-gram data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="gray")
        )
        return fig
    
    data = ngram_df.head(top_n).copy()
    gram_type = f"{ngram_range[0]}-{ngram_range[1]} grams" if ngram_range[0] != ngram_range[1] else f"{ngram_range[0]}-grams"
    
    fig = px.bar(
        data,
        x='count',
        y='ngram',
        orientation='h',
        title=f'Top {min(top_n, len(data))} {gram_type}',
        labels={'count': 'Frequency', 'ngram': 'N-grams'},
        color='count',
        color_continuous_scale='plasma'
    )
    
    fig.update_layout(
        height=max(400, len(data) * 30),
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        title_x=0.5
    )
    
    fig.update_traces(
        texttemplate='%{x}',
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>'
    )
    
    return fig

def create_document_stats(stats: Dict) -> go.Figure:
    """
    Create a dashboard of document statistics.
    
    Args:
        stats: Dictionary containing document statistics
        
    Returns:
        Plotly figure with subplots
    """
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Document Length Distribution', 'Word Count Stats', 
                       'Vocabulary Statistics', 'Most Common Words'),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Document length distribution (placeholder - would need actual data)
    fig.add_trace(
        go.Bar(
            x=['Total Docs', 'Avg Words/Doc', 'Unique Words'],
            y=[stats.get('total_documents', 0), 
               stats.get('avg_words_per_doc', 0),
               stats.get('unique_words', 0)],
            name='Basic Stats'
        ),
        row=1, col=1
    )
    
    # Most common words
    if 'most_common_words' in stats and stats['most_common_words']:
        words, counts = zip(*stats['most_common_words'][:10])
        fig.add_trace(
            go.Bar(
                x=list(counts),
                y=list(words),
                orientation='h',
                name='Common Words'
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Document Analysis Dashboard",
        title_x=0.5
    )
    
    return fig

def create_sentiment_chart(sentiment_df: pd.DataFrame) -> go.Figure:
    """
    Create sentiment analysis visualization.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        
    Returns:
        Plotly figure object
    """
    if 'sentiment_label' not in sentiment_df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="No sentiment data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Count sentiment labels
    sentiment_counts = sentiment_df['sentiment_label'].value_counts()
    
    # Create pie chart
    fig = px.pie(
        values=sentiment_counts.values,
        names=sentiment_counts.index,
        title='Sentiment Distribution',
        color_discrete_map={
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'neutral': '#808080'
        }
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )
    
    fig.update_layout(
        title_x=0.5,
        font=dict(size=12)
    )
    
    return fig

def create_topic_visualization(topic_data: Dict) -> go.Figure:
    """
    Create topic modeling visualization.
    
    Args:
        topic_data: Dictionary containing topic modeling results
        
    Returns:
        Plotly figure object
    """
    if 'topics' not in topic_data or not topic_data['topics']:
        fig = go.Figure()
        fig.add_annotation(
            text="No topic data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    topics = topic_data['topics']
    
    # Create horizontal bar chart for each topic
    fig = make_subplots(
        rows=len(topics),
        cols=1,
        subplot_titles=[f"Topic {i+1}" for i in range(len(topics))],
        vertical_spacing=0.1
    )
    
    for i, topic in enumerate(topics):
        words = topic['words'][:8]  # Top 8 words per topic
        weights = topic['weights'][:8]
        
        fig.add_trace(
            go.Bar(
                x=weights,
                y=words,
                orientation='h',
                name=f'Topic {i+1}',
                showlegend=False
            ),
            row=i+1, col=1
        )
    
    fig.update_layout(
        height=200 * len(topics),
        title_text="Topic Modeling Results",
        title_x=0.5
    )
    
    return fig

def create_similarity_heatmap(similarity_df: pd.DataFrame, max_docs: int = 20) -> go.Figure:
    """
    Create document similarity heatmap.
    
    Args:
        similarity_df: DataFrame with similarity matrix
        max_docs: Maximum number of documents to display
        
    Returns:
        Plotly figure object
    """
    if similarity_df.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No similarity data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )
        return fig
    
    # Limit to max_docs for readability
    if len(similarity_df) > max_docs:
        similarity_df = similarity_df.iloc[:max_docs, :max_docs]
    
    fig = px.imshow(
        similarity_df.values,
        title='Document Similarity Matrix',
        labels=dict(color="Similarity"),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        title_x=0.5,
        width=600,
        height=600
    )
    
    return fig

def plot_term_frequency(freq_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """
    Legacy function for matplotlib bar chart (backward compatibility).
    
    Args:
        freq_df: DataFrame with 'term' and 'count' columns
        top_n: Number of top terms to plot
        
    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    data = freq_df.head(top_n)
    
    if data.empty:
        ax.text(0.5, 0.5, 'No data available', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, color='gray')
        return fig
    
    bars = ax.barh(data['term'], data['count'])
    
    # Color bars with gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(data)))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax.set_xlabel('Count', fontsize=12)
    ax.set_ylabel('Term', fontsize=12)
    ax.set_title(f'Top {len(data)} Term Frequencies', fontsize=14, pad=20)
    
    # Add value labels on bars
    for i, (term, count) in enumerate(zip(data['term'], data['count'])):
        ax.text(count + max(data['count']) * 0.01, i, str(count), 
                va='center', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_wordcloud(freq_df: pd.DataFrame) -> plt.Figure:
    """
    Legacy function for matplotlib word cloud (backward compatibility).
    
    Args:
        freq_df: DataFrame with 'term' and 'count' columns
        
    Returns:
        Matplotlib figure object
    """
    return create_wordcloud(freq_df)