# explorer/__init__.py
"""
Enhanced Document Explorer Package

A comprehensive text analysis and visualization toolkit for exploring
document collections with support for multiple file formats and 
advanced NLP techniques.
"""

from .data import DocumentProcessor, load_corpus, compute_term_freq
from .analysis import TextAnalyzer
from .viz import (
    create_term_frequency_chart,
    create_wordcloud,
    create_ngram_chart,
    create_document_stats,
    create_sentiment_chart,
    create_topic_visualization,
    create_similarity_heatmap,
    plot_term_frequency,
    plot_wordcloud
)

__version__ = "2.0.0"
__author__ = "Enhanced Document Explorer Team"

__all__ = [
    # Data processing
    'DocumentProcessor',
    'load_corpus',
    'compute_term_freq',
    
    # Analysis
    'TextAnalyzer',
    
    # Visualization
    'create_term_frequency_chart',
    'create_wordcloud',
    'create_ngram_chart',
    'create_document_stats',
    'create_sentiment_chart',
    'create_topic_visualization',
    'create_similarity_heatmap',
    'plot_term_frequency',
    'plot_wordcloud'
]