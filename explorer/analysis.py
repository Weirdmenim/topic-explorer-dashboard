# explorer/analysis.py - Complete Enhanced Version
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree
import string
import re
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from functools import lru_cache

# Download required NLTK data with error handling
def download_nltk_data():
    """Download required NLTK data with proper error handling."""
    nltk_downloads = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words']
    for item in nltk_downloads:
        try:
            nltk.download(item, quiet=True)
        except Exception as e:
            logging.warning(f"Could not download NLTK data '{item}': {e}")

download_nltk_data()

logger = logging.getLogger(__name__)

@dataclass
class AnalysisResults:
    """Container for analysis results with metadata."""
    term_frequencies: pd.DataFrame
    tfidf_scores: pd.DataFrame
    ngram_frequencies: pd.DataFrame
    document_stats: Dict
    sentiment_scores: Optional[pd.DataFrame] = None
    topics: Optional[Dict] = None
    similarity_matrix: Optional[pd.DataFrame] = None
    named_entities: Optional[List[Dict]] = None
    clusters: Optional[pd.DataFrame] = None

class TextAnalyzer:
    """Advanced text analysis with caching and parallel processing."""
    
    def __init__(self,
                 remove_stopwords: bool = True,
                 remove_punctuation: bool = True,
                 lowercase: bool = True,
                 remove_numbers: bool = False,
                 stem_words: bool = False,
                 lemmatize_words: bool = False,
                 language: str = 'english',
                 min_word_length: int = 2,
                 ngram_range: Tuple[int, int] = (1, 1),
                 max_features: Optional[int] = None,
                 use_parallel: bool = True,
                 cache_size: int = 128):
        """
        Initialize TextAnalyzer with enhanced options.
        
        Args:
            remove_stopwords: Remove common stopwords
            remove_punctuation: Remove punctuation marks
            lowercase: Convert text to lowercase
            remove_numbers: Remove numeric characters
            stem_words: Apply stemming to words
            lemmatize_words: Apply lemmatization to words
            language: Language for stopwords
            min_word_length: Minimum word length to keep
            ngram_range: Range for n-gram extraction
            max_features: Maximum number of features to extract
            use_parallel: Enable parallel processing
            cache_size: Size of preprocessing cache
        """
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_numbers = remove_numbers
        self.stem_words = stem_words
        self.lemmatize_words = lemmatize_words
        self.language = language
        self.min_word_length = min_word_length
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.use_parallel = use_parallel
        
        # Initialize NLTK components with error handling
        try:
            self.stop_words = set(stopwords.words(language)) if remove_stopwords else set()
        except Exception as e:
            self.stop_words = set()
            logger.warning(f"Could not load stopwords for {language}: {e}")
        
        # Initialize stemmer and lemmatizer
        self.stemmer = PorterStemmer() if stem_words else None
        self.lemmatizer = WordNetLemmatizer() if lemmatize_words else None
        
        # Setup caching
        self.preprocess_text = lru_cache(maxsize=cache_size)(self._preprocess_text_uncached)
    
    def _preprocess_text_uncached(self, text: str) -> str:
        """Internal preprocessing method without caching."""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize with fallback
        try:
            tokens = word_tokenize(text)
        except Exception:
            tokens = text.split()
        
        # Filter and process tokens
        filtered_tokens = []
        for token in tokens:
            # Skip if too short
            if len(token) < self.min_word_length:
                continue
            
            # Skip stopwords
            if self.remove_stopwords and token in self.stop_words:
                continue
            
            # Apply stemming
            if self.stemmer:
                try:
                    token = self.stemmer.stem(token)
                except Exception:
                    pass
            
            # Apply lemmatization
            if self.lemmatizer:
                try:
                    token = self.lemmatizer.lemmatize(token)
                except Exception:
                    pass
            
            filtered_tokens.append(token)
        
        return ' '.join(filtered_tokens)
    
    def compute_comprehensive_analysis(self, df: pd.DataFrame, top_n: int = 20) -> AnalysisResults:
        """
        Perform comprehensive text analysis including all features.
        
        Args:
            df: DataFrame with 'text' column
            top_n: Number of top terms to return
            
        Returns:
            AnalysisResults object containing all analysis results
        """
        logger.info("Starting comprehensive text analysis...")
        
        # Basic analysis
        term_freq = self.compute_term_frequency(df, top_n)
        tfidf_scores = self.compute_tfidf(df, top_n)
        ngram_freq = self.compute_ngram_frequency(df, top_n)
        doc_stats = self.compute_document_statistics(df)
        
        # Advanced analysis
        sentiment_scores = self.compute_sentiment_analysis(df)
        topics = self.compute_topic_modeling(df, n_topics=5)
        similarity_matrix = self.compute_document_similarity(df)
        named_entities = self.extract_named_entities(df)
        clusters = self.compute_document_clusters(df, n_clusters=3)
        
        return AnalysisResults(
            term_frequencies=term_freq,
            tfidf_scores=tfidf_scores,
            ngram_frequencies=ngram_freq,
            document_stats=doc_stats,
            sentiment_scores=sentiment_scores,
            topics=topics,
            similarity_matrix=similarity_matrix,
            named_entities=named_entities,
            clusters=clusters
        )
    
    def compute_sentiment_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute sentiment analysis for documents.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with sentiment scores
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        sentiments = []
        
        def analyze_sentiment(text):
            try:
                blob = TextBlob(str(text))
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                
                # Classify sentiment
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return {
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'sentiment_label': label
                }
            except Exception as e:
                logger.warning(f"Error analyzing sentiment: {e}")
                return {
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'sentiment_label': 'neutral'
                }
        
        if self.use_parallel and len(df) > 10:
            with ThreadPoolExecutor(max_workers=4) as executor:
                sentiment_futures = {executor.submit(analyze_sentiment, text): i 
                                   for i, text in enumerate(df['text'])}
                
                for future in as_completed(sentiment_futures):
                    idx = sentiment_futures[future]
                    result = future.result()
                    sentiments.append((idx, result))
            
            # Sort by original index
            sentiments.sort(key=lambda x: x[0])
            sentiments = [s[1] for s in sentiments]
        else:
            sentiments = [analyze_sentiment(text) for text in df['text']]
        
        return pd.DataFrame(sentiments)
    
    def compute_topic_modeling(self, df: pd.DataFrame, n_topics: int = 5) -> Dict:
        """
        Perform topic modeling using LDA.
        
        Args:
            df: DataFrame with 'text' column
            n_topics: Number of topics to extract
            
        Returns:
            Dictionary containing topic modeling results
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Preprocess texts
            texts = df['text'].apply(self.preprocess_text).tolist()
            texts = [t for t in texts if t.strip()]
            
            if len(texts) < n_topics:
                logger.warning(f"Not enough documents ({len(texts)}) for {n_topics} topics")
                return {'topics': [], 'document_topics': []}
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            doc_term_matrix = vectorizer.fit_transform(texts)
            
            # Perform LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=20
            )
            lda.fit(doc_term_matrix)
            
            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_weights = [topic[i] for i in top_words_idx]
                
                topics.append({
                    'topic_id': topic_idx,
                    'words': top_words,
                    'weights': top_weights
                })
            
            # Get document-topic probabilities
            doc_topics = lda.transform(doc_term_matrix)
            
            return {
                'topics': topics,
                'document_topics': doc_topics.tolist(),
                'perplexity': lda.perplexity(doc_term_matrix)
            }
            
        except Exception as e:
            logger.error(f"Error in topic modeling: {e}")
            return {'topics': [], 'document_topics': []}
    
    def compute_document_similarity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute document similarity matrix.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame containing similarity matrix
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Preprocess texts
            texts = df['text'].apply(self.preprocess_text).tolist()
            texts = [t for t in texts if t.strip()]
            
            if len(texts) < 2:
                return pd.DataFrame()
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Compute similarity matrix
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Create DataFrame
            doc_names = [f"Doc_{i}" for i in range(len(texts))]
            similarity_df = pd.DataFrame(
                similarity_matrix,
                index=doc_names,
                columns=doc_names
            )
            
            return similarity_df
            
        except Exception as e:
            logger.error(f"Error computing document similarity: {e}")
            return pd.DataFrame()
    
    def extract_named_entities(self, df: pd.DataFrame) -> List[Dict]:
        """
        Extract named entities from documents.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            List of dictionaries containing named entities
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        entities = []
        
        def extract_entities_from_text(text):
            try:
                # Tokenize and tag
                tokens = word_tokenize(str(text))
                pos_tags = pos_tag(tokens)
                
                # Named entity recognition
                ne_tree = ne_chunk(pos_tags, binary=False)
                
                doc_entities = []
                for subtree in ne_tree:
                    if isinstance(subtree, Tree):
                        entity_name = ' '.join([token for token, pos in subtree.leaves()])
                        entity_label = subtree.label()
                        doc_entities.append({
                            'entity': entity_name,
                            'label': entity_label
                        })
                
                return doc_entities
                
            except Exception as e:
                logger.warning(f"Error extracting entities: {e}")
                return []
        
        for idx, text in enumerate(df['text']):
            doc_entities = extract_entities_from_text(text)
            for entity in doc_entities:
                entity['document_id'] = idx
                entities.append(entity)
        
        return entities
    
    def compute_document_clusters(self, df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
        """
        Cluster documents using K-means.
        
        Args:
            df: DataFrame with 'text' column
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Preprocess texts
            texts = df['text'].apply(self.preprocess_text).tolist()
            texts = [t for t in texts if t.strip()]
            
            if len(texts) < n_clusters:
                logger.warning(f"Not enough documents for {n_clusters} clusters")
                return pd.DataFrame()
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Create results DataFrame
            cluster_df = pd.DataFrame({
                'document_id': range(len(texts)),
                'cluster': cluster_labels,
                'text_length': [len(text.split()) for text in texts]
            })
            
            return cluster_df
            
        except Exception as e:
            logger.error(f"Error in document clustering: {e}")
            return pd.DataFrame()
    
    def compute_document_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Compute comprehensive document statistics.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            Dictionary containing various statistics
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        texts = df['text'].astype(str)
        
        # Basic statistics
        word_counts = texts.apply(lambda x: len(x.split()))
        char_counts = texts.apply(len)
        
        # Sentence counts with error handling
        def count_sentences(text):
            try:
                return len(sent_tokenize(text))
            except Exception:
                return len(text.split('.'))
        
        sentence_counts = texts.apply(count_sentences)
        
        # Vocabulary statistics
        all_words = []
        for text in texts:
            words = self.preprocess_text(text).split()
            all_words.extend(words)
        
        vocab_counter = Counter(all_words)
        
        stats = {
            'total_documents': len(df),
            'total_words': word_counts.sum(),
            'total_characters': char_counts.sum(),
            'total_sentences': sentence_counts.sum(),
            'avg_words_per_doc': word_counts.mean(),
            'avg_chars_per_doc': char_counts.mean(),
            'avg_sentences_per_doc': sentence_counts.mean(),
            'unique_words': len(vocab_counter),
            'vocabulary_richness': len(vocab_counter) / max(len(all_words), 1),
            'most_common_words': vocab_counter.most_common(20),
            'longest_document': word_counts.max(),
            'shortest_document': word_counts.min(),
            'median_document_length': word_counts.median()
        }
        
        return stats
    
    def compute_term_frequency(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Enhanced term frequency computation with error handling."""
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Preprocess texts in parallel if enabled
            if self.use_parallel and len(df) > 50:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    processed_texts = list(executor.map(self.preprocess_text, df['text']))
            else:
                processed_texts = [self.preprocess_text(text) for text in df['text']]
            
            processed_texts = [t for t in processed_texts if t.strip()]
            
            if not processed_texts:
                return pd.DataFrame(columns=['term', 'count'])
            
            # Create vectorizer
            vectorizer = CountVectorizer(
                ngram_range=(1, 1),  # Only unigrams for term frequency
                max_features=self.max_features,
                token_pattern=r'\b\w+\b'
            )
            
            # Fit and transform
            X = vectorizer.fit_transform(processed_texts)
            
            # Get term frequencies
            feature_names = vectorizer.get_feature_names_out()
            frequencies = X.sum(axis=0).A1
            
            # Create DataFrame
            freq_df = pd.DataFrame({
                'term': feature_names,
                'count': frequencies
            })
            
            # Sort by frequency
            freq_df = freq_df.sort_values('count', ascending=False)
            
            return freq_df.head(top_n).reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error computing term frequencies: {str(e)}")
            return pd.DataFrame(columns=['term', 'count'])
    
    def compute_tfidf(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Compute TF-IDF scores for terms across documents.
        
        Args:
            df: DataFrame with 'text' column
            top_n: Number of top terms to return
            
        Returns:
            DataFrame with terms and their TF-IDF scores
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Preprocess texts
            if self.use_parallel and len(df) > 50:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    processed_texts = list(executor.map(self.preprocess_text, df['text']))
            else:
                processed_texts = [self.preprocess_text(text) for text in df['text']]
            
            processed_texts = [t for t in processed_texts if t.strip()]
            
            if not processed_texts:
                return pd.DataFrame(columns=['term', 'tfidf_score'])
            
            # Create TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 1),  # Only unigrams for TF-IDF
                max_features=self.max_features,
                token_pattern=r'\b\w+\b'
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            
            # Get feature names and compute mean TF-IDF scores
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = tfidf_matrix.mean(axis=0).A1
            
            # Create DataFrame
            tfidf_df = pd.DataFrame({
                'term': feature_names,
                'tfidf_score': mean_scores
            })
            
            # Sort by TF-IDF score
            tfidf_df = tfidf_df.sort_values('tfidf_score', ascending=False)
            
            return tfidf_df.head(top_n).reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error computing TF-IDF scores: {str(e)}")
            return pd.DataFrame(columns=['term', 'tfidf_score'])
    
    def compute_ngram_frequency(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """
        Compute n-gram frequencies from the text data.
        
        Args:
            df: DataFrame with 'text' column
            top_n: Number of top n-grams to return
            
        Returns:
            DataFrame with n-grams and their frequencies
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Preprocess texts
            if self.use_parallel and len(df) > 50:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    processed_texts = list(executor.map(self.preprocess_text, df['text']))
            else:
                processed_texts = [self.preprocess_text(text) for text in df['text']]
            
            processed_texts = [t for t in processed_texts if t.strip()]
            
            if not processed_texts:
                return pd.DataFrame(columns=['ngram', 'count'])
            
            # Create vectorizer with specified n-gram range
            vectorizer = CountVectorizer(
                ngram_range=self.ngram_range,
                max_features=self.max_features,
                token_pattern=r'\b\w+\b'
            )
            
            # Fit and transform
            X = vectorizer.fit_transform(processed_texts)
            
            # Get n-gram frequencies
            feature_names = vectorizer.get_feature_names_out()
            frequencies = X.sum(axis=0).A1
            
            # Create DataFrame
            ngram_df = pd.DataFrame({
                'ngram': feature_names,
                'count': frequencies
            })
            
            # Sort by frequency
            ngram_df = ngram_df.sort_values('count', ascending=False)
            
            return ngram_df.head(top_n).reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error computing n-gram frequencies: {str(e)}")
            return pd.DataFrame(columns=['ngram', 'count'])
    
    def get_vocabulary_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Get detailed vocabulary statistics.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            Dictionary with vocabulary statistics
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Collect all words
            all_words = []
            for text in df['text']:
                processed_text = self.preprocess_text(str(text))
                words = processed_text.split()
                all_words.extend(words)
            
            # Count occurrences
            word_counts = Counter(all_words)
            
            # Calculate statistics
            vocab_size = len(word_counts)
            total_words = sum(word_counts.values())
            
            # Frequency distribution
            freq_dist = list(word_counts.values())
            freq_dist.sort(reverse=True)
            
            # Calculate type-token ratio (vocabulary richness)
            ttr = vocab_size / max(total_words, 1)
            
            # Get hapax legomena (words that appear only once)
            hapax = sum(1 for count in word_counts.values() if count == 1)
            hapax_ratio = hapax / max(vocab_size, 1)
            
            return {
                'vocabulary_size': vocab_size,
                'total_words': total_words,
                'type_token_ratio': ttr,
                'hapax_legomena': hapax,
                'hapax_ratio': hapax_ratio,
                'most_frequent_words': word_counts.most_common(10),
                'frequency_distribution': freq_dist[:50]  # Top 50 frequencies
            }
            
        except Exception as e:
            logger.error(f"Error computing vocabulary statistics: {e}")
            return {}
    
    def analyze_document_complexity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze the complexity of documents based on various metrics.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with complexity metrics for each document
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        complexity_metrics = []
        
        for idx, text in enumerate(df['text']):
            try:
                text_str = str(text)
                processed_text = self.preprocess_text(text_str)
                
                # Basic metrics
                word_count = len(text_str.split())
                char_count = len(text_str)
                sentence_count = len(sent_tokenize(text_str))
                
                # Complexity metrics
                avg_word_length = np.mean([len(word) for word in text_str.split()]) if word_count > 0 else 0
                avg_sentence_length = word_count / max(sentence_count, 1)
                
                # Vocabulary richness for this document
                unique_words = len(set(processed_text.split()))
                vocabulary_richness = unique_words / max(word_count, 1)
                
                # Syllable estimation (simple heuristic)
                def count_syllables(word):
                    vowels = 'aeiouy'
                    count = 0
                    prev_char_vowel = False
                    for char in word.lower():
                        is_vowel = char in vowels
                        if is_vowel and not prev_char_vowel:
                            count += 1
                        prev_char_vowel = is_vowel
                    return max(1, count)
                
                total_syllables = sum(count_syllables(word) for word in text_str.split())
                avg_syllables_per_word = total_syllables / max(word_count, 1)
                
                # Flesch Reading Ease approximation
                if sentence_count > 0 and word_count > 0:
                    flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
                else:
                    flesch_score = 0
                
                complexity_metrics.append({
                    'document_id': idx,
                    'word_count': word_count,
                    'sentence_count': sentence_count,
                    'avg_word_length': avg_word_length,
                    'avg_sentence_length': avg_sentence_length,
                    'vocabulary_richness': vocabulary_richness,
                    'avg_syllables_per_word': avg_syllables_per_word,
                    'flesch_reading_ease': flesch_score,
                    'complexity_score': (avg_word_length + avg_sentence_length + avg_syllables_per_word) / 3
                })
                
            except Exception as e:
                logger.warning(f"Error analyzing document {idx} complexity: {e}")
                complexity_metrics.append({
                    'document_id': idx,
                    'word_count': 0,
                    'sentence_count': 0,
                    'avg_word_length': 0,
                    'avg_sentence_length': 0,
                    'vocabulary_richness': 0,
                    'avg_syllables_per_word': 0,
                    'flesch_reading_ease': 0,
                    'complexity_score': 0
                })
        
        return pd.DataFrame(complexity_metrics)
    
    def compute_readability_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute various readability metrics for documents.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with readability metrics
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        readability_metrics = []
        
        for idx, text in enumerate(df['text']):
            try:
                text_str = str(text)
                
                # Basic counts
                sentences = sent_tokenize(text_str)
                words = text_str.split()
                characters = len(text_str.replace(' ', ''))
                
                sentence_count = len(sentences)
                word_count = len(words)
                char_count = characters
                
                if sentence_count == 0 or word_count == 0:
                    readability_metrics.append({
                        'document_id': idx,
                        'flesch_kincaid_grade': 0,
                        'flesch_reading_ease': 0,
                        'gunning_fog_index': 0,
                        'automated_readability_index': 0,
                        'coleman_liau_index': 0,
                        'readability_consensus': 'Unable to calculate'
                    })
                    continue
                
                # Average sentence length
                avg_sentence_length = word_count / sentence_count
                
                # Count syllables in words
                def count_syllables_advanced(word):
                    word = word.lower().strip(string.punctuation)
                    if not word:
                        return 0
                    
                    vowels = 'aeiouy'
                    syllable_count = 0
                    prev_was_vowel = False
                    
                    for i, char in enumerate(word):
                        is_vowel = char in vowels
                        if is_vowel and not prev_was_vowel:
                            syllable_count += 1
                        prev_was_vowel = is_vowel
                    
                    # Handle silent 'e'
                    if word.endswith('e') and syllable_count > 1:
                        syllable_count -= 1
                    
                    return max(1, syllable_count)
                
                syllable_count = sum(count_syllables_advanced(word) for word in words)
                avg_syllables_per_word = syllable_count / word_count
                
                # Count complex words (3+ syllables)
                complex_words = sum(1 for word in words if count_syllables_advanced(word) >= 3)
                complex_word_percentage = (complex_words / word_count) * 100
                
                # Flesch Reading Ease
                flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
                
                # Flesch-Kincaid Grade Level
                flesch_kincaid_grade = (0.39 * avg_sentence_length) + (11.8 * avg_syllables_per_word) - 15.59
                
                # Gunning Fog Index
                gunning_fog = 0.4 * (avg_sentence_length + complex_word_percentage)
                
                # Automated Readability Index
                avg_chars_per_word = char_count / word_count
                ari = (4.71 * avg_chars_per_word) + (0.5 * avg_sentence_length) - 21.43
                
                # Coleman-Liau Index
                l = (char_count / word_count) * 100  # Average letters per 100 words
                s = (sentence_count / word_count) * 100  # Average sentences per 100 words
                coleman_liau = (0.0588 * l) - (0.296 * s) - 15.8
                
                # Readability consensus
                grade_levels = [flesch_kincaid_grade, gunning_fog, ari, coleman_liau]
                avg_grade = np.mean([g for g in grade_levels if not np.isnan(g)])
                
                if avg_grade <= 6:
                    consensus = 'Elementary School'
                elif avg_grade <= 9:
                    consensus = 'Middle School'
                elif avg_grade <= 13:
                    consensus = 'High School'
                elif avg_grade <= 16:
                    consensus = 'College Level'
                else:
                    consensus = 'Graduate Level'
                
                readability_metrics.append({
                    'document_id': idx,
                    'flesch_kincaid_grade': round(flesch_kincaid_grade, 2),
                    'flesch_reading_ease': round(flesch_reading_ease, 2),
                    'gunning_fog_index': round(gunning_fog, 2),
                    'automated_readability_index': round(ari, 2),
                    'coleman_liau_index': round(coleman_liau, 2),
                    'complex_word_percentage': round(complex_word_percentage, 2),
                    'avg_grade_level': round(avg_grade, 2),
                    'readability_consensus': consensus
                })
                
            except Exception as e:
                logger.warning(f"Error computing readability for document {idx}: {e}")
                readability_metrics.append({
                    'document_id': idx,
                    'flesch_kincaid_grade': 0,
                    'flesch_reading_ease': 0,
                    'gunning_fog_index': 0,
                    'automated_readability_index': 0,
                    'coleman_liau_index': 0,
                    'complex_word_percentage': 0,
                    'avg_grade_level': 0,
                    'readability_consensus': 'Error in calculation'
                })
        
        return pd.DataFrame(readability_metrics)
    
    def compute_linguistic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract advanced linguistic features from documents.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with linguistic features
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        linguistic_features = []
        
        for idx, text in enumerate(df['text']):
            try:
                text_str = str(text)
                
                # POS tagging
                try:
                    tokens = word_tokenize(text_str)
                    pos_tags = pos_tag(tokens)
                    pos_counts = Counter(tag for word, tag in pos_tags)
                    
                    # Calculate POS ratios
                    total_tokens = len(pos_tags)
                    noun_ratio = (pos_counts.get('NN', 0) + pos_counts.get('NNS', 0) + 
                                pos_counts.get('NNP', 0) + pos_counts.get('NNPS', 0)) / max(total_tokens, 1)
                    verb_ratio = (pos_counts.get('VB', 0) + pos_counts.get('VBD', 0) + 
                                pos_counts.get('VBG', 0) + pos_counts.get('VBN', 0) + 
                                pos_counts.get('VBP', 0) + pos_counts.get('VBZ', 0)) / max(total_tokens, 1)
                    adj_ratio = (pos_counts.get('JJ', 0) + pos_counts.get('JJR', 0) + 
                               pos_counts.get('JJS', 0)) / max(total_tokens, 1)
                    adv_ratio = (pos_counts.get('RB', 0) + pos_counts.get('RBR', 0) + 
                               pos_counts.get('RBS', 0)) / max(total_tokens, 1)
                    
                except Exception:
                    noun_ratio = verb_ratio = adj_ratio = adv_ratio = 0
                    pos_counts = Counter()
                
                # Punctuation analysis
                punct_count = sum(1 for char in text_str if char in string.punctuation)
                punct_ratio = punct_count / max(len(text_str), 1)
                
                # Question and exclamation ratios
                question_count = text_str.count('?')
                exclamation_count = text_str.count('!')
                question_ratio = question_count / max(len(sent_tokenize(text_str)), 1)
                exclamation_ratio = exclamation_count / max(len(sent_tokenize(text_str)), 1)
                
                # Capitalization features
                upper_count = sum(1 for char in text_str if char.isupper())
                cap_ratio = upper_count / max(len(text_str.replace(' ', '')), 1)
                
                # Sentence variety (by length)
                sentences = sent_tokenize(text_str)
                if sentences:
                    sent_lengths = [len(sent.split()) for sent in sentences]
                    sent_length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
                else:
                    sent_length_variance = 0
                
                # Function word ratio (approximation using common function words)
                function_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
                                'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
                                'do', 'at', 'this', 'but', 'his', 'by', 'from'}
                
                words_lower = [word.lower().strip(string.punctuation) for word in text_str.split()]
                function_word_count = sum(1 for word in words_lower if word in function_words)
                function_word_ratio = function_word_count / max(len(words_lower), 1)
                
                linguistic_features.append({
                    'document_id': idx,
                    'noun_ratio': round(noun_ratio, 4),
                    'verb_ratio': round(verb_ratio, 4),
                    'adjective_ratio': round(adj_ratio, 4),
                    'adverb_ratio': round(adv_ratio, 4),
                    'punctuation_ratio': round(punct_ratio, 4),
                    'question_ratio': round(question_ratio, 4),
                    'exclamation_ratio': round(exclamation_ratio, 4),
                    'capitalization_ratio': round(cap_ratio, 4),
                    'sentence_length_variance': round(sent_length_variance, 2),
                    'function_word_ratio': round(function_word_ratio, 4),
                    'lexical_diversity': round(len(set(words_lower)) / max(len(words_lower), 1), 4)
                })
                
            except Exception as e:
                logger.warning(f"Error computing linguistic features for document {idx}: {e}")
                linguistic_features.append({
                    'document_id': idx,
                    'noun_ratio': 0,
                    'verb_ratio': 0,
                    'adjective_ratio': 0,
                    'adverb_ratio': 0,
                    'punctuation_ratio': 0,
                    'question_ratio': 0,
                    'exclamation_ratio': 0,
                    'capitalization_ratio': 0,
                    'sentence_length_variance': 0,
                    'function_word_ratio': 0,
                    'lexical_diversity': 0
                })
        
        return pd.DataFrame(linguistic_features)
    
    def compute_keyword_extraction(self, df: pd.DataFrame, method: str = 'tfidf', top_n: int = 10) -> Dict:
        """
        Extract keywords using various methods.
        
        Args:
            df: DataFrame with 'text' column
            method: Extraction method ('tfidf', 'frequency', 'yake')
            top_n: Number of keywords to extract
            
        Returns:
            Dictionary with extracted keywords
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        try:
            # Combine all texts
            all_text = ' '.join(df['text'].astype(str))
            processed_text = self.preprocess_text(all_text)
            
            if method == 'tfidf':
                # TF-IDF based extraction
                vectorizer = TfidfVectorizer(
                    max_features=1000,
                    ngram_range=(1, 2),
                    stop_words='english' if self.language == 'english' else None
                )
                
                texts = [self.preprocess_text(str(text)) for text in df['text']]
                texts = [t for t in texts if t.strip()]
                
                if not texts:
                    return {'keywords': [], 'scores': [], 'method': method}
                
                tfidf_matrix = vectorizer.fit_transform(texts)
                feature_names = vectorizer.get_feature_names_out()
                mean_scores = tfidf_matrix.mean(axis=0).A1
                
                # Get top keywords
                top_indices = mean_scores.argsort()[-top_n:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                scores = [mean_scores[i] for i in top_indices]
                
            elif method == 'frequency':
                # Simple frequency based extraction
                words = processed_text.split()
                word_freq = Counter(words)
                most_common = word_freq.most_common(top_n)
                keywords = [word for word, count in most_common]
                scores = [count for word, count in most_common]
                
            else:
                # Default to frequency if method not recognized
                words = processed_text.split()
                word_freq = Counter(words)
                most_common = word_freq.most_common(top_n)
                keywords = [word for word, count in most_common]
                scores = [count for word, count in most_common]
            
            return {
                'keywords': keywords,
                'scores': scores,
                'method': method,
                'total_documents': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return {'keywords': [], 'scores': [], 'method': method, 'error': str(e)}
    
    def compute_text_quality_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute text quality and coherence metrics.
        
        Args:
            df: DataFrame with 'text' column
            
        Returns:
            DataFrame with quality metrics
        """
        if 'text' not in df.columns:
            raise ValueError("DataFrame must contain a 'text' column")
        
        quality_metrics = []
        
        for idx, text in enumerate(df['text']):
            try:
                text_str = str(text)
                
                # Basic quality indicators
                word_count = len(text_str.split())
                char_count = len(text_str)
                
                if word_count == 0:
                    quality_metrics.append({
                        'document_id': idx,
                        'completeness_score': 0,
                        'coherence_score': 0,
                        'information_density': 0,
                        'repetition_ratio': 0,
                        'average_word_length': 0,
                        'quality_score': 0
                    })
                    continue
                
                # Completeness (based on document length and structure)
                sentences = sent_tokenize(text_str)
                sentence_count = len(sentences)
                
                # Simple completeness heuristic
                if word_count < 10:
                    completeness_score = 0.1
                elif word_count < 50:
                    completeness_score = 0.5
                elif word_count < 100:
                    completeness_score = 0.7
                else:
                    completeness_score = 0.9
                
                # Add bonus for proper sentence structure
                if sentence_count > 0 and any(sent.strip().endswith('.') for sent in sentences):
                    completeness_score += 0.1
                
                completeness_score = min(1.0, completeness_score)
                
                # Coherence (simplified - based on sentence transitions and variety)
                if sentence_count <= 1:
                    coherence_score = 0.5
                else:
                    # Check for transition words and sentence variety
                    transition_words = {'however', 'therefore', 'furthermore', 'moreover', 
                                      'additionally', 'consequently', 'meanwhile', 'similarly',
                                      'in contrast', 'on the other hand', 'as a result'}
                    
                    text_lower = text_str.lower()
                    transition_count = sum(1 for word in transition_words if word in text_lower)
                    
                    # Sentence length variety
                    sent_lengths = [len(sent.split()) for sent in sentences]
                    length_variance = np.var(sent_lengths) if len(sent_lengths) > 1 else 0
                    
                    coherence_score = min(1.0, (transition_count / sentence_count) + 
                                        (length_variance / 100) + 0.3)
                
                # Information density (unique words / total words)
                words = text_str.lower().split()
                unique_words = len(set(words))
                information_density = unique_words / word_count
                
                # Repetition ratio (how much content is repeated)
                word_counts = Counter(words)
                repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
                repetition_ratio = repeated_words / word_count
                
                # Average word length
                avg_word_length = np.mean([len(word.strip(string.punctuation)) for word in words])
                
                # Overall quality score (weighted combination)
                quality_score = (
                    completeness_score * 0.3 +
                    coherence_score * 0.3 +
                    information_density * 0.2 +
                    (1 - repetition_ratio) * 0.1 +
                    min(avg_word_length / 6, 1.0) * 0.1  # Normalize avg word length
                )
                
                quality_metrics.append({
                    'document_id': idx,
                    'completeness_score': round(completeness_score, 3),
                    'coherence_score': round(coherence_score, 3),
                    'information_density': round(information_density, 3),
                    'repetition_ratio': round(repetition_ratio, 3),
                    'average_word_length': round(avg_word_length, 2),
                    'quality_score': round(quality_score, 3)
                })
                
            except Exception as e:
                logger.warning(f"Error computing quality metrics for document {idx}: {e}")
                quality_metrics.append({
                    'document_id': idx,
                    'completeness_score': 0,
                    'coherence_score': 0,
                    'information_density': 0,
                    'repetition_ratio': 0,
                    'average_word_length': 0,
                    'quality_score': 0
                })
        
        return pd.DataFrame(quality_metrics)
    
    def export_analysis_results(self, results: AnalysisResults, output_dir: str = 'output') -> Dict[str, str]:
        """
        Export analysis results to various formats.
        
        Args:
            results: AnalysisResults object
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths of exported results
        """
        import os
        
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            exported_files = {}
            
            # Export term frequencies
            if not results.term_frequencies.empty:
                term_freq_path = os.path.join(output_dir, 'term_frequencies.csv')
                results.term_frequencies.to_csv(term_freq_path, index=False)
                exported_files['term_frequencies'] = term_freq_path
            
            # Export TF-IDF scores
            if not results.tfidf_scores.empty:
                tfidf_path = os.path.join(output_dir, 'tfidf_scores.csv')
                results.tfidf_scores.to_csv(tfidf_path, index=False)
                exported_files['tfidf_scores'] = tfidf_path
            
            # Export n-gram frequencies
            if not results.ngram_frequencies.empty:
                ngram_path = os.path.join(output_dir, 'ngram_frequencies.csv')
                results.ngram_frequencies.to_csv(ngram_path, index=False)
                exported_files['ngram_frequencies'] = ngram_path
            
            # Export document statistics as JSON
            if results.document_stats:
                import json
                stats_path = os.path.join(output_dir, 'document_statistics.json')
                with open(stats_path, 'w') as f:
                    json.dump(results.document_stats, f, indent=2, default=str)
                exported_files['document_statistics'] = stats_path
            
            # Export sentiment analysis
            if results.sentiment_scores is not None and not results.sentiment_scores.empty:
                sentiment_path = os.path.join(output_dir, 'sentiment_analysis.csv')
                results.sentiment_scores.to_csv(sentiment_path, index=False)
                exported_files['sentiment_analysis'] = sentiment_path
            
            # Export similarity matrix
            if results.similarity_matrix is not None and not results.similarity_matrix.empty:
                similarity_path = os.path.join(output_dir, 'document_similarity.csv')
                results.similarity_matrix.to_csv(similarity_path)
                exported_files['document_similarity'] = similarity_path
            
            # Export topics
            if results.topics and results.topics.get('topics'):
                import json
                topics_path = os.path.join(output_dir, 'topic_modeling.json')
                with open(topics_path, 'w') as f:
                    json.dump(results.topics, f, indent=2, default=str)
                exported_files['topic_modeling'] = topics_path
            
            # Export named entities
            if results.named_entities:
                entities_path = os.path.join(output_dir, 'named_entities.csv')
                entities_df = pd.DataFrame(results.named_entities)
                entities_df.to_csv(entities_path, index=False)
                exported_files['named_entities'] = entities_path
            
            # Export clusters
            if results.clusters is not None and not results.clusters.empty:
                clusters_path = os.path.join(output_dir, 'document_clusters.csv')
                results.clusters.to_csv(clusters_path, index=False)
                exported_files['document_clusters'] = clusters_path
            
            logger.info(f"Analysis results exported to {output_dir}")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting analysis results: {e}")
            return {'error': str(e)}


# Utility functions for batch processing and advanced analysis

def batch_analyze_documents(file_paths: List[str], 
                          analyzer_config: Dict = None,
                          output_dir: str = 'batch_output') -> Dict:
    """
    Analyze multiple documents in batch mode.
    
    Args:
        file_paths: List of file paths to analyze
        analyzer_config: Configuration for TextAnalyzer
        output_dir: Directory for output files
        
    Returns:
        Dictionary with batch analysis results
    """
    from .data import DocumentProcessor
    
    if analyzer_config is None:
        analyzer_config = {}
    
    processor = DocumentProcessor()
    analyzer = TextAnalyzer(**analyzer_config)
    
    all_documents = []
    file_metadata = []
    
    # Process all files
    for file_path in file_paths:
        try:
            with open(file_path, 'rb') as f:
                f.name = file_path  # Required for DocumentProcessor
                documents = processor.load_document(f)
                
                for doc in documents:
                    doc['source_file'] = file_path
                    all_documents.append(doc)
                    
                file_metadata.append({
                    'file_path': file_path,
                    'document_count': len(documents),
                    'status': 'success'
                })
                
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            file_metadata.append({
                'file_path': file_path,
                'document_count': 0,
                'status': 'error',
                'error': str(e)
            })
    
    if not all_documents:
        return {'error': 'No documents were successfully processed'}
    
    # Create DataFrame and analyze
    df = pd.DataFrame(all_documents)
    results = analyzer.compute_comprehensive_analysis(df)
    
    # Export results
    exported_files = analyzer.export_analysis_results(results, output_dir)
    
    return {
        'total_documents': len(all_documents),
        'total_files': len(file_paths),
        'file_metadata': file_metadata,
        'analysis_results': results,
        'exported_files': exported_files
    }


def create_analysis_report(results: AnalysisResults, 
                         output_path: str = 'analysis_report.html') -> str:
    """
    Generate a comprehensive HTML report of analysis results.
    
    Args:
        results: AnalysisResults object
        output_path: Path for the output HTML file
        
    Returns:
        Path to the generated report
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Text Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-left: 4px solid #007acc; }}
            table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .chart-placeholder {{ background-color: #e0e0e0; padding: 20px; text-align: center; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Text Analysis Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Document Statistics</h2>
            <div class="metric">
                <h3>Overview</h3>
                <p><strong>Total Documents:</strong> {results.document_stats.get('total_documents', 'N/A')}</p>
                <p><strong>Total Words:</strong> {results.document_stats.get('total_words', 'N/A'):,}</p>
                <p><strong>Unique Words:</strong> {results.document_stats.get('unique_words', 'N/A'):,}</p>
                <p><strong>Vocabulary Richness:</strong> {results.document_stats.get('vocabulary_richness', 0):.3f}</p>
            </div>
        </div>
        
        <div class="section">
            <h2>Top Terms</h2>
            <table>
                <tr><th>Term</th><th>Frequency</th></tr>
    """
    
    # Add top terms table
    if not results.term_frequencies.empty:
        for _, row in results.term_frequencies.head(10).iterrows():
            html_content += f"<tr><td>{row['term']}</td><td>{row['count']}</td></tr>"
    
    html_content += """
            </table>
        </div>
        
        <div class="section">
            <h2>TF-IDF Scores</h2>
            <table>
                <tr><th>Term</th><th>TF-IDF Score</th></tr>
    """
    
    # Add TF-IDF table
    if not results.tfidf_scores.empty:
        for _, row in results.tfidf_scores.head(10).iterrows():
            html_content += f"<tr><td>{row['term']}</td><td>{row['tfidf_score']:.4f}</td></tr>"
    
    html_content += """
            </table>
        </div>
    """
    
    # Add sentiment analysis if available
    if results.sentiment_scores is not None and not results.sentiment_scores.empty:
        sentiment_counts = results.sentiment_scores['sentiment_label'].value_counts()
        html_content += """
        <div class="section">
            <h2>Sentiment Analysis</h2>
            <div class="metric">
        """
        for sentiment, count in sentiment_counts.items():
            html_content += f"<p><strong>{sentiment.capitalize()}:</strong> {count} documents</p>"
        html_content += "</div></div>"
    
    # Add topics if available
    if results.topics and results.topics.get('topics'):
        html_content += """
        <div class="section">
            <h2>Topic Modeling</h2>
        """
        for i, topic in enumerate(results.topics['topics'][:5]):
            html_content += f"""
            <div class="metric">
                <h3>Topic {i+1}</h3>
                <p><strong>Top Words:</strong> {', '.join(topic['words'][:10])}</p>
            </div>
            """
        html_content += "</div>"
    
    html_content += """
        </body>
    </html>
    """
    
    # Write HTML file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Analysis report generated: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return None