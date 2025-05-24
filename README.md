# Enhanced Document Explorer 📚

A comprehensive text analysis and visualization toolkit for exploring document collections with support for multiple file formats and advanced NLP techniques.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🚀 Features

### 📁 Multi-Format Document Support
- **PDF Documents** - Extract text from PDF files with page-level processing
- **Word Documents** - Support for DOCX files with paragraph extraction
- **Text Files** - Plain text files with encoding detection
- **Structured Data** - CSV and JSON files with intelligent text column detection
- **ZIP Archives** - Process multiple files from compressed archives
- **Batch Processing** - Upload and analyze multiple documents simultaneously

### 🔍 Advanced Text Analysis
- **Term Frequency Analysis** - Identify most common terms with customizable parameters
- **N-gram Extraction** - Analyze word patterns from unigrams to trigrams
- **TF-IDF Scoring** - Discover important terms using Term Frequency-Inverse Document Frequency
- **Topic Modeling** - Automatic topic discovery using Latent Dirichlet Allocation (LDA)
- **Document Similarity** - Compare documents using cosine similarity
- **Sentiment Analysis** - Analyze emotional tone of documents

### 🛠️ Flexible Text Preprocessing
- **Stopword Removal** - Configurable stopwords for multiple languages
- **Punctuation Handling** - Optional punctuation removal
- **Case Normalization** - Lowercase conversion options
- **Stemming & Lemmatization** - Word normalization techniques
- **Custom Filtering** - Minimum word length and number removal
- **Language Support** - English, Spanish, French, German, Italian

### 📊 Rich Interactive Visualizations
- **Interactive Bar Charts** - Plotly-powered term frequency visualizations
- **Word Clouds** - Beautiful word cloud generation with customizable styling
- **Distribution Plots** - Document length and statistical distributions
- **Heatmaps** - Document similarity matrices
- **Topic Visualizations** - Topic modeling results with word weights
- **Sentiment Charts** - Pie charts and distribution plots for sentiment analysis

### 💻 User-Friendly Interface
- **Streamlit Web App** - Clean, intuitive web interface
- **Real-time Processing** - Live updates as you adjust parameters
- **Export Capabilities** - Download results as CSV files
- **Document Preview** - View and inspect individual documents
- **Responsive Design** - Works on desktop and mobile devices

## 🏗️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Install
```bash
# Clone the repository
git clone https://github.com/your-username/enhanced-document-explorer.git
cd enhanced-document-explorer

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (if not already present)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')"
```

### Requirements
```txt
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
nltk>=3.7
plotly>=5.0.0
matplotlib>=3.5.0
seaborn>=0.11.0
wordcloud>=1.8.0
PyPDF2>=3.0.0
python-docx>=0.8.11
```

## 🚀 Quick Start

### Running the Web Application
```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

### Using as a Python Library
```python
from explorer.data import DocumentProcessor
from explorer.analysis import TextAnalyzer
from explorer.viz import create_term_frequency_chart

# Initialize components
processor = DocumentProcessor()
analyzer = TextAnalyzer(
    remove_stopwords=True,
    ngram_range=(1, 2),
    min_word_length=3
)

# Process documents
with open('document.txt', 'rb') as f:
    f.name = 'document.txt'  # Required for processing
    documents = processor.load_document(f)

# Create DataFrame and analyze
import pandas as pd
df = pd.DataFrame(documents)
term_freq = analyzer.compute_term_frequency(df, top_n=20)

# Create visualization
fig = create_term_frequency_chart(term_freq)
fig.show()
```

## 📖 Usage Guide

### Web Interface Workflow

1. **Upload Documents**
   - Use the sidebar file uploader
   - Select single or multiple files
   - Supported formats: TXT, CSV, JSON, PDF, DOCX, ZIP

2. **Configure Analysis**
   - Adjust preprocessing options
   - Set analysis parameters (top N terms, n-gram range)
   - Choose language for stopwords

3. **Explore Results**
   - View document statistics
   - Analyze term frequencies
   - Generate word clouds
   - Examine n-gram patterns
   - Preview individual documents

4. **Export Data**
   - Download term frequency tables
   - Export processed document data
   - Save visualization results

### Advanced Configuration

#### Text Preprocessing Options
```python
analyzer = TextAnalyzer(
    remove_stopwords=True,          # Remove common words
    remove_punctuation=True,        # Strip punctuation
    lowercase=True,                 # Convert to lowercase
    remove_numbers=False,           # Keep/remove numbers
    stem_words=False,               # Apply stemming
    lemmatize_words=True,           # Apply lemmatization
    language='english',             # Stopwords language
    min_word_length=3,              # Minimum word length
    ngram_range=(1, 2),             # Unigrams and bigrams
    max_features=1000               # Limit vocabulary size
)
```

#### Custom Document Processing
```python
# Process specific document types
processor = DocumentProcessor()

# CSV with custom text column detection
documents = processor.load_document(csv_file)

# PDF with page-level extraction
documents = processor.load_document(pdf_file)

# ZIP archive processing
documents = processor.load_document(zip_file)
```

## 🔧 API Reference

### DocumentProcessor Class
```python
class DocumentProcessor:
    def load_document(self, file_obj) -> List[Dict]
    """Load and process documents from various formats."""
```

### TextAnalyzer Class
```python
class TextAnalyzer:
    def __init__(self, **kwargs)
    """Initialize with preprocessing options."""
    
    def preprocess_text(self, text: str) -> str
    """Preprocess a single text string."""
    
    def compute_term_frequency(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame
    """Compute term frequencies from text data."""
    
    def compute_tfidf(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame
    """Compute TF-IDF scores for terms."""
    
    def compute_ngram_frequency(self, df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame
    """Extract and count n-grams."""
```

### Visualization Functions
```python
def create_term_frequency_chart(freq_df: pd.DataFrame, top_n: int = 20) -> go.Figure
def create_wordcloud(freq_df: pd.DataFrame, width: int = 800, height: int = 400) -> plt.Figure
def create_ngram_chart(ngram_df: pd.DataFrame, ngram_range: Tuple[int, int], top_n: int = 20) -> go.Figure
def create_document_stats(stats: Dict) -> go.Figure
def create_sentiment_chart(sentiment_df: pd.DataFrame) -> go.Figure
def create_topic_visualization(topic_data: Dict) -> go.Figure
```

## 🎯 Use Cases

### Academic Research
- Analyze research papers and literature reviews
- Extract key terms and themes from academic documents
- Compare document similarity across papers
- Generate word clouds for presentation materials

### Business Intelligence
- Process customer feedback and reviews
- Analyze market research documents
- Extract insights from business reports
- Monitor competitor content and messaging

### Content Analysis
- Social media content analysis
- News article processing
- Blog post and web content analysis
- Email and communication analysis

### Legal Document Review
- Contract analysis and term extraction
- Legal document comparison
- Case law research and analysis
- Compliance document review

## 🛡️ Error Handling

The application includes comprehensive error handling for:
- **File Format Issues** - Unsupported formats, corrupted files
- **Encoding Problems** - Automatic encoding detection and fallback
- **Memory Management** - Large file processing with progress indicators
- **NLTK Dependencies** - Graceful fallback when NLTK data is unavailable
- **Visualization Errors** - Empty data handling and user feedback

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/your-username/enhanced-document-explorer.git
cd enhanced-document-explorer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 explorer/
black explorer/
```

### Reporting Issues
Please use the [GitHub Issues](https://github.com/your-username/enhanced-document-explorer/issues) page to report bugs or request features.

## 📋 Roadmap

### Version 2.1 (Planned)
- [ ] Real-time collaboration features
- [ ] Advanced topic modeling with coherence scoring
- [ ] Named entity recognition (NER)
- [ ] Multi-language document translation
- [ ] API endpoint for programmatic access

### Version 2.2 (Future)
- [ ] Machine learning model training interface
- [ ] Custom preprocessing pipeline builder
- [ ] Advanced similarity algorithms
- [ ] Export to various formats (Word, PowerPoint, etc.)
- [ ] Integration with cloud storage services

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NLTK Team** - Natural Language Toolkit
- **Streamlit Team** - Amazing web app framework
- **Plotly Team** - Interactive visualization library
- **scikit-learn** - Machine learning library
- **Open Source Community** - For inspiration and contributions

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/your-username/enhanced-document-explorer/wiki)
- **Issues**: [GitHub Issues](https://github.com/your-username/enhanced-document-explorer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/enhanced-document-explorer/discussions)
- **Email**: support@document-explorer.com

---

**Made with ❤️ by Nsikak Menim**