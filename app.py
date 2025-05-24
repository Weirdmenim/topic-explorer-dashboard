# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from explorer.data import DocumentProcessor
from explorer.viz import create_term_frequency_chart, create_wordcloud, create_ngram_chart, create_document_stats
from explorer.analysis import TextAnalyzer
import io

# Page configuration
st.set_page_config(
    page_title="Enhanced Document Explorer",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">📚 Enhanced Document Explorer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['txt', 'csv', 'json', 'pdf', 'docx', 'zip'],
            accept_multiple_files=True,
            help="Supported formats: TXT, CSV, JSON, PDF, DOCX, ZIP archives"
        )
        
        st.markdown("---")
        
        # Analysis parameters
        st.subheader("Analysis Parameters")
        top_n = st.slider("Top N terms to display", 5, 100, 20)
        min_word_length = st.slider("Minimum word length", 1, 10, 3)
        ngram_range = st.selectbox("N-gram analysis", 
                                 options=[(1, 1), (1, 2), (1, 3), (2, 2), (2, 3)],
                                 format_func=lambda x: f"{x[0]}-{x[1]} grams")
        
        # Text preprocessing options
        st.subheader("Text Preprocessing")
        remove_stopwords = st.checkbox("Remove stopwords", value=True)
        remove_punctuation = st.checkbox("Remove punctuation", value=True)
        lowercase = st.checkbox("Convert to lowercase", value=True)
        remove_numbers = st.checkbox("Remove numbers", value=False)
        
        # Language selection
        language = st.selectbox("Language for stopwords", 
                               ["english", "spanish", "french", "german", "italian"],
                               index=0)
    
    # Main content area
    if uploaded_files:
        # Process documents
        with st.spinner("Processing documents..."):
            processor = DocumentProcessor()
            
            # Load documents
            documents = []
            for file in uploaded_files:
                try:
                    docs = processor.load_document(file)
                    documents.extend(docs)
                except Exception as e:
                    st.error(f"Error processing {file.name}: {str(e)}")
            
            if documents:
                # Create DataFrame
                df = pd.DataFrame(documents)
                
                # Initialize analyzer
                analyzer = TextAnalyzer(
                    remove_stopwords=remove_stopwords,
                    remove_punctuation=remove_punctuation,
                    lowercase=lowercase,
                    remove_numbers=remove_numbers,
                    language=language,
                    min_word_length=min_word_length,
                    ngram_range=ngram_range
                )
                
                # Store in session state
                st.session_state.processed_data = df
                st.session_state.analyzer = analyzer
        
        # Display results if data is available
        if st.session_state.processed_data is not None:
            df = st.session_state.processed_data
            analyzer = st.session_state.analyzer
            
            # Document statistics
            st.header("📊 Document Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Documents", len(df))
            with col2:
                total_words = df['text'].apply(lambda x: len(str(x).split())).sum()
                st.metric("Total Words", f"{total_words:,}")
            with col3:
                avg_length = df['text'].apply(lambda x: len(str(x).split())).mean()
                st.metric("Avg Words/Doc", f"{avg_length:.0f}")
            with col4:
                unique_sources = df['source'].nunique() if 'source' in df.columns else len(df)
                st.metric("Unique Sources", unique_sources)
            
            # Term frequency analysis
            st.header("🔤 Term Frequency Analysis")
            
            with st.spinner("Analyzing term frequencies..."):
                term_freq = analyzer.compute_term_frequency(df, top_n=top_n)
            
            if not term_freq.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Term Frequency Chart")
                    fig_bar = create_term_frequency_chart(term_freq, top_n)
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                with col2:
                    st.subheader("Word Cloud")
                    fig_cloud = create_wordcloud(term_freq)
                    st.pyplot(fig_cloud, use_container_width=True)
            
            # N-gram analysis
            st.header("📝 N-gram Analysis")
            
            with st.spinner("Performing n-gram analysis..."):
                ngram_freq = analyzer.compute_ngram_frequency(df, top_n=top_n)
            
            if not ngram_freq.empty:
                fig_ngram = create_ngram_chart(ngram_freq, ngram_range, top_n)
                st.plotly_chart(fig_ngram, use_container_width=True)
            
            # Document-level insights
            st.header("📄 Document Insights")
            
            # Document length distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Document Length Distribution")
                doc_lengths = df['text'].apply(lambda x: len(str(x).split()))
                fig_hist = px.histogram(
                    x=doc_lengths,
                    title="Distribution of Document Lengths (Words)",
                    labels={'x': 'Words per Document', 'y': 'Frequency'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                st.subheader("Top Documents by Length")
                top_docs = df.copy()
                top_docs['word_count'] = top_docs['text'].apply(lambda x: len(str(x).split()))
                top_docs = top_docs.nlargest(10, 'word_count')[['source', 'word_count']]
                st.dataframe(top_docs, use_container_width=True)
            
            # Detailed term frequency table
            st.header("📋 Detailed Analysis")
            
            tab1, tab2, tab3 = st.tabs(["Term Frequencies", "Document Preview", "Raw Data"])
            
            with tab1:
                st.subheader(f"Top {top_n} Terms")
                if not term_freq.empty:
                    # Add percentage column
                    total_terms = term_freq['count'].sum()
                    term_freq_display = term_freq.copy()
                    term_freq_display['percentage'] = (term_freq_display['count'] / total_terms * 100).round(2)
                    st.dataframe(term_freq_display, use_container_width=True)
                    
                    # Download button
                    csv = term_freq_display.to_csv(index=False)
                    st.download_button(
                        label="Download Term Frequencies as CSV",
                        data=csv,
                        file_name="term_frequencies.csv",
                        mime="text/csv"
                    )
            
            with tab2:
                st.subheader("Document Preview")
                if len(df) > 0:
                    selected_doc = st.selectbox("Select document to preview:", 
                                              range(len(df)), 
                                              format_func=lambda x: f"Doc {x+1}: {df.iloc[x]['source'] if 'source' in df.columns else f'Document {x+1}'}")
                    
                    if selected_doc is not None:
                        doc_text = df.iloc[selected_doc]['text']
                        st.text_area("Document Content", doc_text, height=300)
                        
                        # Document-specific stats
                        words = len(str(doc_text).split())
                        chars = len(str(doc_text))
                        st.write(f"**Words:** {words} | **Characters:** {chars}")
            
            with tab3:
                st.subheader("Raw Data")
                st.dataframe(df, use_container_width=True)
                
                # Download raw data
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download Raw Data as CSV",
                    data=csv,
                    file_name="processed_documents.csv",
                    mime="text/csv"
                )
    
    else:
        # Welcome message
        st.info("👆 Upload documents using the sidebar to begin analysis")
        
        # Feature overview
        st.header("🚀 Features")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📁 Multiple Formats**
            - PDF documents
            - Word documents (DOCX)
            - Text files (TXT)
            - CSV/JSON data
            - ZIP archives
            """)
        
        with col2:
            st.markdown("""
            **🔍 Advanced Analysis**
            - Term frequency analysis
            - N-gram extraction
            - Word clouds
            - Document statistics
            - Customizable preprocessing
            """)
        
        with col3:
            st.markdown("""
            **📊 Rich Visualizations**
            - Interactive charts
            - Word clouds
            - Distribution plots
            - Downloadable results
            - Document previews
            """)

if __name__ == "__main__":
    main()