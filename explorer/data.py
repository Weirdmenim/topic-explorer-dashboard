# explorer/data.py
import pandas as pd
import os
import io
import zipfile
import json
from typing import List, Dict, Union, Optional
import PyPDF2
from docx import Document
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Enhanced document processor supporting multiple file formats."""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.csv', '.json', '.pdf', '.docx', '.zip'}
    
    def load_document(self, file_obj) -> List[Dict]:
        """
        Load and process a document from various formats.
        
        Args:
            file_obj: Streamlit uploaded file object
            
        Returns:
            List of dictionaries containing processed document data
        """
        file_extension = self._get_file_extension(file_obj.name)
        
        if file_extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_extension}")
        
        try:
            if file_extension == '.txt':
                return self._process_txt(file_obj)
            elif file_extension == '.csv':
                return self._process_csv(file_obj)
            elif file_extension == '.json':
                return self._process_json(file_obj)
            elif file_extension == '.pdf':
                return self._process_pdf(file_obj)
            elif file_extension == '.docx':
                return self._process_docx(file_obj)
            elif file_extension == '.zip':
                return self._process_zip(file_obj)
        except Exception as e:
            logger.error(f"Error processing {file_obj.name}: {str(e)}")
            raise
    
    def _get_file_extension(self, filename: str) -> str:
        """Extract file extension from filename."""
        return os.path.splitext(filename.lower())[1]
    
    def _process_txt(self, file_obj) -> List[Dict]:
        """Process plain text files."""
        try:
            content = file_obj.read().decode('utf-8')
            return [{
                'source': file_obj.name,
                'text': content.strip(),
                'type': 'txt'
            }]
        except UnicodeDecodeError:
            # Try with different encoding
            file_obj.seek(0)
            content = file_obj.read().decode('latin-1')
            return [{
                'source': file_obj.name,
                'text': content.strip(),
                'type': 'txt'
            }]
    
    def _process_csv(self, file_obj) -> List[Dict]:
        """Process CSV files."""
        df = pd.read_csv(file_obj)
        
        # Try to find text column
        text_columns = ['text', 'content', 'body', 'message', 'description']
        text_col = None
        
        for col in text_columns:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            # Use the first string column
            string_cols = df.select_dtypes(include=['object']).columns
            if len(string_cols) > 0:
                text_col = string_cols[0]
            else:
                raise ValueError("No suitable text column found in CSV")
        
        documents = []
        for idx, row in df.iterrows():
            documents.append({
                'source': f"{file_obj.name}_row_{idx}",
                'text': str(row[text_col]),
                'type': 'csv',
                'metadata': {k: v for k, v in row.items() if k != text_col}
            })
        
        return documents
    
    def _process_json(self, file_obj) -> List[Dict]:
        """Process JSON files."""
        data = json.load(file_obj)
        
        if isinstance(data, list):
            documents = []
            for idx, item in enumerate(data):
                text = self._extract_text_from_json(item)
                documents.append({
                    'source': f"{file_obj.name}_item_{idx}",
                    'text': text,
                    'type': 'json',
                    'metadata': item if isinstance(item, dict) else {}
                })
            return documents
        elif isinstance(data, dict):
            text = self._extract_text_from_json(data)
            return [{
                'source': file_obj.name,
                'text': text,
                'type': 'json',
                'metadata': data
            }]
        else:
            return [{
                'source': file_obj.name,
                'text': str(data),
                'type': 'json'
            }]
    
    def _extract_text_from_json(self, obj) -> str:
        """Extract text from JSON object, trying common text fields."""
        if isinstance(obj, str):
            return obj
        elif isinstance(obj, dict):
            text_fields = ['text', 'content', 'body', 'message', 'description', 'title']
            for field in text_fields:
                if field in obj:
                    return str(obj[field])
            # Concatenate all string values
            text_parts = []
            for value in obj.values():
                if isinstance(value, str):
                    text_parts.append(value)
            return ' '.join(text_parts)
        else:
            return str(obj)
    
    def _process_pdf(self, file_obj) -> List[Dict]:
        """Process PDF files."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_obj.read())
                tmp_file.flush()
                
                # Read PDF
                with open(tmp_file.name, 'rb') as pdf_file:
                    pdf_reader = PyPDF2.PdfReader(pdf_file)
                    
                    documents = []
                    for page_num, page in enumerate(pdf_reader.pages):
                        text = page.extract_text()
                        if text.strip():  # Only add non-empty pages
                            documents.append({
                                'source': f"{file_obj.name}_page_{page_num + 1}",
                                'text': text.strip(),
                                'type': 'pdf',
                                'metadata': {'page': page_num + 1}
                            })
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
            return documents
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_obj.name}: {str(e)}")
            raise ValueError(f"Could not process PDF file: {str(e)}")
    
    def _process_docx(self, file_obj) -> List[Dict]:
        """Process DOCX files."""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
                tmp_file.write(file_obj.read())
                tmp_file.flush()
                
                # Read DOCX
                doc = Document(tmp_file.name)
                
                # Extract text from paragraphs
                paragraphs = []
                for para in doc.paragraphs:
                    if para.text.strip():
                        paragraphs.append(para.text.strip())
                
                # Combine all text
                full_text = '\n'.join(paragraphs)
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
            return [{
                'source': file_obj.name,
                'text': full_text,
                'type': 'docx',
                'metadata': {'paragraphs': len(paragraphs)}
            }]
            
        except Exception as e:
            logger.error(f"Error processing DOCX {file_obj.name}: {str(e)}")
            raise ValueError(f"Could not process DOCX file: {str(e)}")
    
    def _process_zip(self, file_obj) -> List[Dict]:
        """Process ZIP archives containing multiple files."""
        documents = []
        
        try:
            with zipfile.ZipFile(file_obj, 'r') as zip_file:
                for file_info in zip_file.filelist:
                    if file_info.is_dir():
                        continue
                    
                    file_ext = self._get_file_extension(file_info.filename)
                    
                    try:
                        with zip_file.open(file_info.filename) as inner_file:
                            if file_ext == '.txt':
                                content = inner_file.read().decode('utf-8')
                                documents.append({
                                    'source': f"{file_obj.name}/{file_info.filename}",
                                    'text': content.strip(),
                                    'type': 'txt_from_zip'
                                })
                            elif file_ext in ['.csv', '.json']:
                                # Create a file-like object for processing
                                inner_content = inner_file.read()
                                inner_file_obj = io.BytesIO(inner_content)
                                inner_file_obj.name = file_info.filename
                                
                                if file_ext == '.csv':
                                    inner_docs = self._process_csv(inner_file_obj)
                                else:
                                    inner_docs = self._process_json(inner_file_obj)
                                
                                # Update source to include zip path
                                for doc in inner_docs:
                                    doc['source'] = f"{file_obj.name}/{doc['source']}"
                                
                                documents.extend(inner_docs)
                    
                    except Exception as e:
                        logger.warning(f"Could not process {file_info.filename} from ZIP: {str(e)}")
                        continue
        
        except Exception as e:
            logger.error(f"Error processing ZIP {file_obj.name}: {str(e)}")
            raise ValueError(f"Could not process ZIP file: {str(e)}")
        
        if not documents:
            raise ValueError("No processable files found in ZIP archive")
        
        return documents


def load_corpus(path_or_buffer):
    """
    Legacy function for backward compatibility.
    Use DocumentProcessor.load_document() for new implementations.
    """
    processor = DocumentProcessor()
    
    # Handle different input types
    if hasattr(path_or_buffer, 'name'):
        # File-like object
        return pd.DataFrame(processor.load_document(path_or_buffer))
    elif isinstance(path_or_buffer, str):
        # File path - create a file object
        with open(path_or_buffer, 'rb') as f:
            # Create a mock file object with name attribute
            class MockFile:
                def __init__(self, file_obj, name):
                    self.file_obj = file_obj
                    self.name = name
                
                def read(self):
                    return self.file_obj.read()
                
                def seek(self, pos):
                    return self.file_obj.seek(pos)
            
            mock_file = MockFile(f, path_or_buffer)
            documents = processor.load_document(mock_file)
            return pd.DataFrame(documents)
    else:
        raise TypeError("Input must be a file path or file-like object")


def compute_term_freq(df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Legacy function for backward compatibility.
    Use TextAnalyzer.compute_term_frequency() for new implementations.
    """
    from sklearn.feature_extraction.text import CountVectorizer
    
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