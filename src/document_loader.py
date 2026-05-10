"""
Document Loader Module
Handles loading and parsing of multiple document formats (PDF, TXT, DOCX, MD, HTML)
with comprehensive error handling and metadata extraction.
"""

import os
from typing import List, Tuple, Dict, Optional
from pathlib import Path
from io import BytesIO

from .document_stats import DocumentStats


class DocumentLoader:
    """Load and extract text from multiple document formats."""
    
    SUPPORTED_FORMATS = {'.pdf', '.txt', '.docx', '.md', '.html', '.htm'}
    
    def __init__(self):
        """Initialize the document loader."""
        self.supported_formats = self.SUPPORTED_FORMATS
    
    def load_document(self, file_path: str) -> Tuple[str, str, Dict]:
        """
        Load a document and extract its text content with metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Tuple of (document_name, content, metadata)
            Metadata includes: source, file_type, page_count
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported: {', '.join(self.supported_formats)}"
            )
        
        doc_name = file_path.stem
        metadata = {
            'source': str(file_path),
            'file_type': file_ext[1:],  # Remove leading dot
            'file_name': file_path.name,
            'doc_name': doc_name,
        }
        
        # Route to appropriate loader (we need content first for stats)
        if file_ext == '.pdf':
            content, page_count = self._load_pdf(file_path)
            metadata['page_count'] = page_count
        elif file_ext == '.docx':
            content = self._load_docx(file_path)
        elif file_ext == '.txt':
            content = self._load_txt(file_path)
        elif file_ext in {'.html', '.htm'}:
            content = self._load_html(file_path)
        elif file_ext == '.md':
            content = self._load_markdown(file_path)
        
        # Calculate and add document statistics
        stats = DocumentStats.calculate_stats(content, doc_name)
        metadata.update(stats)
        
        return doc_name, content, metadata
    
    def load_uploaded_file(self, uploaded_file) -> Tuple[str, str, Dict]:
        """
        Load a file uploaded via Streamlit.
        
        Args:
            uploaded_file: Streamlit UploadedFile object
            
        Returns:
            Tuple of (document_name, content, metadata)
        """
        file_name = uploaded_file.name
        file_ext = Path(file_name).suffix.lower()
        
        if file_ext not in self.supported_formats:
            raise ValueError(
                f"Unsupported file format: {file_ext}. "
                f"Supported: {', '.join(self.supported_formats)}"
            )
        
        doc_name = Path(file_name).stem
        file_bytes = uploaded_file.read()
        
        metadata = {
            'source': 'uploaded',
            'file_type': file_ext[1:],
            'file_name': file_name,
            'file_size': len(file_bytes),
            'doc_name': doc_name,
        }
        
        # Route to appropriate loader
        if file_ext == '.pdf':
            content, page_count = self._load_pdf_from_bytes(file_bytes)
            metadata['page_count'] = page_count
        elif file_ext == '.docx':
            content = self._load_docx_from_bytes(file_bytes)
        elif file_ext == '.txt':
            content = file_bytes.decode('utf-8', errors='ignore')
        elif file_ext in {'.html', '.htm'}:
            content = self._load_html_from_bytes(file_bytes)
        elif file_ext == '.md':
            content = file_bytes.decode('utf-8', errors='ignore')
        
        # Calculate and add document statistics
        stats = DocumentStats.calculate_stats(content, doc_name)
        metadata.update(stats)
        
        return doc_name, content, metadata
    
    @staticmethod
    def _load_pdf(file_path: Path) -> Tuple[str, int]:
        """
        Extract text from PDF file with page tracking.
        Uses pdfplumber for better extraction.
        """
        try:
            import pdfplumber
            text_content = []
            page_count = 0
            
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        # Add page marker for metadata
                        text = f"[PAGE {page_num}]\n{text}"
                        text_content.append(text)
            
            if not text_content:
                raise ValueError("No text could be extracted from PDF")
                
            return '\n\n'.join(text_content), page_count
        
        except ImportError:
            # Fallback to PyPDF2 if pdfplumber not available
            try:
                from pypdf import PdfReader
                text_content = []
                
                with open(file_path, 'rb') as pdf_file:
                    pdf_reader = PdfReader(pdf_file)
                    page_count = len(pdf_reader.pages)
                    
                    for page_num, page in enumerate(pdf_reader.pages, 1):
                        text = page.extract_text()
                        if text:
                            text = f"[PAGE {page_num}]\n{text}"
                            text_content.append(text)
                
                if not text_content:
                    raise ValueError("No text could be extracted from PDF")
                    
                return '\n\n'.join(text_content), page_count
            
            except Exception as e:
                raise ValueError(f"Error reading PDF file: {str(e)}")
        
        except Exception as e:
            raise ValueError(f"Error reading PDF file: {str(e)}")
    
    @staticmethod
    def _load_pdf_from_bytes(file_bytes: bytes) -> Tuple[str, int]:
        """Extract text from PDF bytes with page tracking."""
        try:
            import pdfplumber
            text_content = []
            
            with pdfplumber.open(BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text = f"[PAGE {page_num}]\n{text}"
                        text_content.append(text)
            
            if not text_content:
                raise ValueError("No text could be extracted from PDF")
                
            return '\n\n'.join(text_content), page_count
        
        except ImportError:
            # Fallback to PyPDF2
            try:
                from pypdf import PdfReader
                text_content = []
                
                pdf_file = BytesIO(file_bytes)
                pdf_reader = PdfReader(pdf_file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text:
                        text = f"[PAGE {page_num}]\n{text}"
                        text_content.append(text)
                
                if not text_content:
                    raise ValueError("No text could be extracted from PDF")
                    
                return '\n\n'.join(text_content), page_count
            
            except Exception as e:
                raise ValueError(f"Error reading PDF: {str(e)}")
        
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
    
    @staticmethod
    def _load_docx(file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            from docx import Document
            
            doc = Document(file_path)
            text_content = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)
            
            # Also extract from tables
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(
                        cell.text.strip() for cell in row.cells
                    )
                    if row_text.strip():
                        text_content.append(row_text)
            
            if not text_content:
                raise ValueError("No text could be extracted from DOCX")
            
            return '\n\n'.join(text_content)
        
        except ImportError:
            raise ImportError("python-docx is required for DOCX support")
        except Exception as e:
            raise ValueError(f"Error reading DOCX file: {str(e)}")
    
    @staticmethod
    def _load_docx_from_bytes(file_bytes: bytes) -> str:
        """Extract text from DOCX bytes."""
        try:
            from docx import Document
            
            doc = Document(BytesIO(file_bytes))
            text_content = []
            
            for para in doc.paragraphs:
                if para.text.strip():
                    text_content.append(para.text)
            
            for table in doc.tables:
                for row in table.rows:
                    row_text = ' | '.join(
                        cell.text.strip() for cell in row.cells
                    )
                    if row_text.strip():
                        text_content.append(row_text)
            
            if not text_content:
                raise ValueError("No text could be extracted from DOCX")
            
            return '\n\n'.join(text_content)
        
        except ImportError:
            raise ImportError("python-docx is required for DOCX support")
        except Exception as e:
            raise ValueError(f"Error reading DOCX: {str(e)}")
    
    @staticmethod
    def _load_txt(file_path: Path) -> str:
        """Extract text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                raise ValueError("TXT file is empty")
            
            return content
        
        except Exception as e:
            raise ValueError(f"Error reading TXT file: {str(e)}")
    
    @staticmethod
    def _load_html(file_path: Path) -> str:
        """Extract text from HTML file."""
        try:
            from bs4 import BeautifulSoup
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(['script', 'style']):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            if not text.strip():
                raise ValueError("No text could be extracted from HTML")
            
            return text
        
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML support")
        except Exception as e:
            raise ValueError(f"Error reading HTML file: {str(e)}")
    
    @staticmethod
    def _load_html_from_bytes(file_bytes: bytes) -> str:
        """Extract text from HTML bytes."""
        try:
            from bs4 import BeautifulSoup
            
            html_content = file_bytes.decode('utf-8', errors='ignore')
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for script in soup(['script', 'style']):
                script.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            if not text.strip():
                raise ValueError("No text could be extracted from HTML")
            
            return text
        
        except ImportError:
            raise ImportError("beautifulsoup4 is required for HTML support")
        except Exception as e:
            raise ValueError(f"Error reading HTML: {str(e)}")
    
    @staticmethod
    def _load_markdown(file_path: Path) -> str:
        """Extract text from Markdown file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            if not content.strip():
                raise ValueError("Markdown file is empty")
            
            return content
        
        except Exception as e:
            raise ValueError(f"Error reading Markdown file: {str(e)}")
