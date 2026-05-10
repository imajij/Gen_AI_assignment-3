"""
Text Chunker Module
Uses LangChain's RecursiveCharacterTextSplitter for semantic-aware text chunking
with preservation of metadata (source, page number, chunk index).
"""

from typing import List, Dict, Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextChunker:
    """Split documents into chunks using recursive character splitting."""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """
        Initialize the text chunker with LangChain's RecursiveCharacterTextSplitter.
        
        Args:
            chunk_size: Size of each chunk in characters (default: 800)
            chunk_overlap: Number of overlapping characters between chunks (default: 150)
            
        Raises:
            ValueError: If chunk_size or chunk_overlap are invalid
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LangChain's RecursiveCharacterTextSplitter
        # Uses separators in order: paragraphs, sentences, spaces, characters
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",           # Paragraph breaks (strongest preference)
                "\n",             # Line breaks
                ". ",             # Sentences
                " ",              # Words
                ""                # Characters (fallback)
            ],
            length_function=len,
            is_separator_regex=False
        )
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """
        Split text into chunks using recursive character splitting.
        
        Args:
            text: The text content to chunk
            metadata: Dictionary with keys: doc_name, page_count (optional), source, file_type, etc.
            
        Returns:
            List of chunk dictionaries with structure:
            {
                'id': 'unique_chunk_id',
                'content': 'chunk text...',
                'metadata': {
                    'source': 'filename or path',
                    'doc_name': 'document name',
                    'chunk_index': 0,
                    'page_number': 1,  # if extractable from text
                    'file_type': 'pdf',
                    'page_count': total,
                    ...other_metadata
                }
            }
            
        Raises:
            ValueError: If text is empty or invalid
        """
        if not text or not isinstance(text, str):
            raise ValueError("Text must be a non-empty string")
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        if not text.strip():
            return []
        
        # Split using LangChain's recursive splitter
        text_chunks = self.splitter.split_text(text)
        
        if not text_chunks:
            return []
        
        # Create chunk objects with metadata
        chunks = self._create_chunk_objects(
            text_chunks=text_chunks,
            metadata=metadata
        )
        
        return chunks
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace while preserving structure
        lines = text.split('\n')
        lines = [line.rstrip() for line in lines]
        
        # Remove empty lines, but keep paragraph breaks
        cleaned_lines = []
        prev_empty = False
        
        for line in lines:
            if line.strip():
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty and cleaned_lines:
                # Keep single empty lines for paragraph breaks
                cleaned_lines.append('')
                prev_empty = True
        
        return '\n'.join(cleaned_lines)
    
    def _create_chunk_objects(self, text_chunks: List[str], metadata: Dict) -> List[Dict]:
        """
        Create structured chunk objects with metadata.
        
        Args:
            text_chunks: List of text chunks from splitter
            metadata: Original document metadata
            
        Returns:
            List of structured chunk dictionaries
        """
        chunks = []
        doc_name = metadata.get('doc_name', 'unknown')
        
        for chunk_index, chunk_text in enumerate(text_chunks):
            # Extract page number from [PAGE N] markers in text
            page_number = self._extract_page_number(chunk_text)
            
            # Create chunk metadata
            chunk_metadata = {
                'source': metadata.get('source', 'unknown'),
                'doc_name': doc_name,
                'file_type': metadata.get('file_type', 'unknown'),
                'chunk_index': chunk_index,
                'page_number': page_number,
                'total_chunks': len(text_chunks),
            }
            
            # Add optional metadata
            if 'page_count' in metadata:
                chunk_metadata['page_count'] = metadata['page_count']
            if 'file_name' in metadata:
                chunk_metadata['file_name'] = metadata['file_name']
            if 'file_size' in metadata:
                chunk_metadata['file_size'] = metadata['file_size']
            
            # Add any custom metadata from original
            for key, value in metadata.items():
                if key not in chunk_metadata and key not in ['doc_name', 'source', 'file_type', 'page_count', 'file_name', 'file_size']:
                    chunk_metadata[key] = value
            
            # Create chunk object
            chunk_obj = {
                'id': f"{doc_name}_chunk_{chunk_index}",
                'content': chunk_text.strip(),
                'metadata': chunk_metadata
            }
            
            chunks.append(chunk_obj)
        
        return chunks
    
    @staticmethod
    def _extract_page_number(text: str) -> int:
        """
        Extract page number from [PAGE N] markers inserted by document loader.
        
        Args:
            text: Chunk text that may contain page markers
            
        Returns:
            Page number or 1 if not found
        """
        import re
        match = re.search(r'\[PAGE (\d+)\]', text)
        if match:
            return int(match.group(1))
        return 1
    
    def get_splitter_config(self) -> Dict:
        """
        Get current splitter configuration.
        
        Returns:
            Dictionary with current settings
        """
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'splitter_type': 'RecursiveCharacterTextSplitter',
            'separators': ["\n\n", "\n", ". ", " ", ""]
        }


class ChunkValidator:
    """Validate chunk quality and structure."""
    
    @staticmethod
    def validate_chunk(chunk: Dict) -> tuple[bool, str]:
        """
        Validate a chunk object.
        
        Args:
            chunk: Chunk dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if 'id' not in chunk:
            return False, "Missing 'id' field"
        
        if 'content' not in chunk:
            return False, "Missing 'content' field"
        
        if 'metadata' not in chunk:
            return False, "Missing 'metadata' field"
        
        # Validate content
        if not isinstance(chunk['content'], str):
            return False, "'content' must be a string"
        
        if not chunk['content'].strip():
            return False, "'content' is empty"
        
        # Validate metadata
        metadata = chunk['metadata']
        required_meta = ['source', 'doc_name', 'chunk_index', 'page_number']
        
        for field in required_meta:
            if field not in metadata:
                return False, f"Missing metadata field: '{field}'"
        
        # Validate types
        if not isinstance(metadata['chunk_index'], int) or metadata['chunk_index'] < 0:
            return False, "'chunk_index' must be non-negative integer"
        
        if not isinstance(metadata['page_number'], int) or metadata['page_number'] < 1:
            return False, "'page_number' must be positive integer"
        
        return True, "Valid chunk"
    
    @staticmethod
    def validate_chunks(chunks: List[Dict]) -> tuple[int, int, List[str]]:
        """
        Validate a list of chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Tuple of (valid_count, invalid_count, error_messages)
        """
        valid_count = 0
        invalid_count = 0
        errors = []
        
        for i, chunk in enumerate(chunks):
            is_valid, message = ChunkValidator.validate_chunk(chunk)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_count += 1
                errors.append(f"Chunk {i}: {message}")
        
        return valid_count, invalid_count, errors

