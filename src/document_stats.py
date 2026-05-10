"""
Document Statistics Module
Extracts and stores deterministic document metadata (paragraph count, sentence count, word count, etc.)
so factual queries about document structure can be answered without relying on retrieval.
"""

from typing import Dict, Any
import re


class DocumentStats:
    """Calculate and store document-level statistics."""
    
    @staticmethod
    def calculate_stats(text: str, doc_name: str) -> Dict[str, Any]:
        """
        Calculate comprehensive document statistics.
        
        Args:
            text: Full document text
            doc_name: Document name for reference
            
        Returns:
            Dictionary of statistics
        """
        if not text:
            return DocumentStats._empty_stats(doc_name)
        
        # Basic counts
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Sentence count (simple: count periods, question marks, exclamation marks)
        sentence_pattern = r'[.!?]+'
        sentences = re.findall(sentence_pattern, text)
        sentence_count = len(sentences)
        
        # Word count
        words = text.split()
        word_count = len(words)
        
        # Average metrics
        avg_words_per_paragraph = word_count / paragraph_count if paragraph_count > 0 else 0
        avg_words_per_sentence = word_count / sentence_count if sentence_count > 0 else 0
        
        # Character count
        char_count = len(text)
        
        # Line count
        lines = text.split('\n')
        line_count = len(lines)
        
        # Unique word count (approximate)
        unique_words = len(set(word.lower() for word in words if len(word) > 2))
        
        # Page estimate (assuming ~300 words per page)
        estimated_pages = round(word_count / 300, 1)
        
        return {
            'doc_name': doc_name,
            'paragraph_count': paragraph_count,
            'sentence_count': sentence_count,
            'word_count': word_count,
            'character_count': char_count,
            'line_count': line_count,
            'unique_words': unique_words,
            'avg_words_per_paragraph': round(avg_words_per_paragraph, 2),
            'avg_words_per_sentence': round(avg_words_per_sentence, 2),
            'estimated_pages': estimated_pages,
        }
    
    @staticmethod
    def _empty_stats(doc_name: str) -> Dict[str, Any]:
        """Return zero stats for empty documents."""
        return {
            'doc_name': doc_name,
            'paragraph_count': 0,
            'sentence_count': 0,
            'word_count': 0,
            'character_count': 0,
            'line_count': 0,
            'unique_words': 0,
            'avg_words_per_paragraph': 0,
            'avg_words_per_sentence': 0,
            'estimated_pages': 0,
        }
    
    @staticmethod
    def format_stats_for_display(stats: Dict[str, Any]) -> str:
        """
        Format statistics for human-readable display.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Formatted string
        """
        if not stats:
            return "No statistics available."
        
        return (
            f"Document: {stats.get('doc_name', 'Unknown')}\n"
            f"  Paragraphs: {stats.get('paragraph_count', 0)}\n"
            f"  Sentences: {stats.get('sentence_count', 0)}\n"
            f"  Words: {stats.get('word_count', 0)}\n"
            f"  Characters: {stats.get('character_count', 0)}\n"
            f"  Unique words: {stats.get('unique_words', 0)}\n"
            f"  Avg words/paragraph: {stats.get('avg_words_per_paragraph', 0)}\n"
            f"  Avg words/sentence: {stats.get('avg_words_per_sentence', 0)}\n"
            f"  Estimated pages: {stats.get('estimated_pages', 0)}"
        )
    
    @staticmethod
    def answer_stat_question(question: str, stats: Dict[str, Any]) -> str:
        """
        Try to answer a factual question using document stats.
        
        Args:
            question: User question (lowercase for matching)
            stats: Statistics dictionary
            
        Returns:
            Answer string if found, None otherwise
        """
        q_lower = question.lower()
        
        if not stats:
            return None
        
        # Guardrail: do not treat semantic/content questions as stat questions.
        semantic_markers = [
            'main character', 'white rabbit', 'tea party',
            'plot', 'story summary', 'summarize the story'
        ]
        if any(marker in q_lower for marker in semantic_markers):
            return None

        count_markers = [
            'how many', 'number of', 'count', 'total', 'stats', 'statistics',
            'how much', 'average', 'avg', 'length', 'size'
        ]
        has_count_intent = any(marker in q_lower for marker in count_markers)

        # Paragraph questions
        if has_count_intent and ('paragraph' in q_lower or 'section' in q_lower):
            return f"{stats.get('paragraph_count', 0)} paragraphs"
        
        # Sentence questions
        if has_count_intent and 'sentence' in q_lower:
            return f"{stats.get('sentence_count', 0)} sentences"
        
        # Word count questions
        if has_count_intent and ('word' in q_lower or 'words' in q_lower):
            return f"{stats.get('word_count', 0)} words"
        
        # Character questions
        if (
            'character count' in q_lower
            or 'number of characters' in q_lower
            or 'how many characters' in q_lower
            or (has_count_intent and 'characters' in q_lower)
        ):
            return f"{stats.get('character_count', 0)} characters"
        
        # Length/size questions
        if (
            'how many pages' in q_lower
            or 'number of pages' in q_lower
            or 'page count' in q_lower
            or 'length' in q_lower
            or 'size' in q_lower
        ):
            return f"Approximately {stats.get('estimated_pages', 0)} pages ({stats.get('word_count', 0)} words)"
        
        # Line questions
        if has_count_intent and 'line' in q_lower:
            return f"{stats.get('line_count', 0)} lines"
        
        return None


__all__ = ["DocumentStats"]
