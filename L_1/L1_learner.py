"""
L1 Learner: Pattern-based Entity Learning and Intelligent Chunking System

This module provides tools for:
1. Learning patterns from text data for entity recognition
2. Pattern-based intelligent text chunking for LLM processing
3. Semantic chunking using sentence transformers
4. Configurable entity patterns via external files
"""

import re
import json
import os
from typing import Dict, Any, List, Optional
import pandas as pd

try:
    from sentence_transformers import SentenceTransformer, util
    from nltk.tokenize import sent_tokenize
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers or nltk not available. LLM_Chunker will not work.")


class LLM_Chunker:
    """
    Semantic chunker using sentence transformers for similarity-based text splitting.
    """
    
    def __init__(self):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers and nltk are required for LLM_Chunker")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def chunk_text(self, text: str, similarity_threshold: float = 0.7, max_chunk_sentences: int = 5) -> List[str]:
        """
        Chunk text based on semantic similarity between sentences.
        
        Args:
            text: Input text to chunk
            similarity_threshold: Threshold for semantic similarity (0.0-1.0)
            max_chunk_sentences: Maximum sentences per chunk
            
        Returns:
            List of text chunks
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return [text]
            
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        chunks = []
        start = 0
        
        for i in range(1, len(sentences)):
            sim = util.cos_sim(embeddings[i-1], embeddings[i]).item()
            # Create new chunk if similarity drops or chunk gets too long
            if sim < similarity_threshold or (i - start >= max_chunk_sentences):
                chunk = " ".join(sentences[start:i])
                chunks.append(chunk)
                start = i
        
        # Add final chunk
        if start < len(sentences):
            chunks.append(" ".join(sentences[start:]))
        
        return chunks


class ConfigurableBootstrapper:
    """
    A configurable bootstrapping system for learning entity patterns from external files.
    Supports loading entity types, examples, and heuristics from configuration files.
    """
    
    def __init__(self, entity_type: str, config_dir: str = "configs", confidence_threshold: float = 0.8):
        self.entity_type = entity_type
        self.config_dir = config_dir
        self.confidence_threshold = confidence_threshold
        self.patterns: Dict[str, Any] = {
            'word_patterns': {},
            'position_patterns': {},
            'context_patterns': {},
        }
        self.validated_examples: Dict[str, str] = {}
        self.entity_heuristics: Dict[str, Any] = {}
        
        # Load configuration from files
        self._load_configuration()
        
    def _load_configuration(self) -> None:
        """Load entity configuration from files."""
        if not os.path.exists(self.config_dir):
            print(f"Warning: Config directory '{self.config_dir}' not found. Using empty configuration.")
            return
            
        # Load validated examples
        examples_file = os.path.join(self.config_dir, f"{self.entity_type}_examples.txt")
        self._load_examples_from_file(examples_file)
        
        # Load heuristics
        heuristics_file = os.path.join(self.config_dir, f"{self.entity_type}_heuristics.json")
        self._load_heuristics_from_file(heuristics_file)
        
        # Load pre-trained patterns if available
        patterns_file = os.path.join(self.config_dir, f"{self.entity_type}_patterns.json")
        self._load_patterns_from_file(patterns_file)
    
    def _load_examples_from_file(self, file_path: str) -> None:
        """
        Load validated examples from text file.
        
        File format (one per line):
        original_text|normalized_form
        or
        original_text  # (if original and normalized are same)
        """
        if not os.path.exists(file_path):
            print(f"Info: Examples file '{file_path}' not found. Starting with empty examples.")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                        
                    if '|' in line:
                        parts = line.split('|', 1)
                        original_text = parts[0].strip()
                        normalized_form = parts[1].strip()
                    else:
                        original_text = line.strip()
                        normalized_form = original_text
                    
                    if original_text:
                        self.add_validated_example(original_text, normalized_form)
                        
            print(f"Loaded {len(self.validated_examples)} examples for {self.entity_type}")
            
        except Exception as e:
            print(f"Error loading examples from {file_path}: {e}")
    
    def _load_heuristics_from_file(self, file_path: str) -> None:
        """
        Load entity-specific heuristics from JSON file.
        
        File format:
        {
            "indicators": ["SECURITIES", "BROKING", "CAPITAL"],
            "suffixes": ["LTD", "LIMITED", "PVT"],
            "known_entities": ["ZERODHA", "UPSTOX"],
            "weights": {
                "indicators": 0.4,
                "suffixes": 0.3,
                "known_entities": 0.5
            }
        }
        """
        if not os.path.exists(file_path):
            print(f"Info: Heuristics file '{file_path}' not found. Using basic heuristics.")
            self.entity_heuristics = {"indicators": [], "suffixes": [], "known_entities": [], 
                                    "weights": {"indicators": 0.3, "suffixes": 0.2, "known_entities": 0.4}}
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.entity_heuristics = json.load(f)
            print(f"Loaded heuristics for {self.entity_type}")
            
        except Exception as e:
            print(f"Error loading heuristics from {file_path}: {e}")
            self.entity_heuristics = {"indicators": [], "suffixes": [], "known_entities": [], 
                                    "weights": {"indicators": 0.3, "suffixes": 0.2, "known_entities": 0.4}}
    
    def _load_patterns_from_file(self, file_path: str) -> None:
        """Load pre-trained patterns from JSON file."""
        if not os.path.exists(file_path):
            print(f"Info: Patterns file '{file_path}' not found. Starting with empty patterns.")
            return
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_patterns = json.load(f)
                
            # Merge loaded patterns with existing ones
            for pattern_type, pattern_data in loaded_patterns.items():
                if pattern_type in self.patterns:
                    self.patterns[pattern_type].update(pattern_data)
                    
            print(f"Loaded pre-trained patterns for {self.entity_type}")
            
        except Exception as e:
            print(f"Error loading patterns from {file_path}: {e}")
    
    def save_patterns_to_file(self, file_path: Optional[str] = None) -> None:
        """Save learned patterns to JSON file."""
        if file_path is None:
            file_path = os.path.join(self.config_dir, f"{self.entity_type}_patterns.json")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.patterns, f, indent=2, ensure_ascii=False)
            print(f"Saved patterns to {file_path}")
            
        except Exception as e:
            print(f"Error saving patterns to {file_path}: {e}")
    
    def save_examples_to_file(self, file_path: Optional[str] = None) -> None:
        """Save validated examples to text file."""
        if file_path is None:
            file_path = os.path.join(self.config_dir, f"{self.entity_type}_examples.txt")
            
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Validated examples for {self.entity_type}\n")
                f.write("# Format: original_text|normalized_form\n\n")
                
                for original, normalized in self.validated_examples.items():
                    if original == normalized:
                        f.write(f"{original}\n")
                    else:
                        f.write(f"{original}|{normalized}\n")
                        
            print(f"Saved examples to {file_path}")
            
        except Exception as e:
            print(f"Error saving examples to {file_path}: {e}")
        
    def add_validated_example(self, original_text: str, normalized_form: str) -> None:
        """Add a manually validated example to the knowledge base."""
        self.validated_examples[original_text] = normalized_form
        self._update_patterns_from_example(original_text)
    
    def _update_patterns_from_example(self, original_text: str) -> None:
        """Update pattern recognition from a validated example."""
        # Extract word patterns
        words = re.findall(r'\b\w+\b', original_text.upper())
        for word in words:
            self.patterns['word_patterns'][word] = self.patterns['word_patterns'].get(word, 0) + 1
    
    def learn_from_paragraph(self, paragraph: str, known_entities: Dict[str, str]) -> None:
        """
        Learn patterns from a paragraph where entities are marked.
        
        Args:
            paragraph: Text paragraph containing entities
            known_entities: Dict mapping entity text to normalized form
        """
        sentences = re.split(r'[.!?]+', paragraph)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            for entity_text, normalized_form in known_entities.items():
                if entity_text.lower() in sentence.lower():
                    self._learn_position_pattern(sentence, entity_text)
                    self._learn_context_pattern(sentence, entity_text)
                    self.add_validated_example(entity_text, normalized_form)
    
    def _learn_position_pattern(self, sentence: str, entity_text: str) -> None:
        """Learn where entities typically appear in sentences."""
        words = sentence.split()
        total_words = len(words)
        entity_words = entity_text.split()
        
        # Find entity position
        for i in range(len(words) - len(entity_words) + 1):
            if ' '.join(words[i:i+len(entity_words)]).lower() == entity_text.lower():
                relative_pos = i / max(total_words - 1, 1)
                position = 'beginning' if relative_pos < 0.33 else 'middle' if relative_pos < 0.66 else 'end'
                self.patterns['position_patterns'][position] = self.patterns['position_patterns'].get(position, 0) + 1
                break
    
    def _learn_context_pattern(self, sentence: str, entity_text: str) -> None:
        """Learn what words typically appear before/after entities."""
        words = sentence.split()
        entity_words = entity_text.split()
        
        for i in range(len(words) - len(entity_words) + 1):
            if ' '.join(words[i:i+len(entity_words)]).lower() == entity_text.lower():
                # Learn context words (2 words before and after)
                before_words = words[max(0, i-2):i]
                after_words = words[i+len(entity_words):i+len(entity_words)+2]
                
                for word in before_words:
                    key = f"before:{word.lower()}"
                    self.patterns['context_patterns'][key] = self.patterns['context_patterns'].get(key, 0) + 1
                
                for word in after_words:
                    key = f"after:{word.lower()}"
                    self.patterns['context_patterns'][key] = self.patterns['context_patterns'].get(key, 0) + 1
                break
    
    def _apply_entity_specific_heuristics(self, text: str) -> float:
        """Apply configurable entity-specific heuristics."""
        score = 0.0
        text_upper = text.upper()
        
        # Check indicators
        indicators = self.entity_heuristics.get('indicators', [])
        indicator_weight = self.entity_heuristics.get('weights', {}).get('indicators', 0.3)
        for indicator in indicators:
            if indicator.upper() in text_upper:
                score += indicator_weight
        
        # Check known entities
        known_entities = self.entity_heuristics.get('known_entities', [])
        known_weight = self.entity_heuristics.get('weights', {}).get('known_entities', 0.4)
        for entity in known_entities:
            if entity.upper() in text_upper:
                score += known_weight
        
        # Check suffixes
        suffixes = self.entity_heuristics.get('suffixes', [])
        suffix_weight = self.entity_heuristics.get('weights', {}).get('suffixes', 0.2)
        for suffix in suffixes:
            if text_upper.endswith(suffix.upper()):
                score += suffix_weight
        
        return min(score, 1.0)


# Keep the GeneralizedBootstrapper as a simple alias for backward compatibility
class GeneralizedBootstrapper(ConfigurableBootstrapper):
    """Backward compatibility alias for ConfigurableBootstrapper."""
    pass


class PatternBasedChunker:
    """
    A chunker that uses learned patterns to intelligently split text for LLM processing.
    Preserves all information while tagging chunks based on entity patterns.
    """
    
    def __init__(self, learners: List[ConfigurableBootstrapper], max_chunk_size: int = 512):
        self.learners = learners
        self.max_chunk_size = max_chunk_size
        self.entity_patterns = {}
        
        # Consolidate patterns from all learners
        for learner in learners:
            self.entity_patterns[learner.entity_type] = {
                'word_patterns': learner.patterns['word_patterns'],
                'position_patterns': learner.patterns['position_patterns'],
                'context_patterns': learner.patterns['context_patterns'],
                'validated_examples': learner.validated_examples,
                'heuristics': learner.entity_heuristics
            }
    
    def chunk_text(self, text: str, overlap_size: int = 50) -> List[Dict[str, Any]]:
        """
        Chunk text based on learned patterns while preserving all information.
        
        Args:
            text: Text to chunk
            overlap_size: Number of characters to overlap between chunks
            
        Returns:
            List of chunks with pattern-based metadata
        """
        sentences = self._split_into_sentences(text)
        sentence_analysis = [self._analyze_sentence_patterns(sentence, i) 
                           for i, sentence in enumerate(sentences)]
        return self._create_pattern_aware_chunks(sentence_analysis, overlap_size)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving all content."""
        sentences = re.split(r'([.!?]+)', text)
        rejoined = []
        
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i].strip()
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]  # Add punctuation back
                if sentence:
                    rejoined.append(sentence)
        
        return rejoined
    
    def _analyze_sentence_patterns(self, sentence: str, sentence_idx: int) -> Dict[str, Any]:
        """Analyze a sentence for entity patterns from all learners."""
        analysis = {
            'sentence': sentence,
            'sentence_idx': sentence_idx,
            'length': len(sentence),
            'entity_matches': {},
            'pattern_strength': 0.0,
            'boundary_score': 0.0
        }
        
        words = sentence.split()
        total_pattern_strength = 0.0
        
        # Check patterns for each entity type
        for entity_type, patterns in self.entity_patterns.items():
            entity_matches = self._find_entity_patterns_in_sentence(sentence, words, patterns, entity_type)
            if entity_matches:
                analysis['entity_matches'][entity_type] = entity_matches
                for match in entity_matches:
                    total_pattern_strength += match['confidence']
        
        analysis['pattern_strength'] = total_pattern_strength
        analysis['boundary_score'] = self._calculate_boundary_score(analysis)
        
        return analysis
    
    def _find_entity_patterns_in_sentence(self, sentence: str, words: List[str], 
                                        patterns: Dict[str, Any], entity_type: str) -> List[Dict[str, Any]]:
        """Find entity pattern matches in a sentence."""
        matches = []
        
        # Check for multi-word patterns (1-4 words)
        for i in range(len(words)):
            for length in range(1, min(5, len(words) - i + 1)):
                candidate = ' '.join(words[i:i+length])
                confidence = self._calculate_pattern_confidence(candidate, sentence, words, i, patterns)
                
                if confidence > 0.3:
                    matches.append({
                        'text': candidate,
                        'confidence': confidence,
                        'position': i,
                        'length': length,
                        'entity_type': entity_type,
                        'context_before': ' '.join(words[max(0, i-2):i]),
                        'context_after': ' '.join(words[i+length:i+length+2])
                    })
        
        return self._remove_overlapping_matches(matches)
    
    def _calculate_pattern_confidence(self, candidate: str, sentence: str, 
                                    words: List[str], position: int, 
                                    patterns: Dict[str, Any]) -> float:
        """Calculate confidence based on learned patterns."""
        if candidate in patterns['validated_examples']:
            return 1.0
        
        # Word pattern score
        candidate_words = candidate.upper().split()
        word_score = sum(patterns['word_patterns'].get(word, 0) for word in candidate_words) / 10.0
        word_score = min(word_score / max(len(candidate_words), 1), 1.0)
        
        # Position pattern score
        relative_pos = position / max(len(words) - 1, 1)
        position_key = 'beginning' if relative_pos < 0.33 else 'middle' if relative_pos < 0.66 else 'end'
        position_score = min(patterns['position_patterns'].get(position_key, 0) / 10.0, 1.0)
        
        # Context pattern score
        before_words = words[max(0, position-2):position]
        after_words = words[position+len(candidate_words):position+len(candidate_words)+2]
        
        context_score = 0.0
        for word in before_words:
            context_score += patterns['context_patterns'].get(f"before:{word.lower()}", 0) / 10.0
        for word in after_words:
            context_score += patterns['context_patterns'].get(f"after:{word.lower()}", 0) / 10.0
        
        context_score = min(context_score, 1.0)
        
        # Configurable heuristics score
        heuristics_score = self._apply_configurable_heuristics(candidate, patterns.get('heuristics', {}))
        
        # Combined score
        return min(0.25 * word_score + 0.2 * position_score + 0.25 * context_score + 0.3 * heuristics_score, 1.0)
    
    def _apply_configurable_heuristics(self, text: str, heuristics: Dict[str, Any]) -> float:
        """Apply configurable heuristics from patterns."""
        score = 0.0
        text_upper = text.upper()
        
        # Check indicators
        indicators = heuristics.get('indicators', [])
        indicator_weight = heuristics.get('weights', {}).get('indicators', 0.3)
        for indicator in indicators:
            if indicator.upper() in text_upper:
                score += indicator_weight
        
        # Check known entities
        known_entities = heuristics.get('known_entities', [])
        known_weight = heuristics.get('weights', {}).get('known_entities', 0.4)
        for entity in known_entities:
            if entity.upper() in text_upper:
                score += known_weight
        
        # Check suffixes
        suffixes = heuristics.get('suffixes', [])
        suffix_weight = heuristics.get('weights', {}).get('suffixes', 0.2)
        for suffix in suffixes:
            if text_upper.endswith(suffix.upper()):
                score += suffix_weight
        
        return min(score, 1.0)
    
    def _remove_overlapping_matches(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove overlapping matches, keeping highest confidence."""
        if not matches:
            return []
        
        sorted_matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
        filtered: List[Dict[str, Any]] = []
        
        for match in sorted_matches:
            overlaps = any(self._matches_overlap(match, existing) for existing in filtered)
            if not overlaps:
                filtered.append(match)
        
        return filtered
    
    def _matches_overlap(self, match1: Dict[str, Any], match2: Dict[str, Any]) -> bool:
        """Check if two matches overlap."""
        start1, end1 = match1['position'], match1['position'] + match1['length']
        start2, end2 = match2['position'], match2['position'] + match2['length']
        return not (end1 <= start2 or end2 <= start1)
    
    def _calculate_boundary_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate how good this sentence is as a chunk boundary."""
        score = 0.0
        
        if analysis['entity_matches']:
            score += 0.5
        
        # Moderate sentences are better boundaries
        length = analysis['length']
        if 50 <= length <= 200:
            score += 0.3
        elif 20 <= length <= 300:
            score += 0.1
        
        score += analysis['pattern_strength'] * 0.2
        return min(score, 1.0)
    
    def _create_pattern_aware_chunks(self, sentence_analysis: List[Dict[str, Any]], 
                                   overlap_size: int) -> List[Dict[str, Any]]:
        """Create chunks based on pattern analysis while preserving all content."""
        chunks: List[Dict[str, Any]] = []
        current_chunk: Dict[str, Any] = {
            'content': '',
            'sentences': [],
            'entity_matches': {},
            'pattern_summary': {},
            'chunk_id': 0,
            'total_patterns': 0,
            'boundary_quality': 0.0
        }
        
        current_size = 0
        
        for analysis in sentence_analysis:
            sentence = analysis['sentence']
            sentence_size = len(sentence)
            
            # Check if adding this sentence would exceed max chunk size
            if (current_size + sentence_size > self.max_chunk_size and 
                current_chunk['content'] and 
                analysis['boundary_score'] > 0.3):
                
                self._finalize_chunk(current_chunk)
                chunks.append(current_chunk)
                
                # Start new chunk with overlap
                current_chunk = self._start_new_chunk_with_overlap(current_chunk, overlap_size, len(chunks))
                current_size = len(str(current_chunk['content']))
            
            # Add sentence to current chunk
            if current_chunk['content']:
                current_chunk['content'] += ' '
                current_size += 1
            
            current_chunk['content'] += sentence
            current_chunk['sentences'].append(analysis)
            current_size += sentence_size
            
            # Aggregate entity matches
            for entity_type, matches in analysis['entity_matches'].items():
                if entity_type not in current_chunk['entity_matches']:
                    current_chunk['entity_matches'][entity_type] = []
                current_chunk['entity_matches'][entity_type].extend(matches)
        
        # Add final chunk
        if current_chunk['content']:
            self._finalize_chunk(current_chunk)
            chunks.append(current_chunk)
        
        return chunks
    
    def _finalize_chunk(self, chunk: Dict[str, Any]) -> None:
        """Finalize chunk by calculating summary statistics."""
        total_patterns = sum(len(matches) for matches in chunk['entity_matches'].values())
        chunk['total_patterns'] = total_patterns
        
        pattern_summary = {}
        for entity_type, matches in chunk['entity_matches'].items():
            if matches:
                pattern_summary[entity_type] = {
                    'count': len(matches),
                    'avg_confidence': sum(m['confidence'] for m in matches) / len(matches),
                    'entities': [m['text'] for m in matches]
                }
        chunk['pattern_summary'] = pattern_summary
        
        if chunk['sentences']:
            boundary_scores = [s['boundary_score'] for s in chunk['sentences']]
            chunk['boundary_quality'] = sum(boundary_scores) / len(boundary_scores)
    
    def _start_new_chunk_with_overlap(self, prev_chunk: Dict[str, Any], 
                                    overlap_size: int, chunk_id: int) -> Dict[str, Any]:
        """Start new chunk with overlap from previous chunk."""
        new_chunk = {
            'content': '',
            'sentences': [],
            'entity_matches': {},
            'pattern_summary': {},
            'chunk_id': chunk_id,
            'total_patterns': 0,
            'boundary_quality': 0.0
        }
        
        if prev_chunk['content'] and overlap_size > 0:
            overlap_text = str(prev_chunk['content'])[-overlap_size:]
            new_chunk['content'] = overlap_text
        
        return new_chunk
    
    def get_chunking_stats(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about the chunking results."""
        if not chunks:
            return {}
        
        total_entities = sum(chunk['total_patterns'] for chunk in chunks)
        total_sentences = sum(len(chunk['sentences']) for chunk in chunks)
        
        entity_distribution = {}
        for chunk in chunks:
            for entity_type, summary in chunk['pattern_summary'].items():
                entity_distribution[entity_type] = entity_distribution.get(entity_type, 0) + summary['count']
        
        return {
            'total_chunks': len(chunks),
            'total_entities_found': total_entities,
            'total_sentences': total_sentences,
            'avg_entities_per_chunk': total_entities / len(chunks),
            'avg_sentences_per_chunk': total_sentences / len(chunks),
            'entity_distribution': entity_distribution,
            'avg_chunk_size': sum(len(chunk['content']) for chunk in chunks) / len(chunks),
            'avg_boundary_quality': sum(chunk['boundary_quality'] for chunk in chunks) / len(chunks)
        }


def demo_configurable_chunking():
    """Demo function showing configurable pattern-based chunking."""
    
    # Create and train broker learner (will attempt to load from config files)
    broker_learner = ConfigurableBootstrapper('broker', config_dir='configs')
    
    # Since we don't have config files yet, add some training data programmatically
    training_paragraphs = [
        "The client opened an account with ZERODHA for equity trading. ZERODHA provides discount brokerage services.",
        "HDFC SECURITIES offers comprehensive wealth management services. The portfolio was managed through HDFC SECURITIES platform.",
        "Investment advisory services were provided by ICICI DIRECT. The client transferred funds to ICICI DIRECT for investment.",
        "KOTAK SECURITIES executed the transaction on behalf of the client. The commission charged by KOTAK SECURITIES was competitive.",
        "Angel Broking Limited assisted with IPO applications. The client received confirmation from Angel Broking Limited."
    ]
    
    known_brokers = {
        "ZERODHA": "ZERODHA",
        "HDFC SECURITIES": "HDFC SECURITIES", 
        "ICICI DIRECT": "ICICI DIRECT",
        "KOTAK SECURITIES": "KOTAK SECURITIES",
        "Angel Broking Limited": "ANGEL BROKING LIMITED"
    }
    
    # Train the learner
    for paragraph in training_paragraphs:
        broker_learner.learn_from_paragraph(paragraph, known_brokers)
    
    print("=== Training Complete ===")
    print(f"Position patterns: {broker_learner.patterns['position_patterns']}")
    print(f"Loaded {len(broker_learner.validated_examples)} examples")
    print()
    
    # Save learned patterns for future use
    broker_learner.save_patterns_to_file()
    broker_learner.save_examples_to_file()
    
    # Create chunker
    chunker = PatternBasedChunker([broker_learner], max_chunk_size=200)
    
    # Test document
    test_document = """
    The investment portfolio review was conducted by SBI SECURITIES last month. 
    The client wanted to diversify holdings and consulted with SBI SECURITIES advisor. 
    Additionally, Sharekhan provided market analysis for the technology sector. 
    Upstox was used for executing the trades at lower brokerage costs. 
    HDFC SECURITIES managed the debt portfolio with conservative strategies.
    """
    
    print("=== Test Document ===")
    print(test_document.strip())
    print()
    
    # Chunk the document
    chunks = chunker.chunk_text(test_document)
    
    print("=== Chunking Results ===")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} ---")
        print(f"Content: {chunk['content'][:100]}...")
        print(f"Length: {len(chunk['content'])} characters")
        print(f"Total patterns found: {chunk['total_patterns']}")
        print(f"Boundary quality: {chunk['boundary_quality']:.2f}")
        
        if chunk['pattern_summary']:
            print("Pattern Summary:")
            for entity_type, summary in chunk['pattern_summary'].items():
                print(f"  {entity_type}: {summary['count']} entities (avg confidence: {summary['avg_confidence']:.2f})")
                print(f"    Entities: {', '.join(summary['entities'])}")
        print()
    
    # Show statistics
    stats = chunker.get_chunking_stats(chunks)
    print("=== Chunking Statistics ===")
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    demo_configurable_chunking()
