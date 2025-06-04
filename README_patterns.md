# Configurable Pattern-Based Learning System

This system provides a flexible, file-based configuration approach for learning entity patterns from text data and performing intelligent chunking for LLM processing.

## Features

- **Configurable Entity Types**: Load patterns and examples from external files
- **Pattern Learning**: Learn position, context, and word patterns from training data
- **Intelligent Chunking**: Pattern-aware text chunking that preserves information
- **Semantic Chunking**: Additional similarity-based chunking using sentence transformers
- **No Hardcoded Data**: All entity patterns loaded from configuration files

## Configuration Structure

The system uses a `configs/` directory with the following file structure:

```
configs/
├── {entity_type}_examples.txt       # Validated entity examples
├── {entity_type}_heuristics.json    # Entity-specific detection rules
└── {entity_type}_patterns.json      # Learned patterns (auto-generated)
```

### Examples File Format (`{entity_type}_examples.txt`)

```
# Validated examples for {entity_type}
# Format: original_text|normalized_form

ZERODHA
UPSTOX
ANGEL BROKING|ANGEL BROKING LIMITED
ICICI DIRECT
HDFC SECURITIES
```

- Lines starting with `#` are comments
- Use `|` to separate original text from normalized form
- If no `|` is present, original text is used as normalized form

### Heuristics File Format (`{entity_type}_heuristics.json`)

```json
{
  "indicators": [
    "SECURITIES",
    "BROKING",
    "CAPITAL"
  ],
  "suffixes": [
    "LTD",
    "LIMITED",
    "PVT"
  ],
  "known_entities": [
    "ZERODHA",
    "UPSTOX",
    "HDFC"
  ],
  "weights": {
    "indicators": 0.4,
    "suffixes": 0.3,
    "known_entities": 0.5
  }
}
```

- **indicators**: Words that suggest the entity type
- **suffixes**: Common endings for this entity type
- **known_entities**: Partial matches for known entities
- **weights**: Importance weights for each heuristic type

## Usage Examples

### Basic Pattern Learning

```python
from L1_learner import ConfigurableBootstrapper

# Create learner that loads from configs/broker_* files
broker_learner = ConfigurableBootstrapper('broker', config_dir='configs')

# Train on paragraph data
training_paragraph = "The client opened an account with ZERODHA for trading."
known_entities = {"ZERODHA": "ZERODHA"}
broker_learner.learn_from_paragraph(training_paragraph, known_entities)

# Save learned patterns
broker_learner.save_patterns_to_file()
broker_learner.save_examples_to_file()
```

### Pattern-Based Chunking

```python
from L1_learner import ConfigurableBootstrapper, PatternBasedChunker

# Create multiple learners for different entity types
broker_learner = ConfigurableBootstrapper('broker')
company_learner = ConfigurableBootstrapper('company')

# Create chunker with all learners
chunker = PatternBasedChunker([broker_learner, company_learner], max_chunk_size=200)

# Chunk text intelligently
document = "SBI SECURITIES handled the portfolio. RELIANCE INDUSTRIES performed well."
chunks = chunker.chunk_text(document)

# Get chunking statistics
stats = chunker.get_chunking_stats(chunks)
print(f"Total entities found: {stats['total_entities_found']}")
```

### Multiple Entity Types

```python
# Each entity type has its own configuration files
broker_learner = ConfigurableBootstrapper('broker')     # Uses configs/broker_*
company_learner = ConfigurableBootstrapper('company')   # Uses configs/company_*
drug_learner = ConfigurableBootstrapper('drug')         # Uses configs/drug_*

# Combine for comprehensive analysis
all_learners = [broker_learner, company_learner, drug_learner]
chunker = PatternBasedChunker(all_learners)
```

## Pre-Configured Entity Types

The system comes with sample configurations for:

### Broker Names
- **Examples**: ZERODHA, UPSTOX, HDFC SECURITIES, etc.
- **Indicators**: SECURITIES, BROKING, CAPITAL, TRADING
- **Suffixes**: LTD, LIMITED, PVT, SECURITIES

### Company Names  
- **Examples**: RELIANCE INDUSTRIES, TCS LIMITED, etc.
- **Indicators**: INDUSTRIES, CORPORATION, BANK, FINANCE
- **Suffixes**: LTD, LIMITED, INC, CORP, PVT

## Adding New Entity Types

1. **Create Examples File**: `configs/{entity_type}_examples.txt`
   ```
   # Validated examples for drug
   # Format: original_text|normalized_form
   
   Aspirin|ASPIRIN
   Paracetamol|PARACETAMOL
   ```

2. **Create Heuristics File**: `configs/{entity_type}_heuristics.json`
   ```json
   {
     "indicators": ["TABLET", "CAPSULE", "SYRUP"],
     "suffixes": ["MG", "ML"],
     "known_entities": ["ASPIRIN", "PARACETAMOL"],
     "weights": {
       "indicators": 0.3,
       "suffixes": 0.2,
       "known_entities": 0.5
     }
   }
   ```

3. **Use in Code**:
   ```python
   drug_learner = ConfigurableBootstrapper('drug')
   ```

## Pattern Learning Process

1. **Load Configuration**: System loads examples and heuristics from files
2. **Learn from Training Data**: Extract position, context, and word patterns
3. **Apply Heuristics**: Use configurable rules for entity detection
4. **Save Patterns**: Automatically save learned patterns for future use
5. **Intelligent Chunking**: Use patterns to guide text segmentation

## Advanced Features

### Custom Configuration Directory
```python
learner = ConfigurableBootstrapper('broker', config_dir='custom_configs')
```

### Pattern Persistence
```python
# Patterns are automatically saved during training
learner.save_patterns_to_file('custom_path/patterns.json')
learner.save_examples_to_file('custom_path/examples.txt')
```

### Chunking Statistics
```python
stats = chunker.get_chunking_stats(chunks)
# Returns: total_chunks, entities_found, boundary_quality, etc.
```

## Benefits

- **No Hardcoding**: All entity data comes from external files
- **Easy Customization**: Modify patterns without changing code
- **Scalable**: Add new entity types by creating configuration files
- **Persistent Learning**: Patterns improve over time and are saved automatically
- **Domain Adaptable**: Configure for any domain (finance, medical, legal, etc.)
- **Quality Metrics**: Track chunking quality and entity detection confidence 