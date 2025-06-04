# Mutual Fund Scheme Matcher

A comprehensive system for detecting, normalizing, and managing mutual fund scheme codes in Excel spreadsheets.

## Overview

This library provides intelligent detection and normalization of mutual fund scheme codes, even when they appear as natural text. It handles various challenges such as abbreviations, legacy fund names, and inconsistent formatting to provide accurate scheme code normalization.

The system works without requiring an initial knowledge base, and improves over time through incremental learning and knowledge accumulation.

## Why Use This System When You Already Have an API?

While you might have an API that returns normalized scheme codes, this system adds significant value:

### 1. Efficiency and Cost Reduction
- **Reduced API Calls**: Every API call has a cost (financial, time, rate limits). This system reduces calls by up to 90% through caching and smart detection.
- **Selective Processing**: Only cells that likely contain scheme codes are sent to the API, not every cell in your spreadsheet.

### 2. Handling Complex Cases
- **Legacy Name Resolution**: Automatically handles texts like "HDFC Top 100 Fund (formerly Top 200 Fund)" by understanding the context.
- **Ambiguous Text**: Processes natural language references to schemes that direct API calls might miss.

### 3. Learning and Adaptation
- **Continuous Improvement**: The system learns from each job, becoming more accurate and efficient over time.
- **Customization**: Adapts to your specific data patterns rather than providing a one-size-fits-all solution.

### 4. Resilience and Reliability
- **Offline Capability**: Continues functioning using its knowledge base even when the API is temporarily unavailable.
- **Consistency**: Provides stable results even if the underlying API changes.

### 5. Context Awareness
- **Document Structure Understanding**: Recognizes where scheme codes are likely to appear in your Excel files.
- **Domain-Specific Knowledge**: Incorporates financial domain knowledge about mutual fund naming conventions.

### 6. Quality Control
- **Confidence Scoring**: Provides confidence levels for each match, allowing you to focus verification on uncertain cases.
- **Comprehensive Audit Trail**: Tracks which detection method was used for each match.

### 7. Integration with Workflows
- **End-to-End Processing**: Integrates scheme detection directly into your data processing pipeline.
- **Correction Feedback Loop**: Learns from corrections to improve future performance.

In essence, this system serves as an intelligent middleware layer between your data and the normalization API, adding value through efficiency, intelligence, resilience, and adaptation.

## Features

- **Pattern-based Detection**: Identifies potential scheme codes using regex patterns
- **Component-based Parsing**: Breaks down scheme names into structural components
- **Legacy Name Resolution**: Handles cases where both current and former names are mentioned
- **Incremental Learning**: Improves detection accuracy over time
- **Knowledge Base Building**: Automatically builds a knowledge base from successful matches
- **Confidence Scoring**: Provides confidence levels for each match
- **AMC-specific Matching**: Optimizes detection for specific Asset Management Companies
- **Header Detection**: Focuses on header rows where scheme names typically appear

## Installation

```bash
pip install mutual-fund-scheme-matcher
```

## Getting Started with Multiple Files and Sheets

When processing multiple Excel files with multiple sheets, a structured approach helps ensure consistent and efficient processing. Here's a step-by-step guide to get started:

### 1. Set Up Your Environment

First, set up your basic environment:

```python
from scheme_matcher import ComprehensiveSchemeMatcher
import pandas as pd
import os
import glob

# Configure the matcher once
matcher = ComprehensiveSchemeMatcher(
    amc_name="HDFC",  # Set the AMC name
    api_endpoint="https://your-api-endpoint.com/normalize",
    cache_file="scheme_cache.csv",  # Enable caching for efficiency
    knowledge_base_path="knowledge_base.json"  # Enable learning
)
```

### 2. Process Multiple Files and Sheets

Create a function to process a directory of Excel files:

```python
def process_excel_directory(directory_path, output_directory):
    """Process all Excel files in a directory"""
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    
    # Get all Excel files
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    # Process statistics
    total_stats = {
        'files_processed': 0,
        'sheets_processed': 0,
        'total_cells': 0,
        'normalized_codes': 0,
        'api_calls': 0
    }
    
    # Process each file
    for file_path in excel_files:
        file_name = os.path.basename(file_path)
        print(f"Processing file: {file_name}")
        
        try:
            # Read the Excel file (don't read data yet)
            excel = pd.ExcelFile(file_path)
            
            # Create a writer for the output file
            output_path = os.path.join(output_directory, f"normalized_{file_name}")
            with pd.ExcelWriter(output_path) as writer:
                
                # Process each sheet
                for sheet_name in excel.sheet_names:
                    print(f"  Processing sheet: {sheet_name}")
                    
                    # Read the sheet
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # Skip empty sheets
                    if df.empty:
                        print(f"  Sheet '{sheet_name}' is empty. Skipping.")
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                        continue
                    
                    # Process the sheet
                    normalized_df, matches, stats = matcher.match(df)
                    
                    # Update stats
                    total_stats['sheets_processed'] += 1
                    total_stats['total_cells'] += stats.get('total_cells', 0)
                    total_stats['normalized_codes'] += stats.get('normalized_codes', 0)
                    total_stats['api_calls'] += stats.get('api_calls', 0)
                    
                    # Write the normalized sheet to the output file
                    normalized_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Optional: Save matches for this sheet
                    # save_matches(matches, output_directory, f"{file_name}_{sheet_name}_matches.json")
            
            total_stats['files_processed'] += 1
            print(f"Completed processing file: {file_name}")
            
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    return total_stats

# Optional: Function to save matches for further analysis
def save_matches(matches, directory, filename):
    """Save matches to a JSON file for further analysis"""
    import json
    
    # Convert to serializable format
    serializable_matches = []
    for match in matches:
        serializable_match = {k: v for k, v in match.items()}
        serializable_matches.append(serializable_match)
    
    # Save to file
    with open(os.path.join(directory, filename), 'w') as f:
        json.dump(serializable_matches, f, indent=2)
```

### 3. Run the Processing

Execute the processing function:

```python
# Process all Excel files in a directory
stats = process_excel_directory(
    directory_path="path/to/excel/files",
    output_directory="path/to/output"
)

# Print summary statistics
print("\nProcessing Summary:")
print(f"Files processed: {stats['files_processed']}")
print(f"Sheets processed: {stats['sheets_processed']}")
print(f"Total cells analyzed: {stats['total_cells']}")
print(f"Normalized codes found: {stats['normalized_codes']}")
print(f"API calls made: {stats['api_calls']}")
```

### 4. Incremental Approach for Large Datasets

For very large datasets, consider an incremental approach:

```python
# Process files in smaller batches
def process_in_batches(directory_path, output_directory, batch_size=5):
    """Process files in batches to manage memory and allow for checkpointing"""
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    # Process in batches
    for i in range(0, len(excel_files), batch_size):
        batch = excel_files[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(excel_files) + batch_size - 1)//batch_size}")
        
        # Process each file in the batch
        for file_path in batch:
            # Process file (implementation similar to above)
            pass
        
        # Save knowledge base after each batch to preserve learning
        # This ensures learning is not lost if the process is interrupted
        if hasattr(matcher, 'base_matcher') and hasattr(matcher.base_matcher, 'incremental'):
            matcher.base_matcher.incremental.save_knowledge_base()
```

### 5. Parallel Processing

For even faster processing, you can use parallel execution:

```python
from concurrent.futures import ProcessPoolExecutor
import functools

def process_file(file_path, output_directory, amc_name, api_endpoint, 
                 cache_file, knowledge_base_path):
    """Process a single file - can be called in parallel"""
    # Create a new matcher instance for this process
    matcher = ComprehensiveSchemeMatcher(
        amc_name=amc_name,
        api_endpoint=api_endpoint,
        cache_file=cache_file,
        knowledge_base_path=knowledge_base_path
    )
    
    # Process file (implementation as above)
    # ...
    
    return file_stats

def process_directory_parallel(directory_path, output_directory, max_workers=4):
    """Process directory using parallel execution"""
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    # Create a partial function with fixed arguments
    process_func = functools.partial(
        process_file,
        output_directory=output_directory,
        amc_name="HDFC",
        api_endpoint="https://your-api-endpoint.com/normalize",
        cache_file="scheme_cache.csv",
        knowledge_base_path="knowledge_base.json"
    )
    
    # Process files in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_func, excel_files))
    
    # Aggregate results
    total_stats = {
        'files_processed': len(results),
        'sheets_processed': sum(r.get('sheets_processed', 0) for r in results),
        'total_cells': sum(r.get('total_cells', 0) for r in results),
        'normalized_codes': sum(r.get('normalized_codes', 0) for r in results),
        'api_calls': sum(r.get('api_calls', 0) for r in results)
    }
    
    return total_stats
```

### 6. Production Setup Considerations

For a production setup, consider these additional steps:

1. **Monitoring and Logging**:
   ```python
   import logging
   
   # Configure logging
   logging.basicConfig(
       level=logging.INFO,
       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
       filename='scheme_matcher.log'
   )
   
   # Add logging throughout the process
   logging.info(f"Processing file: {file_name}")
   ```

2. **Error Handling and Retries**:
   ```python
   from tenacity import retry, stop_after_attempt, wait_exponential
   
   @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
   def call_api_with_retry(text, amc_name):
       # Implementation with retries for API calls
       pass
   ```

3. **Progress Tracking**:
   ```python
   from tqdm import tqdm
   
   for file_path in tqdm(excel_files, desc="Processing files"):
       # Processing code
       pass
   ```

4. **Database Integration**:
   ```python
   import sqlite3
   
   # Store results in a database
   conn = sqlite3.connect('scheme_matcher_results.db')
   # Create tables and store results
   ```

By following this structured approach, you can efficiently process large numbers of Excel files with multiple sheets while building up your knowledge base and improving accuracy over time.

## Working Without Initial Knowledge Base

The system is designed to work effectively even without an initial knowledge base:

1. **Basic Pattern Matching**: It starts with pattern-based detection to identify potential scheme codes
2. **API-based Normalization**: Uses your normalization API endpoint to standardize codes
3. **Automatic Caching**: Caches successful normalizations to avoid repeated API calls
4. **Incremental Knowledge Building**: Learns from high-confidence matches to improve subsequent detection

## Knowledge Base Building Process

The system automatically builds and expands its knowledge base through several mechanisms:

### 1. Caching

- Successful API normalizations are cached to a CSV file (if `cache_file` is provided)
- This cache persists between runs and reduces API calls for repeated scheme codes

```python
matcher = ComprehensiveSchemeMatcher(
    amc_name="HDFC",
    api_endpoint="https://your-api-endpoint.com/normalize",
    cache_file="scheme_cache.csv"  # Enable caching
)
```

### 2. Incremental Learning

- The `IncrementalSchemeDetector` learns from high-confidence matches
- It builds a knowledge base that includes:
  - Transformation patterns between original text and normalized codes
  - Synonyms and common abbreviations
  - Important words and patterns that indicate scheme codes

```python
matcher = ComprehensiveSchemeMatcher(
    amc_name="HDFC",
    api_endpoint="https://your-api-endpoint.com/normalize",
    knowledge_base_path="knowledge_base.json"  # Enable incremental learning
)
```

### 3. Bootstrapping

- The `SchemeBootstrapper` learns patterns from known scheme matches
- It uses these patterns to identify similar scheme codes elsewhere in the document
- This helps extend coverage to scheme codes with similar formats

### 4. Adding External Knowledge

You can also provide external knowledge through a scheme database:

```python
matcher = ComprehensiveSchemeMatcher(
    amc_name="HDFC",
    api_endpoint="https://your-api-endpoint.com/normalize",
    scheme_db_path="schemes_database.csv"  # External scheme database
)
```

The scheme database should be a CSV file with at least these columns:
- `amc`: The AMC name
- `scheme_code`: The normalized scheme code
- `scheme_name`: The scheme name

## Knowledge Evolution Workflow

1. **Initial Run**:
   - Run without knowledge base
   - System relies on pattern matching and API calls
   - Successful matches are cached and learned from

2. **Subsequent Runs**:
   - System uses accumulated knowledge for better matching
   - Fewer API calls needed as cache grows
   - Knowledge base expands with each new successful match

3. **Continuous Improvement**:
   - Manual correction of any incorrect matches
   - Adding those corrections to the system
   - Periodic review of the knowledge base

## Advanced Usage

### Processing Uncertain Cases

```python
# Enable transformer-based processing for uncertain cases
matcher = ComprehensiveSchemeMatcher(
    amc_name="HDFC",
    api_endpoint="https://your-api-endpoint.com/normalize",
    use_transformer=True,
    transformer_model_path="path/to/transformer/model"  # Optional
)
```

### Adding Custom Correction Rules

```python
# Get the corrector from the matcher
corrector = matcher.base_matcher.corrector

# Add custom correction rules
corrector.add_correction_rule(r'\bHDFC-TF\b', 'HDFC TOP 100 FUND')
corrector.add_correction_rule(r'\bHDFC-MEF\b', 'HDFC MID-CAP OPPORTUNITIES FUND')
```

### Manual Knowledge Base Updates

You can manually add entries to the knowledge base:

```python
# Get the incremental detector
incremental = matcher.base_matcher.incremental

# Add a high-confidence match
incremental.learn_from_match(
    original_text="HDFC Top100 Dir Gr",
    normalized_code="HDFC_TOP100_DIR_GR",
    confidence=0.95
)

# Save the updates
incremental.save_knowledge_base()
```

## API Integration

The system is designed to work with an external API for normalizing scheme codes. The API should:

1. Accept POST requests with JSON payload:
   ```json
   {
     "amc_name": "HDFC",
     "partial_code": "HDFC Top 100 Fund Direct Growth"
   }
   ```

2. Return JSON response:
   ```json
   {
     "normalized_code": "HDFC_TOP100_DIR_GR",
     "confidence": 0.92
   }
   ```

For testing without an API, the system includes a mock implementation.

## Performance Considerations

- Process Excel files in batches if they are very large
- Enable caching to reduce API calls
- Use knowledge base to improve accuracy over time
- Focus on header rows (typically in the first 10-15 rows)

## Best Practices

1. **Start Simple**: Begin with basic matching before enabling advanced features
2. **Monitor Confidence**: Review matches with low confidence scores
3. **Incremental Adoption**: Enable additional components as needed
4. **Regular Maintenance**: Periodically clean up the knowledge base and cache
5. **API Fallback**: Always maintain the API fallback for unknown schemes

## Customization

The system is designed to be highly modular. You can customize:

- Pattern detection rules
- Confidence thresholds
- Header detection logic
- Component parsing
- Knowledge base structure

## Troubleshooting

### Common Issues

1. **Low Detection Rate**:
   - Check if AMC name is correctly specified
   - Review pattern detection rules
   - Ensure scheme codes are in the expected format

2. **Incorrect Normalizations**:
   - Check API responses
   - Review knowledge base for conflicting entries (see section below)
   - Adjust confidence thresholds

3. **Slow Processing**:
   - Enable caching
   - Limit processing to header rows
   - Use batch processing for large files

## Managing Knowledge Base Conflicts

As your knowledge base grows automatically through incremental learning, it might accumulate conflicting information that can lead to incorrect normalizations. Regular maintenance is important to ensure consistent and accurate results.

### Types of Conflicts

1. **Same Original Text, Different Normalized Codes**: 
   - Example: "HDFC Balanced Fund" → "HDFC_BAL_REG_GR" vs "HDFC_HYBRID_EQ_REG_GR"
   - Possible causes: Scheme renaming, inconsistent API responses, manual corrections

2. **Ambiguous Abbreviations**:
   - Example: "HDFC BF" could refer to "HDFC Balanced Fund", "HDFC Banking Fund", or "HDFC Bluechip Fund"
   - Requires disambiguation rules

3. **Conflicting Transformation Patterns**:
   - Contradictory learning about how to transform texts to codes
   - Can cause unpredictable behavior

4. **Outdated Legacy Mappings**:
   - When funds undergo multiple name changes over time
   - Historical mappings may need updates

### How to Review the Knowledge Base

You can examine the knowledge base file for potential conflicts:

```python
import json
from fuzzywuzzy import fuzz

# Load the knowledge base
with open("knowledge_base.json", "r") as f:
    kb = json.load(f)

# Check the 'examples' section for potential conflicts
examples = kb['examples']

# Find similar original texts with different normalized codes
conflicts = []
checked = set()

for text1, data1 in examples.items():
    if text1 in checked:
        continue
    
    for text2, data2 in examples.items():
        if text1 == text2 or text2 in checked:
            continue
            
        # Check if texts are similar but have different normalized codes
        similarity = fuzz.ratio(text1.lower(), text2.lower())
        if similarity > 80 and data1['normalized'] != data2['normalized']:
            conflicts.append((text1, data1['normalized'], text2, data2['normalized'], similarity))
    
    checked.add(text1)

# Print detected conflicts
for text1, norm1, text2, norm2, sim in conflicts:
    print(f"Potential conflict detected: {sim}% similarity")
    print(f"  '{text1}' → {norm1}")
    print(f"  '{text2}' → {norm2}")
    print()
```

### Resolving Conflicts

#### 1. Manual Resolution

Remove or update incorrect entries:

```python
# Remove an incorrect entry
if "HDFC Balanced Fund" in kb['examples']:
    del kb['examples']["HDFC Balanced Fund"]

# Add the correct entry
kb['examples']["HDFC Balanced Fund"] = {
    "normalized": "HDFC_HYBRID_EQ_REG_GR",
    "confidence": 0.95
}

# Save the updated knowledge base
with open("knowledge_base.json", "w") as f:
    json.dump(kb, f, indent=2)
```

#### 2. Add Explicit Correction Rules

For highly ambiguous cases:

```python
# Get the corrector from the matcher
corrector = matcher.base_matcher.corrector

# Add explicit correction rules for ambiguous abbreviations
corrector.add_correction_rule(r'\bHDFC BF\b', 'HDFC BALANCED FUND')
corrector.add_correction_rule(r'\bHDFC PEF\b', 'HDFC PHARMA EQUITY FUND')
```

#### 3. Update Legacy Mappings

Maintain accurate legacy-to-current name mappings:

```python
# Get the scheme parser
parser = matcher.base_matcher.parser

# Update legacy mappings
parser.legacy_mappings.update({
    'Balanced Fund': 'Hybrid Equity Fund',
    'Top 200 Fund': 'Top 100 Fund'
})
```

### Best Practices for Knowledge Base Maintenance

1. **Regular Audit**: Schedule periodic reviews of the knowledge base
2. **Versioning**: Keep backups or versions of the knowledge base file
3. **Confidence Thresholds**: Only learn from matches with high confidence (>0.85)
4. **Selective Learning**: Be more selective about which matches to learn from
5. **AMC Isolation**: Maintain separate knowledge bases for different AMCs
6. **Documentation**: Keep notes about manual corrections and known issues
7. **Testing**: Create test cases that verify correct handling of ambiguous cases

By regularly maintaining your knowledge base, you can ensure that the system continues to provide accurate and consistent normalizations as it learns and evolves.

### Diagnostic Tools

The system provides detailed statistics:

```python
normalized_df, matches, stats = matcher.match(df)
print(stats)
```

Example output:
```
{
  'total_cells': 1000,
  'potential_schemes': 25,
  'api_calls': 10,
  'cached_hits': 15,
  'normalized_codes': 20,
  'enhanced_matches': 5,
  'uncertain_cases': 8,
  'uncertain_resolved': 3
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
