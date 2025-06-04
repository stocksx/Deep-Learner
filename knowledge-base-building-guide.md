# Building a Reliable Knowledge Base from Scratch

When starting with a brand new dataset and no existing knowledge base, it's critical to establish a solid foundation. This guide outlines a systematic approach to building a reliable knowledge base from the beginning, with special attention to handling potentially incorrect API responses.

## Initial Setup Phase: Validation-First Approach

### Step 1: Small-Scale Validation

Start by processing a small, representative sample to validate API responses:

```python
import pandas as pd
from scheme_matcher import ComprehensiveSchemeMatcher

# Create a matcher WITHOUT knowledge base or caching initially
matcher = ComprehensiveSchemeMatcher(
    amc_name="HDFC",
    api_endpoint="https://your-api-endpoint.com/normalize",
    # No knowledge_base_path or cache_file yet
)

# Process a small sample file or sheet
sample_df = pd.read_excel('sample_portfolio.xlsx')
normalized_df, matches, stats = matcher.match(sample_df)

# Export matches for manual verification
import json
with open('initial_matches_for_verification.json', 'w') as f:
    # Convert matches to a serializable format
    serializable_matches = []
    for match in matches:
        serializable_match = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                             for k, v in match.items()}
        serializable_matches.append(serializable_match)
    json.dump(serializable_matches, f, indent=2)

print(f"Found {len(matches)} potential scheme matches for verification")
```

### Step 2: Manual Verification

Manually review the exported matches to verify API response accuracy:

1. Open `initial_matches_for_verification.json` in a text editor or JSON viewer
2. Review each match, comparing `original_text` with `normalized_code`
3. Flag any incorrect mappings

Example verification workflow:
```python
# Create a verification dataframe for easier review
import pandas as pd

# Load the matches
with open('initial_matches_for_verification.json', 'r') as f:
    matches = json.load(f)

# Create a verification dataframe
verification_df = pd.DataFrame([
    {
        'original_text': m['original_text'],
        'normalized_code': m['normalized_code'],
        'confidence': m['confidence'],
        'is_correct': True,  # Default to True, change manually during review
        'correct_code': m['normalized_code']  # Fill in correct code if original is wrong
    }
    for m in matches
])

# Export to Excel for manual verification
verification_df.to_excel('verification_sheet.xlsx', index=False)

# After manual verification, read back the updated sheet
verified_df = pd.read_excel('verification_sheet.xlsx')
```

### Step 3: Create a Verified Seed Knowledge Base

Build an initial knowledge base using only verified correct mappings:

```python
# After manual verification, read back the updated sheet
verified_df = pd.read_excel('verification_sheet.xlsx')

# Filter out incorrect mappings
correct_mappings = verified_df[verified_df['is_correct'] == True]
corrected_mappings = verified_df[verified_df['is_correct'] == False]

# Create a new matcher with knowledge base path
matcher_with_kb = ComprehensiveSchemeMatcher(
    amc_name="HDFC",
    api_endpoint="https://your-api-endpoint.com/normalize",
    knowledge_base_path="verified_knowledge_base.json"
)

# Initialize the knowledge base with verified correct mappings
for _, row in correct_mappings.iterrows():
    matcher_with_kb.base_matcher.incremental.learn_from_match(
        original_text=row['original_text'],
        normalized_code=row['normalized_code'],
        confidence=0.95  # High confidence for verified entries
    )

# Add corrected mappings
for _, row in corrected_mappings.iterrows():
    matcher_with_kb.base_matcher.incremental.learn_from_match(
        original_text=row['original_text'],
        normalized_code=row['correct_code'],  # Use manually corrected code
        confidence=0.95  # High confidence for verified entries
    )

# Save the verified knowledge base
matcher_with_kb.base_matcher.incremental.save_knowledge_base()
```

### Step 4: Create Correction Rules for Systematic Errors

If you identify patterns in API errors, add explicit correction rules:

```python
# Add correction rules for any patterns of API errors
matcher_with_kb.base_matcher.corrector.add_correction_rule(
    r'\bHDFC Top 200\b', 'HDFC TOP 100 FUND'  # If the API consistently makes this error
)

# For cases where the API might give different codes for the same fund
legacy_resolver = matcher_with_kb.legacy_resolver
legacy_resolver.legacy_mappings.update({
    'HDFC BALANCED FUND': 'HDFC HYBRID EQUITY FUND'  # If API doesn't handle legacy names
})
```

## Controlled Growth Phase

### Step 5: Expand with Confidence Filtering

Process more data while filtering by confidence to avoid incorporating errors:

```python
def process_with_confidence_filtering(file_path, confidence_threshold=0.8):
    """Process a file but only incorporate high-confidence matches into knowledge base"""
    df = pd.read_excel(file_path)
    normalized_df, matches, _ = matcher_with_kb.match(df)
    
    # Split matches by confidence
    high_confidence = [m for m in matches if m['confidence'] >= confidence_threshold]
    low_confidence = [m for m in matches if m['confidence'] < confidence_threshold]
    
    print(f"High confidence matches: {len(high_confidence)}")
    print(f"Low confidence matches: {len(low_confidence)}")
    
    # Export low confidence matches for possible review
    with open('low_confidence_for_review.json', 'w') as f:
        serializable_matches = []
        for match in low_confidence:
            serializable_match = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                                for k, v in match.items()}
            serializable_matches.append(serializable_match)
        json.dump(serializable_matches, f, indent=2)
    
    return normalized_df
```

### Step 6: Establish Feedback Loop

Create a system for ongoing verification and correction:

```python
def apply_feedback(feedback_file):
    """Incorporate feedback into the knowledge base"""
    # Read feedback file
    feedback_df = pd.read_excel(feedback_file)
    
    # Process corrections
    for _, row in feedback_df.iterrows():
        if row['action'] == 'correct':
            # Remove incorrect mapping if it exists
            if row['original_text'] in matcher_with_kb.base_matcher.incremental.knowledge_base['examples']:
                del matcher_with_kb.base_matcher.incremental.knowledge_base['examples'][row['original_text']]
            
            # Add correct mapping
            matcher_with_kb.base_matcher.incremental.learn_from_match(
                original_text=row['original_text'],
                normalized_code=row['correct_code'],
                confidence=0.95
            )
        elif row['action'] == 'add_rule':
            # Add correction rule
            matcher_with_kb.base_matcher.corrector.add_correction_rule(
                row['pattern'],
                row['replacement']
            )
    
    # Save updated knowledge base
    matcher_with_kb.base_matcher.incremental.save_knowledge_base()
    print("Feedback applied and knowledge base updated")
```

## Scaling Phase

### Step 7: Process Multiple Files with Verification Sampling

As you scale up, implement random sampling for ongoing verification:

```python
import random

def process_with_verification_sampling(directory_path, sampling_rate=0.05):
    """Process files with random sampling for verification"""
    excel_files = glob.glob(os.path.join(directory_path, "*.xlsx"))
    
    # Choose random files for verification
    verification_files = random.sample(excel_files, max(1, int(len(excel_files) * sampling_rate)))
    
    # Process all files
    for file_path in excel_files:
        df = pd.read_excel(file_path)
        normalized_df, matches, _ = matcher_with_kb.match(df)
        
        # If this is a verification file, export matches for review
        if file_path in verification_files:
            file_name = os.path.basename(file_path)
            with open(f'verification_sample_{file_name}.json', 'w') as f:
                serializable_matches = []
                for match in matches:
                    serializable_match = {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                                        for k, v in match.items()}
                    serializable_matches.append(serializable_match)
                json.dump(serializable_matches, f, indent=2)
            print(f"Exported verification sample for {file_name}")
        
        # Save normalized file
        output_path = os.path.join("output_directory", f"normalized_{os.path.basename(file_path)}")
        normalized_df.to_excel(output_path, index=False)
```

### Step 8: Implement Knowledge Base Versioning

Maintain versions of your knowledge base to track changes and allow rollback:

```python
import shutil
from datetime import datetime

def version_knowledge_base():
    """Create a versioned copy of the current knowledge base"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    src = "verified_knowledge_base.json"
    dst = f"knowledge_base_versions/kb_{timestamp}.json"
    
    # Create directory if it doesn't exist
    os.makedirs("knowledge_base_versions", exist_ok=True)
    
    # Copy current knowledge base to versioned file
    shutil.copy2(src, dst)
    print(f"Created knowledge base version: {dst}")
    
    return dst
```

## Special Case: Brand New Dataset with Different AMC

When processing files for a different AMC, create a separate knowledge base:

```python
# Initialize a separate matcher for a different AMC
icici_matcher = ComprehensiveSchemeMatcher(
    amc_name="ICICI Prudential",
    api_endpoint="https://your-api-endpoint.com/normalize",
    knowledge_base_path="icici_knowledge_base.json"  # Separate knowledge base
)

# Follow the same validation-first approach
# 1. Process a small sample
# 2. Manually verify
# 3. Build a verified seed knowledge base
# 4. Expand with confidence filtering
```

## Best Practices for Reliable Knowledge Base Building

1. **Start Small**: Always begin with a small, manageable sample
2. **Verify First**: Manually verify before incorporating into knowledge base
3. **Confidence Thresholds**: Use high confidence thresholds for learning (0.85+)
4. **Categorize Errors**: Track patterns in API errors to create correction rules
5. **Version Control**: Maintain dated versions of your knowledge base
6. **Separate AMCs**: Use separate knowledge bases for different AMCs
7. **Regular Audits**: Periodically review random samples from the knowledge base
8. **Document Corrections**: Keep records of all manual corrections
9. **Incremental Growth**: Build up the knowledge base gradually
10. **Feedback Loop**: Establish a systematic process for incorporating feedback

By following this structured approach, you can build a reliable knowledge base even when starting with potentially incorrect API responses, ensuring your scheme matching system improves in accuracy over time.
