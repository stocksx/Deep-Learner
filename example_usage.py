#!/usr/bin/env python3
"""
Example usage of the Configurable Pattern-Based Learning System

This script demonstrates how to:
1. Load entity patterns from configuration files
2. Train on multiple entity types
3. Perform intelligent chunking
4. Analyze results
"""

import sys
import os
sys.path.append('L_1')

from L1_learner import ConfigurableBootstrapper, PatternBasedChunker


def demo_multi_entity_chunking():
    """Demo showing multi-entity pattern learning and chunking."""
    
    print("🚀 Configurable Pattern-Based Learning System Demo")
    print("=" * 60)
    
    # Create learners for different entity types
    print("\n📁 Loading Configuration Files...")
    
    # Broker learner - loads from configs/broker_*
    broker_learner = ConfigurableBootstrapper('broker', config_dir='configs')
    print(f"   ✓ Broker learner: {len(broker_learner.validated_examples)} pre-loaded examples")
    
    # Company learner - loads from configs/company_*  
    company_learner = ConfigurableBootstrapper('company', config_dir='configs')
    print(f"   ✓ Company learner: {len(company_learner.validated_examples)} pre-loaded examples")
    
    # Additional training data
    print("\n📚 Training on Paragraph Data...")
    
    training_data = [
        {
            'text': "The client portfolio was managed by ZERODHA with support from RELIANCE INDUSTRIES team.",
            'brokers': {"ZERODHA": "ZERODHA"},
            'companies': {"RELIANCE INDUSTRIES": "RELIANCE INDUSTRIES LIMITED"}
        },
        {
            'text': "UPSTOX collaborated with TCS LIMITED for technology solutions.",
            'brokers': {"UPSTOX": "UPSTOX"}, 
            'companies': {"TCS LIMITED": "TATA CONSULTANCY SERVICES LIMITED"}
        },
        {
            'text': "HDFC SECURITIES and INFOSYS LIMITED announced a strategic partnership.",
            'brokers': {"HDFC SECURITIES": "HDFC SECURITIES"},
            'companies': {"INFOSYS LIMITED": "INFOSYS LIMITED"}
        }
    ]
    
    # Train both learners
    for data in training_data:
        broker_learner.learn_from_paragraph(data['text'], data['brokers'])
        company_learner.learn_from_paragraph(data['text'], data['companies'])
    
    print(f"   ✓ Trained broker learner: {len(broker_learner.patterns['position_patterns'])} position patterns")
    print(f"   ✓ Trained company learner: {len(company_learner.patterns['position_patterns'])} position patterns")
    
    # Save learned patterns
    print("\n💾 Saving Learned Patterns...")
    broker_learner.save_patterns_to_file()
    company_learner.save_patterns_to_file()
    print("   ✓ Patterns saved to configs/ directory")
    
    # Create multi-entity chunker
    print("\n🔧 Creating Multi-Entity Chunker...")
    chunker = PatternBasedChunker([broker_learner, company_learner], max_chunk_size=150)
    print(f"   ✓ Chunker created with {len(chunker.entity_patterns)} entity types")
    
    # Test document with mixed entities
    test_document = """
    The quarterly investment review meeting was conducted by SBI SECURITIES last month.
    The client portfolio included shares of RELIANCE INDUSTRIES and TCS LIMITED.
    Additional analysis was provided by ICICI DIRECT for the technology sector.
    The recommendations included diversification into INFOSYS LIMITED and banking stocks.
    ZERODHA was chosen as the primary broker for executing the trades.
    The final portfolio included positions in HDFC BANK LIMITED and other blue-chip companies.
    """
    
    print("\n📄 Test Document:")
    print("-" * 40)
    print(test_document.strip())
    
    # Perform intelligent chunking
    print(f"\n✂️  Chunking Text (max chunk size: {chunker.max_chunk_size} chars)...")
    chunks = chunker.chunk_text(test_document.strip())
    
    print(f"\n📊 Chunking Results:")
    print("-" * 40)
    
    for i, chunk in enumerate(chunks, 1):
        print(f"\n📦 Chunk {i}:")
        print(f"   Length: {len(chunk['content'])} characters")
        print(f"   Sentences: {len(chunk['sentences'])}")
        print(f"   Boundary Quality: {chunk['boundary_quality']:.2f}")
        print(f"   Total Patterns: {chunk['total_patterns']}")
        
        # Show detected entities
        if chunk['pattern_summary']:
            print("   🎯 Detected Entities:")
            for entity_type, summary in chunk['pattern_summary'].items():
                entities_str = ", ".join(summary['entities'])
                print(f"      {entity_type.title()}: {entities_str} (confidence: {summary['avg_confidence']:.2f})")
        
        # Show chunk content (truncated)
        content_preview = chunk['content'].replace('\n', ' ').strip()
        if len(content_preview) > 100:
            content_preview = content_preview[:100] + "..."
        print(f"   📝 Content: {content_preview}")
    
    # Overall statistics
    stats = chunker.get_chunking_stats(chunks)
    print(f"\n📈 Overall Statistics:")
    print("-" * 40)
    print(f"   Total Chunks: {stats['total_chunks']}")
    print(f"   Total Entities Found: {stats['total_entities_found']}")
    print(f"   Average Entities per Chunk: {stats['avg_entities_per_chunk']:.1f}")
    print(f"   Average Chunk Size: {stats['avg_chunk_size']:.0f} characters")
    print(f"   Average Boundary Quality: {stats['avg_boundary_quality']:.2f}")
    
    print(f"\n   🏷️  Entity Distribution:")
    for entity_type, count in stats['entity_distribution'].items():
        print(f"      {entity_type.title()}: {count} entities")
    
    print(f"\n✅ Demo Complete! Check the configs/ directory for saved patterns.")


def show_configuration_files():
    """Show what configuration files are available."""
    print("\n📁 Available Configuration Files:")
    print("-" * 40)
    
    config_dir = 'configs'
    if os.path.exists(config_dir):
        files = os.listdir(config_dir)
        entity_types = set()
        
        for file in sorted(files):
            if file.endswith('_examples.txt'):
                entity_type = file.replace('_examples.txt', '')
                entity_types.add(entity_type)
                print(f"   📝 {file} - Examples for {entity_type}")
            elif file.endswith('_heuristics.json'):
                entity_type = file.replace('_heuristics.json', '')
                entity_types.add(entity_type)
                print(f"   ⚙️  {file} - Heuristics for {entity_type}")
            elif file.endswith('_patterns.json'):
                entity_type = file.replace('_patterns.json', '')
                entity_types.add(entity_type)
                print(f"   🧠 {file} - Learned patterns for {entity_type}")
        
        print(f"\n   🎯 Available Entity Types: {', '.join(sorted(entity_types))}")
    else:
        print("   ⚠️  No configs directory found. Run the demo to create sample configurations.")


if __name__ == "__main__":
    show_configuration_files()
    demo_multi_entity_chunking() 