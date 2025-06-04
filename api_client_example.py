#!/usr/bin/env python3
"""
Example client for the Pattern-Based Learning & Chunking API

This script demonstrates how to:
1. Train entity patterns via API
2. Perform text chunking via API
3. Manage configurations via API
4. Test all major API endpoints
"""

import requests
import json
from typing import Dict, List, Any

class PatternLearningClient:
    """Client for interacting with the Pattern-Based Learning API."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def health_check(self) -> Dict[str, Any]:
        """Check if the API server is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and available configurations."""
        response = self.session.get(f"{self.base_url}/info")
        response.raise_for_status()
        return response.json()
    
    def train_patterns(self, entity_type: str, text: str, entities: Dict[str, str]) -> Dict[str, Any]:
        """Train entity patterns from text data."""
        payload = {
            "entity_type": entity_type,
            "text": text,
            "entities": entities
        }
        response = self.session.post(f"{self.base_url}/train", json=payload)
        response.raise_for_status()
        return response.json()
    
    def chunk_text(self, text: str, entity_types: List[str], max_chunk_size: int = 512) -> Dict[str, Any]:
        """Chunk text using learned patterns."""
        payload = {
            "text": text,
            "entity_types": entity_types,
            "max_chunk_size": max_chunk_size
        }
        response = self.session.post(f"{self.base_url}/chunk", json=payload)
        response.raise_for_status()
        return response.json()
    
    def semantic_chunk_text(self, text: str, similarity_threshold: float = 0.7) -> Dict[str, Any]:
        """Chunk text using semantic similarity."""
        payload = {
            "text": text,
            "similarity_threshold": similarity_threshold
        }
        response = self.session.post(f"{self.base_url}/chunk/semantic", json=payload)
        response.raise_for_status()
        return response.json()
    
    def configure_entity_type(self, entity_type: str, examples: List[Dict[str, str]] = None, 
                            heuristics: Dict[str, Any] = None) -> Dict[str, Any]:
        """Configure a new entity type."""
        payload = {"entity_type": entity_type}
        if examples:
            payload["examples"] = examples
        if heuristics:
            payload["heuristics"] = heuristics
            
        response = self.session.post(f"{self.base_url}/configure", json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_entity_types(self) -> List[str]:
        """List all available entity types."""
        response = self.session.get(f"{self.base_url}/entity-types")
        response.raise_for_status()
        return response.json()
    
    def get_entity_type_info(self, entity_type: str) -> Dict[str, Any]:
        """Get detailed information about an entity type."""
        response = self.session.get(f"{self.base_url}/entity-types/{entity_type}/info")
        response.raise_for_status()
        return response.json()


def demo_api_training_and_chunking():
    """Comprehensive demo of the API functionality."""
    
    print("🚀 Pattern-Based Learning API Client Demo")
    print("=" * 50)
    
    # Initialize client
    client = PatternLearningClient()
    
    try:
        # Check health
        print("\n🏥 Health Check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        
        # Get system info
        print("\n📊 System Information...")
        info = client.get_system_info()
        print(f"   Available entity types: {info['available_entity_types']}")
        print(f"   System status: {info['system_status']}")
        
        # Configure a new entity type (drugs)
        print("\n💊 Configuring Drug Entity Type...")
        drug_examples = [
            {"original_text": "Aspirin", "normalized_form": "ASPIRIN"},
            {"original_text": "Paracetamol", "normalized_form": "PARACETAMOL"},
            {"original_text": "Ibuprofen", "normalized_form": "IBUPROFEN"},
            {"original_text": "Amoxicillin", "normalized_form": "AMOXICILLIN"}
        ]
        
        drug_heuristics = {
            "indicators": ["TABLET", "CAPSULE", "SYRUP", "INJECTION", "MEDICINE"],
            "suffixes": ["MG", "ML", "TABS", "CAP"],
            "known_entities": ["ASPIRIN", "PARACETAMOL", "IBUPROFEN"],
            "weights": {
                "indicators": 0.3,
                "suffixes": 0.2,
                "known_entities": 0.5
            }
        }
        
        config_result = client.configure_entity_type(
            entity_type="drug",
            examples=drug_examples,
            heuristics=drug_heuristics
        )
        print(f"   ✓ {config_result['message']}")
        
        # Train broker patterns
        print("\n🏢 Training Broker Patterns...")
        training_text = "The client opened a trading account with ZERODHA for equity investments. HDFC SECURITIES provided advisory services."
        broker_entities = {
            "ZERODHA": "ZERODHA",
            "HDFC SECURITIES": "HDFC SECURITIES"
        }
        
        training_result = client.train_patterns("broker", training_text, broker_entities)
        print(f"   ✓ {training_result['message']}")
        print(f"   ✓ Learned {training_result['examples_learned']} examples")
        print(f"   ✓ Patterns learned: {training_result['patterns_learned']}")
        
        # Train drug patterns
        print("\n💊 Training Drug Patterns...")
        drug_text = "The patient was prescribed Aspirin 75mg tablets and Paracetamol 500mg for pain relief."
        drug_entities = {
            "Aspirin": "ASPIRIN",
            "Paracetamol": "PARACETAMOL"
        }
        
        drug_training = client.train_patterns("drug", drug_text, drug_entities)
        print(f"   ✓ {drug_training['message']}")
        
        # Test chunking with multiple entity types
        print("\n✂️  Testing Text Chunking...")
        test_document = """
        The investment portfolio was managed by SBI SECURITIES with regular monitoring.
        The patient was given Aspirin tablets for heart health as recommended by the doctor.
        Additional trading services were provided by ICICI DIRECT for futures and options.
        The medication included Paracetamol for pain management and regular check-ups.
        ZERODHA platform was used for equity transactions with competitive pricing.
        """
        
        chunking_result = client.chunk_text(
            text=test_document,
            entity_types=["broker", "drug"],
            max_chunk_size=200
        )
        
        print(f"   ✓ Created {chunking_result['total_chunks']} chunks")
        print(f"   ✓ Statistics: {chunking_result['statistics']}")
        
        # Show chunk details
        print("\n📦 Chunk Details:")
        for i, chunk in enumerate(chunking_result['chunks'], 1):
            print(f"   Chunk {i}: {len(chunk['content'])} chars, {chunk['total_patterns']} patterns")
            if chunk['pattern_summary']:
                for entity_type, summary in chunk['pattern_summary'].items():
                    print(f"      {entity_type}: {summary['entities']}")
        
        # Test semantic chunking (if available)
        print("\n🧠 Testing Semantic Chunking...")
        try:
            semantic_result = client.semantic_chunk_text(test_document, similarity_threshold=0.6)
            print(f"   ✓ Created {semantic_result['total_chunks']} semantic chunks")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 503:
                print("   ⚠️  Semantic chunking unavailable (missing dependencies)")
            else:
                raise
        
        # Get entity type information
        print("\n📋 Entity Type Information...")
        available_types = client.list_entity_types()
        print(f"   Available types: {available_types}")
        
        for entity_type in available_types[:2]:  # Show info for first 2 types
            try:
                info = client.get_entity_type_info(entity_type)
                print(f"   {entity_type}: {info['examples_count']} examples, {sum(info['patterns'].values())} patterns")
            except requests.exceptions.HTTPError:
                print(f"   {entity_type}: Info not available")
        
        print("\n✅ API Demo Complete!")
        print("\n📖 Visit http://127.0.0.1:8000/docs for interactive API documentation")
        
    except requests.exceptions.ConnectionError:
        print("❌ Error: Could not connect to API server")
        print("   Make sure the server is running: python3 api_server.py")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e.response.status_code}")
        print(f"   {e.response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")


def demo_api_workflow():
    """Demo showing a complete API workflow."""
    
    print("\n🔄 API Workflow Demo")
    print("-" * 30)
    
    client = PatternLearningClient()
    
    try:
        # Step 1: Configure entity type
        print("1. Configuring 'currency' entity type...")
        currency_examples = [
            {"original_text": "USD", "normalized_form": "US DOLLAR"},
            {"original_text": "EUR", "normalized_form": "EURO"},
            {"original_text": "INR", "normalized_form": "INDIAN RUPEE"},
            {"original_text": "GBP", "normalized_form": "BRITISH POUND"}
        ]
        
        client.configure_entity_type("currency", examples=currency_examples)
        print("   ✓ Currency entity type configured")
        
        # Step 2: Train patterns
        print("2. Training currency patterns...")
        training_text = "The portfolio includes USD investments and EUR bonds. INR deposits provide local exposure."
        entities = {"USD": "US DOLLAR", "EUR": "EURO", "INR": "INDIAN RUPEE"}
        
        result = client.train_patterns("currency", training_text, entities)
        print(f"   ✓ Trained with {result['examples_learned']} examples")
        
        # Step 3: Chunk text
        print("3. Chunking text with currency patterns...")
        test_text = "The fund allocated 40% to USD assets, 30% to EUR securities, and 30% to INR bonds for diversification."
        
        chunks = client.chunk_text(test_text, ["currency"], max_chunk_size=100)
        print(f"   ✓ Created {chunks['total_chunks']} chunks with {chunks['statistics']['total_entities_found']} entities")
        
        print("\n✅ Workflow complete!")
        
    except Exception as e:
        print(f"❌ Workflow error: {e}")


if __name__ == "__main__":
    print("🌟 Pattern-Based Learning API Client")
    print("Starting comprehensive demo...")
    
    # Run main demo
    demo_api_training_and_chunking()
    
    # Run workflow demo
    demo_api_workflow()
    
    print("\n🎯 Demo Summary:")
    print("  • Configured multiple entity types via API")
    print("  • Trained patterns from text data")
    print("  • Performed intelligent chunking")
    print("  • Tested system information endpoints")
    print("  • Demonstrated complete workflow")
    print("\n📚 Next steps:")
    print("  • Start server: python3 api_server.py")
    print("  • View docs: http://127.0.0.1:8000/docs")
    print("  • Test endpoints: python3 api_client_example.py") 