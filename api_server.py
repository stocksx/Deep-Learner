#!/usr/bin/env python3
"""
FastAPI Server for Configurable Pattern-Based Learning System

This server provides REST API endpoints for:
1. Training entity patterns from text data
2. Chunking text using learned patterns
3. Managing configuration files
4. Getting statistics and system information
"""

import sys
import os
import json
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# Add L_1 directory to path for imports
sys.path.append('L_1')

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("Error: FastAPI and uvicorn are required. Install with: pip install fastapi uvicorn")
    sys.exit(1)

from L1_learner import ConfigurableBootstrapper, PatternBasedChunker, LLM_Chunker

# Initialize FastAPI app
app = FastAPI(
    title="Pattern-Based Learning & Chunking API",
    description="API for training entity patterns and performing intelligent text chunking",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Global storage for learners and chunkers
learners: Dict[str, ConfigurableBootstrapper] = {}
chunkers: Dict[str, PatternBasedChunker] = {}

# Configuration
CONFIG_DIR = "configs"
DEFAULT_CHUNK_SIZE = 512

# Pydantic models for request/response
class TrainingRequest(BaseModel):
    entity_type: str = Field(..., description="Entity type to train (e.g., 'broker', 'company')")
    text: str = Field(..., description="Training text containing entities")
    entities: Dict[str, str] = Field(..., description="Dict mapping entity text to normalized form")
    config_dir: Optional[str] = Field(CONFIG_DIR, description="Configuration directory")

class ChunkingRequest(BaseModel):
    text: str = Field(..., description="Text to chunk")
    entity_types: List[str] = Field(..., description="List of entity types to use for chunking")
    max_chunk_size: Optional[int] = Field(DEFAULT_CHUNK_SIZE, description="Maximum chunk size in characters")
    overlap_size: Optional[int] = Field(50, description="Overlap size between chunks")
    config_dir: Optional[str] = Field(CONFIG_DIR, description="Configuration directory")

class SemanticChunkingRequest(BaseModel):
    text: str = Field(..., description="Text to chunk semantically")
    similarity_threshold: Optional[float] = Field(0.7, description="Similarity threshold (0.0-1.0)")
    max_chunk_sentences: Optional[int] = Field(5, description="Maximum sentences per chunk")

class EntityExample(BaseModel):
    original_text: str = Field(..., description="Original entity text")
    normalized_form: str = Field(..., description="Normalized form of entity")

class ConfigurationRequest(BaseModel):
    entity_type: str = Field(..., description="Entity type to configure")
    examples: Optional[List[EntityExample]] = Field(None, description="Entity examples")
    heuristics: Optional[Dict[str, Any]] = Field(None, description="Heuristics configuration")

class TrainingResponse(BaseModel):
    success: bool
    entity_type: str
    examples_learned: int
    patterns_learned: Dict[str, int]
    message: str

class ChunkingResponse(BaseModel):
    success: bool
    total_chunks: int
    chunks: List[Dict[str, Any]]
    statistics: Dict[str, Any]

class SystemInfoResponse(BaseModel):
    available_entity_types: List[str]
    config_files: Dict[str, List[str]]
    loaded_learners: List[str]
    system_status: str

# Helper functions
def get_learner(entity_type: str, config_dir: str = CONFIG_DIR) -> ConfigurableBootstrapper:
    """Get or create a learner for the specified entity type."""
    learner_key = f"{entity_type}:{config_dir}"
    
    if learner_key not in learners:
        learners[learner_key] = ConfigurableBootstrapper(entity_type, config_dir)
    
    return learners[learner_key]

def get_chunker(entity_types: List[str], max_chunk_size: int, config_dir: str = CONFIG_DIR) -> PatternBasedChunker:
    """Get or create a chunker for the specified entity types."""
    chunker_key = f"{':'.join(sorted(entity_types))}:{max_chunk_size}:{config_dir}"
    
    if chunker_key not in chunkers:
        learner_list = [get_learner(et, config_dir) for et in entity_types]
        chunkers[chunker_key] = PatternBasedChunker(learner_list, max_chunk_size)
    
    return chunkers[chunker_key]

def get_available_entity_types(config_dir: str = CONFIG_DIR) -> List[str]:
    """Get list of available entity types based on configuration files."""
    if not os.path.exists(config_dir):
        return []
    
    entity_types = set()
    for file in os.listdir(config_dir):
        if file.endswith('_examples.txt') or file.endswith('_heuristics.json'):
            entity_type = file.split('_')[0]
            entity_types.add(entity_type)
    
    return sorted(list(entity_types))

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Pattern-Based Learning & Chunking API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=Dict[str, str])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "pattern-learning-api"}

@app.get("/info", response_model=SystemInfoResponse)
async def get_system_info():
    """Get system information and available configurations."""
    entity_types = get_available_entity_types()
    
    config_files = {}
    if os.path.exists(CONFIG_DIR):
        for entity_type in entity_types:
            files = []
            examples_file = f"{entity_type}_examples.txt"
            heuristics_file = f"{entity_type}_heuristics.json"
            patterns_file = f"{entity_type}_patterns.json"
            
            if os.path.exists(os.path.join(CONFIG_DIR, examples_file)):
                files.append(examples_file)
            if os.path.exists(os.path.join(CONFIG_DIR, heuristics_file)):
                files.append(heuristics_file)
            if os.path.exists(os.path.join(CONFIG_DIR, patterns_file)):
                files.append(patterns_file)
            
            config_files[entity_type] = files
    
    return SystemInfoResponse(
        available_entity_types=entity_types,
        config_files=config_files,
        loaded_learners=list(learners.keys()),
        system_status="operational"
    )

@app.post("/train", response_model=TrainingResponse)
async def train_entity_patterns(request: TrainingRequest):
    """Train entity patterns from text data."""
    try:
        # Get or create learner
        learner = get_learner(request.entity_type, request.config_dir)
        
        # Count patterns before training
        patterns_before = {
            'word_patterns': len(learner.patterns['word_patterns']),
            'position_patterns': len(learner.patterns['position_patterns']),
            'context_patterns': len(learner.patterns['context_patterns'])
        }
        
        # Train on provided data
        learner.learn_from_paragraph(request.text, request.entities)
        
        # Count patterns after training
        patterns_after = {
            'word_patterns': len(learner.patterns['word_patterns']),
            'position_patterns': len(learner.patterns['position_patterns']),
            'context_patterns': len(learner.patterns['context_patterns'])
        }
        
        # Calculate patterns learned
        patterns_learned = {
            key: patterns_after[key] - patterns_before[key] 
            for key in patterns_before
        }
        
        # Save learned patterns
        learner.save_patterns_to_file()
        learner.save_examples_to_file()
        
        return TrainingResponse(
            success=True,
            entity_type=request.entity_type,
            examples_learned=len(request.entities),
            patterns_learned=patterns_learned,
            message=f"Successfully trained {request.entity_type} patterns from provided text"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/chunk", response_model=ChunkingResponse)
async def chunk_text(request: ChunkingRequest):
    """Chunk text using learned entity patterns."""
    try:
        # Validate entity types
        available_types = get_available_entity_types(request.config_dir)
        invalid_types = [et for et in request.entity_types if et not in available_types]
        
        if invalid_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unknown entity types: {invalid_types}. Available: {available_types}"
            )
        
        # Get chunker
        chunker = get_chunker(request.entity_types, request.max_chunk_size, request.config_dir)
        
        # Perform chunking
        chunks = chunker.chunk_text(request.text, request.overlap_size)
        
        # Get statistics
        statistics = chunker.get_chunking_stats(chunks)
        
        return ChunkingResponse(
            success=True,
            total_chunks=len(chunks),
            chunks=chunks,
            statistics=statistics
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunking failed: {str(e)}")

@app.post("/chunk/semantic", response_model=Dict[str, Any])
async def chunk_text_semantic(request: SemanticChunkingRequest):
    """Chunk text using semantic similarity."""
    try:
        # Check if sentence transformers are available
        chunker = LLM_Chunker()
        
        # Perform semantic chunking
        chunks = chunker.chunk_text(
            request.text, 
            request.similarity_threshold, 
            request.max_chunk_sentences
        )
        
        return {
            "success": True,
            "total_chunks": len(chunks),
            "chunks": [{"content": chunk, "length": len(chunk)} for chunk in chunks],
            "similarity_threshold": request.similarity_threshold,
            "max_chunk_sentences": request.max_chunk_sentences
        }
        
    except ImportError:
        raise HTTPException(
            status_code=503, 
            detail="Semantic chunking unavailable. Install sentence-transformers and nltk."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Semantic chunking failed: {str(e)}")

@app.post("/configure", response_model=Dict[str, str])
async def configure_entity_type(request: ConfigurationRequest):
    """Configure a new entity type with examples and heuristics."""
    try:
        # Create config directory if it doesn't exist
        os.makedirs(CONFIG_DIR, exist_ok=True)
        
        messages = []
        
        # Save examples if provided
        if request.examples:
            examples_file = os.path.join(CONFIG_DIR, f"{request.entity_type}_examples.txt")
            with open(examples_file, 'w', encoding='utf-8') as f:
                f.write(f"# Validated examples for {request.entity_type}\n")
                f.write("# Format: original_text|normalized_form\n\n")
                for example in request.examples:
                    if example.original_text == example.normalized_form:
                        f.write(f"{example.original_text}\n")
                    else:
                        f.write(f"{example.original_text}|{example.normalized_form}\n")
            messages.append(f"Saved {len(request.examples)} examples")
        
        # Save heuristics if provided
        if request.heuristics:
            heuristics_file = os.path.join(CONFIG_DIR, f"{request.entity_type}_heuristics.json")
            with open(heuristics_file, 'w', encoding='utf-8') as f:
                json.dump(request.heuristics, f, indent=2, ensure_ascii=False)
            messages.append("Saved heuristics configuration")
        
        # Clear any cached learners for this entity type
        keys_to_remove = [key for key in learners.keys() if key.startswith(f"{request.entity_type}:")]
        for key in keys_to_remove:
            del learners[key]
        
        return {
            "success": "true",
            "entity_type": request.entity_type,
            "message": "; ".join(messages) if messages else "Configuration updated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Configuration failed: {str(e)}")

@app.get("/entity-types", response_model=List[str])
async def list_entity_types():
    """List all available entity types."""
    return get_available_entity_types()

@app.get("/entity-types/{entity_type}/info", response_model=Dict[str, Any])
async def get_entity_type_info(entity_type: str):
    """Get detailed information about a specific entity type."""
    try:
        learner = get_learner(entity_type)
        
        return {
            "entity_type": entity_type,
            "examples_count": len(learner.validated_examples),
            "patterns": {
                "word_patterns": len(learner.patterns['word_patterns']),
                "position_patterns": len(learner.patterns['position_patterns']),
                "context_patterns": len(learner.patterns['context_patterns'])
            },
            "heuristics": learner.entity_heuristics,
            "sample_examples": list(learner.validated_examples.items())[:5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Entity type not found: {str(e)}")

@app.delete("/entity-types/{entity_type}")
async def delete_entity_type(entity_type: str):
    """Delete configuration files for an entity type."""
    try:
        files_deleted = []
        
        # Delete configuration files
        for suffix in ['_examples.txt', '_heuristics.json', '_patterns.json']:
            file_path = os.path.join(CONFIG_DIR, f"{entity_type}{suffix}")
            if os.path.exists(file_path):
                os.remove(file_path)
                files_deleted.append(f"{entity_type}{suffix}")
        
        # Clear cached learners
        keys_to_remove = [key for key in learners.keys() if key.startswith(f"{entity_type}:")]
        for key in keys_to_remove:
            del learners[key]
        
        # Clear cached chunkers
        keys_to_remove = [key for key in chunkers.keys() if entity_type in key.split(':')[0].split(':')]
        for key in keys_to_remove:
            del chunkers[key]
        
        return {
            "success": True,
            "entity_type": entity_type,
            "files_deleted": files_deleted,
            "message": f"Successfully deleted {entity_type} configuration"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")

@app.post("/clear-cache")
async def clear_cache():
    """Clear all cached learners and chunkers."""
    global learners, chunkers
    
    learners_count = len(learners)
    chunkers_count = len(chunkers)
    
    learners.clear()
    chunkers.clear()
    
    return {
        "success": True,
        "message": f"Cleared {learners_count} learners and {chunkers_count} chunkers from cache"
    }

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 Pattern-Based Learning & Chunking API starting up...")
    print(f"📁 Config directory: {CONFIG_DIR}")
    
    # Ensure config directory exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Load available entity types
    entity_types = get_available_entity_types()
    print(f"🎯 Available entity types: {entity_types}")
    
    print("✅ API ready to serve requests!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    print("🛑 Pattern-Based Learning & Chunking API shutting down...")
    
    # Save any unsaved patterns
    for learner in learners.values():
        try:
            learner.save_patterns_to_file()
        except Exception as e:
            print(f"Warning: Failed to save patterns: {e}")
    
    print("✅ Shutdown complete!")

# Development server runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Pattern-Based Learning & Chunking API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--config-dir", default=CONFIG_DIR, help="Configuration directory")
    
    args = parser.parse_args()
    
    # Update global config directory
    CONFIG_DIR = args.config_dir
    
    print(f"🚀 Starting API server on {args.host}:{args.port}")
    print(f"📁 Using config directory: {CONFIG_DIR}")
    print(f"📖 API docs available at: http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    ) 