# Diabetes Decision Support System with Logic Tensor Networks (LTN)

This project implements a clinical decision support system for diabetes management that combines Knowledge Graph retrieval (Neo4j GraphRAG) with Logic Tensor Networks (LTN) for enhanced reasoning and explainability.

## Project Overview

The system provides evidence-based diabetes guidance by:
1. Retrieving relevant medical facts from a Neo4j knowledge graph
2. Using LTN to validate and score the confidence of retrieved facts
3. Generating clinically-focused responses with proper evidence citations
4. Evaluating system performance using multiple metrics and judge models

## File Descriptions

### Core Pipeline Files

#### `src/main/main_pipeline2.py`
The main diabetes decision support pipeline that:
- Integrates Neo4j GraphRAG retrieval with LTN reasoning
- Implements cosine similarity-based relevance filtering
- Generates structured clinical responses with proper evidence citations
- Handles both general queries and patient-specific assessments
- Key classes:
  - `Neo4jDiabetesRetriever`: Handles knowledge graph queries with hybrid vector/fulltext search
  - `DiabetesDecisionSupport`: Main pipeline orchestrator that combines retrieval, LTN validation, and response generation

#### `src/main/diabetes_ltn_runtime.py`
Runtime interface for the trained LTN model:
- Loads pre-trained LTN checkpoint from disk
- Provides triple confidence scoring for fact validation
- Generates explanations for predictions based on fuzzy rules
- Key methods:
  - `triple_confidence()`: Score confidence of (head, relation, tail) triples
  - `explain()`: Get top supporting rules for a triple with their satisfaction scores

### Training Components

#### `src/training/main_train.py`
Training script for the LTN model:
- Loads medical knowledge graph and patient facts from JSON files
- Trains bilinear LTN with fuzzy rule regularization
- Implements alpha sweep for optimal rule weight selection
- Saves trained model checkpoints and performance metrics
- Supports numeric comparisons (LE/GE relations) for clinical measurements

#### `src/training/ltn_core.py`
Core Logic Tensor Network implementation containing:
- Bilinear LTN model for triple scoring using entity-relation embeddings
- Rule evaluation engine for fuzzy logic rules with product t-norm
- Training utilities (loss functions, negative sampling, validation)
- Data structures for atoms, rules, and rule banks
- Key components:
  - `BilinearLTN`: Neural model with bilinear scoring and numeric comparison support
  - `evaluate_rules()`: Fuzzy rule evaluation with caching for efficiency
  - `ltn_loss()`: Combined loss with BCE, rule regularization, and KG constraints

### Evaluation Framework

#### `src/evaluation/test_streamlined_evaluation.py`
Comprehensive evaluation framework using DeepEval:
- Multi-judge evaluation with Claude, GPT-4, and Mistral models
- RAG metrics (contextual precision/recall, faithfulness, relevancy)
- Custom medical metrics for healthcare AI assessment
- BLEURT semantic similarity scoring

#### `src/evaluation/custom_metrics.py`
Custom evaluation metrics for medical AI:
- `MedicalExplainabilityMetric`: Evaluates clarity of medical reasoning
- `MedicalVeracityMetric`: Checks factual accuracy against medical knowledge
- `Transparency`: Assesses answer transparency
- `BLEURTMetric`: BLEURT-20 semantic similarity for response quality

### Baseline and Configuration

#### `src/baseline/baseline.py`
Simple baseline implementation:
- Direct OpenAI API calls without retrieval augmentation
- Used for performance comparison against the full pipeline
- No system prompts or context enhancement
- Minimal preprocessing of user queries

#### `config/config.yaml`
Configuration file containing:
- API keys for various services (Mistral, Neo4j, OpenAI, Anthropic, etc.)
- Neo4j connection parameters (URI, credentials)
- LTN model settings (confidence threshold, checkpoint directory)
- Retrieval parameters (top_k for search results)

### Data Files

#### Knowledge Graph Files
- **`data/KG-triples.csv`**: Medical knowledge graph triples in CSV format with columns (head, relation, tail)
- **`data/train_triples.json`, `valid_triples.json`, `test_triples.json`**: Pre-split datasets for model training/validation/testing
- **`data/entities.json`, `relations.json`**: Mappings from entity/relation names to unique IDs
- **`data/entities_after.json`, `relations_after.json`**: Updated mappings after training including new entities from patient facts

#### Rules and Clinical Facts
- **`data/AMIE-rules.csv`**: Association rules mined from medical data using the AMIE algorithm

#### Model Artifacts
- **`data/ltn_final.pt`**: PyTorch checkpoint of the trained LTN model with learned embeddings and parameters
- **`data/ltn_metrics.json`**: Training metrics including loss curves and validation performance
- **`data/ltn_test_metrics.json`**: Final test set evaluation metrics (MRR, Hits@K, etc.)

#### Evaluation Data
- **`data/diabetes_prompts.jsonl`**: 20 nerated prompts covering various diabetes-related queries for system evaluation
- **`data/diabetesQA.csvprompts`**: 20 questions isnpired by TruthfulQA Benchmark
#### DiaKG Dataset
- **`data/diakg/0521_new_format/`**: Original Chinese diabetes knowledge graph files
- **`data/diakg/0521_new_format_translated/`**: English translations of the DiaKG dataset


### Preprocessing Scripts

#### Data Processing
- **`preprocessing/DiaKG-Preprocess_and_upload2neo4j.py`**: Processes DiaKG JSON files and uploads to Neo4j database
- **`preprocessing/DiaKG-Translation.py`**: Translates Chinese DiaKG data to English using translation APIs
- **`preprocessing/embeddings-neo4j.py`**: Generates and stores embeddings for Neo4j entities
- **`preprocessing/patients-to-neo4j.py`**: Uploads patient facts to Neo4j as a separate graph
- **`preprocessing/exclusion-entities.json`**: List of entities to exclude during processing

#### Analysis Notebooks
- **`preprocessing/KG-translation-evaluation.ipynb`**: Evaluates quality of DiaKG translations
- **`preprocessing/Synthetic-data-analysis.ipynb`**: Analyzes synthetic patient data distributions
- **`preprocessing/amie-rule-analysis.ipynb`**: Explores and visualizes AMIE-mined rules
- **`preprocessing/preprocess.ipynb`**: General data preprocessing and exploration

#### Other Tools
- **`preprocessing/MIIC-SDG.R`**: R script for causal discovery using MIIC algorithm

## Key Features

1. **Hybrid Retrieval**: Combines vector search and fulltext search in Neo4j
2. **LTN Reasoning**: Validates retrieved facts using neural-symbolic reasoning
3. **Evidence Grounding**: All clinical statements cite evidence sources
4. **Multi-Judge Evaluation**: Uses multiple LLMs for robust evaluation
5. **Clinical Safety**: Built-in safety checks and uncertainty expression
6. **Explainability**: Provides reasoning chains for predictions

## Usage

### Running the Main Pipeline
```bash
python src/main/main_pipeline2.py
```

### Training the LTN Model
```bash
python src/training/main_train.py
```

### Running Evaluation
```bash
deepeval test run src/evaluation/test_streamlined_evaluation.py
```

## Architecture

The system follows a modular architecture:
1. **Retrieval Layer**: Neo4j GraphRAG for knowledge retrieval
2. **Reasoning Layer**: LTN for fact validation and scoring
3. **Generation Layer**: LLM (Mistral) for response generation
4. **Evaluation Layer**: Multi-metric, multi-judge assessment

## Dependencies

See `requirements.txt` for a complete list of dependencies. Key packages include:
- PyTorch for neural network implementation
- Neo4j Python driver for knowledge graph interaction
- DeepEval for evaluation framework
- LangChain for LLM integration
- Transformers for BLEURT metrics
- Various API clients (OpenAI, Anthropic, Mistral)
