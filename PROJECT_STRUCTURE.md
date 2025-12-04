# Project Structure Documentation

## Overview
This document explains the organization of the Diabetes Decision Support System with Logic Tensor Networks (LTN) thesis project.

## Directory Structure

```
thesis_code/
├── src/                    # Source code
│   ├── main/              # Main pipeline implementations
│   │   ├── main_pipeline2.py         # Main diabetes decision support pipeline
│   │   └── diabetes_ltn_runtime.py   # LTN runtime interface
│   │
│   ├── training/          # Training scripts
│   │   ├── main_train.py            # LTN model training script
│   │   └── ltn_core.py              # Core LTN implementation
│   │
│   ├── evaluation/        # Evaluation framework
│   │   ├── test_streamlined_evaluation.py  # Comprehensive evaluation
│   │   └── custom_metrics.py              # Custom medical AI metrics
│   │
│   └── baseline/          # Baseline implementations
│       └── baseline.py              # Simple baseline without retrieval
│
├── data/                  # Data directory
│   ├── AMIE-rules.csv          # Association rule mining results
│   ├── KG-triples.csv          # Medical knowledge graph triples
│   ├── patient_facts.jsonl     # Patient-specific medical facts
│   ├── diakg/                  # DiaKG dataset files (before and after translation)
│   ├── diabetesQA.csv          # TruthfulQA inspired questions
│   ├── Synthetic-data/         # Synthetic diabetes datasets
│   ├── train_triples.json      # Training dataset
│   ├── valid_triples.json      # Validation dataset
│   ├── test_triples.json       # Test dataset
│   ├── entities.json           # Entity mappings
│   ├── relations.json          # Relation mappings
│   ├── empirical_rules_compiled.json  # Compiled rules
│   ├── diabetes_prompts.jsonl  # Evaluation prompts
│   ├── unified_evaluation_*.jsonl     # Evaluation results
│   ├── ltn_final.pt            # Trained LTN checkpoint
│   ├── ltn_metrics.json        # Training metrics
│   └── ltn_test_metrics.json   # Test metrics
│
├── preprocessing/         # Data preprocessing scripts
│   ├── DiaKG-Preprocess_and_upload2neo4j.py  # DiaKG to Neo4j
│   ├── DiaKG-Translation.py                   # Translation utilities
│   ├── embeddings-neo4j.py                    # Embedding generation
│   ├── patients-to-neo4j.py                   # Patient data upload
│   ├── exclusion-entities.json                # Entity exclusion list
│   └── MIIC-SDG.R                            # R script for SDG
│   ├── KG-translation-evaluation.ipynb        # Translation evaluation
│   ├── Synthetic-data-analysis.ipynb         # Synthetic data analysis
│   ├── amie-rule-analysis.ipynb              # AMIE rule analysis
│   └── preprocess.ipynb                       # Preprocessing notebook
│
├── config/              # Configuration files
│   └── config.yaml                            # API keys and settings
│
├── docs/                # Documentation
│   └── README.md                              # Main project README
│
└── requirements.txt     # Python dependencies
```

## Key Components

### 1. Main Pipeline (`src/main/`)
- **main_pipeline2.py**: Integrates Neo4j retrieval with LTN reasoning
- **diabetes_ltn_runtime.py**: Provides runtime interface for trained LTN

### 2. Training (`src/training/`)
- **main_train.py**: Handles LTN model training with medical knowledge
- **ltn_core.py**: Implements the Logic Tensor Network architecture

### 3. Evaluation (`src/evaluation/`)
- **test_streamlined_evaluation.py**: Multi-judge evaluation framework
- **custom_metrics.py**: Medical-specific evaluation metrics

### 4. Data Organization (`data/`)
- Original datasets before processing
- Cleaned and split datasets ready for training
- Saved model checkpoints and performance metrics

### 5. Preprocessing (`preprocessing/`)
Contains all scripts for data preparation, Neo4j integration, and format conversion, exploratory analysis and visualization

## Usage Guide

1. **Configuration**: Update `config/config.yaml` with API keys
2. **Data Preparation**: Run preprocessing scripts in `preprocessing/`
3. **Training**: Execute `python src/training/main_train.py`
4. **Running Pipeline**: Use `python src/main/main_pipeline2.py`
5. **Evaluation**: Run `deepeval test run src/evaluation/test_streamlined_evaluation.py`

