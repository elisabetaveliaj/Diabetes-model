#!/usr/bin/env python3
"""
Streamlined Multi-Judge DeepEval test with unified output format
- Combines single and multi-judge results into one structure
- Includes prompt metadata in output
- RAG metrics + 1 holistic metric with multiple judges
- Custom medical metrics with single judge for efficiency
"""

import os
import json
from json_repair import repair_json
from pathlib import Path
from datetime import datetime
import pytest
from typing import Dict, Any, List
import numpy as np
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", message="The default returned 'id' field", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Accessing the 'model_fields' attribute", category=DeprecationWarning)

from anthropic import Anthropic
from mistralai import Mistral
from openai import OpenAI

import deepeval
from deepeval import assert_test, evaluate
from deepeval.dataset import EvaluationDataset
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    GEval,
)
from deepeval.models.base_model import DeepEvalBaseLLM

# Import custom metrics (kept for single-judge evaluation)
try:
    from custom_metrics import (
        MedicalExplainabilityMetric,
        MedicalVeracityMetric,
        TransparencyMetric,
        BLEURTMetric,
    )
except ImportError:
    # Fallback if custom metrics file doesn't exist yet
    print("Warning: custom_metrics not found, using placeholder")
    MedicalExplainabilityMetric = None
    MedicalVeracityMetric = None
    TransparencyMetric = None
    BLEURTMetric = None



# Judge Model Wrappers following DeepEval documentation
class ClaudeJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "claude-3-haiku-20240307", api_key: str = None):
        self.model = model
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        try:
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0  # DeepEval recommends 0 for consistency
            )
            response_text = response.content[0].text
            
            # Handle schema if provided
            if schema:
                try:
                    import json
                    import re

                    
                    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
                    if code_block_match:
                        json_str = code_block_match.group(1).strip()
                    else:
                        # Find JSON object
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        json_str = json_match.group(0) if json_match else response_text
                    
                    # Repair malformed JSON
                    repaired_json = repair_json(json_str)
                    json_data = json.loads(repaired_json)
                    
                    return schema.model_validate(json_data)
                    
                except Exception as e:
                    print(f"JSON parsing failed: {e}")
                    # Return a minimal valid schema object if possible
                    try:
                        # Create empty schema with required fields
                        return schema.model_validate({"verdicts": []})
                    except:
                        return response_text
            return response_text
        except Exception as e:
            print(f"Claude API error: {e}")
            return ""

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model


class MistralJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "mistral-small-latest", api_key: str = None):
        self.model = model
        self.client = Mistral(api_key=api_key or os.getenv("MISTRAL_API_KEY"))

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0
            )
            response_text = response.choices[0].message.content
            
            # Handle schema if provided
            if schema:
                try:
                    import json
                    import re

                
                    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
                    if code_block_match:
                        json_str = code_block_match.group(1).strip()
                    else:
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        json_str = json_match.group(0) if json_match else response_text
                    
                    repaired_json = repair_json(json_str)
                    json_data = json.loads(repaired_json)
                    return schema.model_validate(json_data)
                    
                except Exception as e:
                    print(f"JSON parsing failed: {e}")
                    try:
                        return schema.model_validate({"verdicts": []})
                    except:
                        return response_text
                        
            return response_text
        except Exception as e:
            print(f"Mistral API error: {e}")
            return ""

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model


class GPTJudge(DeepEvalBaseLLM):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str = None):
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def load_model(self):
        return self.model

    def generate(self, prompt: str, schema=None):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0
            )
            response_text = response.choices[0].message.content
            
            # Handle schema if provided
            if schema:
                try:
                    import json
                    import re

                
                    code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response_text)
                    if code_block_match:
                        json_str = code_block_match.group(1).strip()
                    else:
                        json_match = re.search(r'\{[\s\S]*\}', response_text)
                        json_str = json_match.group(0) if json_match else response_text
                    
                    repaired_json = repair_json(json_str)
                    json_data = json.loads(repaired_json)
                    return schema.model_validate(json_data)
                    
                except Exception as e:
                    print(f"JSON parsing failed: {e}")
                    try:
                        return schema.model_validate({"verdicts": []})
                    except:
                        return response_text
                        
            return response_text
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return ""

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return self.model


# Load config and set API keys
def load_config(path: str = "data/config.yaml") -> dict:
    import yaml
    with open(path, 'r') as f:
        return yaml.safe_load(f)

config = load_config()

# Set API keys from config if not in environment
if not os.getenv("OPENAI_API_KEY") and config.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]
if not os.getenv("ANTHROPIC_API_KEY") and config.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = config["ANTHROPIC_API_KEY"]
if not os.getenv("MISTRAL_API_KEY") and config.get("mistral_api_key"):
    os.environ["MISTRAL_API_KEY"] = config["mistral_api_key"]

# --- HF Token Setup ---
HF_TOKEN = ""
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN


# Initialize judges
judges = []
if os.getenv("ANTHROPIC_API_KEY"):
    judges.append(ClaudeJudge())

if os.getenv("MISTRAL_API_KEY"):
    judges.append(MistralJudge())

if os.getenv("OPENAI_API_KEY"):
    judges.append(GPTJudge())

# Select primary judge for custom metrics
primary_judge = judges[0] if judges else None

# Import systems
from main_pipeline2 import DiabetesDecisionSupport
from baseline import run_baseline

# Enable full evidence for RAG evaluation
config["include_full_evidence"] = True
diabetes_system = DiabetesDecisionSupport(config)

# Load test prompts
_PROMPT_PATH = Path(__file__).with_name("diabetes_prompts.jsonl")
if _PROMPT_PATH.exists():
    with _PROMPT_PATH.open(encoding="utf-8") as fp:
        test_prompts = [json.loads(line) for line in fp][:100]
else:
    print("prompts were not loaded")

# Storage for results - organized by prompt
prompt_results = {}


@pytest.mark.parametrize("prompt_data", test_prompts)
@pytest.mark.parametrize("system_name,system", [
    ("DiabetesDecisionSupport", diabetes_system),
    ("Baseline", None)
])
def test_comprehensive_evaluation(prompt_data, system_name, system):
    """Test each system with multi-judge RAG metrics AND single-judge custom metrics"""
    
    # Initialize prompt result structure if not exists
    prompt_id = prompt_data["id"]
    if prompt_id not in prompt_results:
        prompt_results[prompt_id] = {
            "prompt_id": prompt_id,
            "input": prompt_data["input"],
            "target": prompt_data.get("target", prompt_data.get("expected_output", "")),
            "metadata": prompt_data.get("metadata", {}),
            "systems": {}
        }

    print(f"\n=== Evaluating {system_name} System ===")
    print(f"Prompt: {prompt_data['input'][:80]}...")

    # Get response
    if system_name == "Baseline":
        response = run_baseline(prompt_data["input"])
    else:
        response = system.query(prompt_data["input"], mode="general")

    # Create test case
    if system_name != "Baseline":
        # Access retrieved facts from full_evidence
        full_evidence = response.get("full_evidence", {})
        retrieval_context = full_evidence.get("retrieved_facts", [])
    else:
        retrieval_context = []
    
    print(f"Retrieval context: {len(retrieval_context) if retrieval_context else 0} documents")
    
    test_case = LLMTestCase(
        input=prompt_data["input"],
        actual_output=response.get("response", ""),
        expected_output=prompt_data.get("target", prompt_data.get("expected_output", "")),
        retrieval_context=retrieval_context
    )

    # Store results for this system
    system_result = {
        "actual_output": response.get("response", ""),
        "response_length": len(response.get("response", "")),
        "retrieval_context_count": len(retrieval_context),
        "metrics": {}
    }

    # 1. RAG Evaluation
    if primary_judge:
        print(f"\nRAG Evaluation ({primary_judge.get_model_name()}):")
        
        # RAG metrics
        rag_metrics = [
            ContextualPrecisionMetric(model=primary_judge, include_reason=False),
            ContextualRecallMetric(model=primary_judge, include_reason=False), 
            ContextualRelevancyMetric(model=primary_judge, include_reason=False),
            AnswerRelevancyMetric(model=primary_judge, include_reason=False),
            FaithfulnessMetric(model=primary_judge, include_reason=False)
        ]
        
        for metric in rag_metrics:
            metric_name = getattr(metric, 'name', metric.__class__.__name__)
            try:
                metric.measure(test_case)
                system_result["metrics"][metric_name] = {
                    "score": metric.score,
                    "judge": primary_judge.get_model_name()
                }
                print(f"  {metric_name}: {metric.score:.3f}")
            except Exception as e:
                print(f"  {metric_name}: FAILED - {str(e)[:50]}...")
                system_result["metrics"][metric_name] = {
                    "score": None,
                    "error": str(e),
                    "judge": primary_judge.get_model_name()
                }
    
    # 2. Multi-Judge Holistic Evaluation
    print(f"\nMulti-Judge Holistic Evaluation:")
    holistic_scores = []
    holistic_judge_scores = {}
    
    for judge in judges:
        judge_name = judge.get_model_name()
        print(f"  Evaluating with {judge_name}...")
        
        holistic_metric = GEval(
            name="Holistic Clinical Quality",
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
            criteria=(
                "Holistic evaluation of clinical decision support quality considering: "
                "1. Medical accuracy and adherence to best practices "
                "2. Clarity and usefulness for healthcare professionals "
                "3. Appropriate evidence grounding and citation usage "
                "4. Safe and responsible medical guidance "
                "5. Overall quality as a clinical decision support tool"
            ),
            model=judge
        )
        
        try:
            holistic_metric.measure(test_case)
            score = holistic_metric.score
            holistic_scores.append(score)
            holistic_judge_scores[judge_name] = score
            print(f"    Holistic Clinical Quality: {score:.3f}")
        except Exception as e:
            print(f"    Holistic Clinical Quality: FAILED - {str(e)[:50]}...")
            holistic_judge_scores[judge_name] = None
    
    # Store holistic metric with aggregated scores
    if holistic_scores:
        system_result["metrics"]["Holistic Clinical Quality"] = {
            "score": np.mean(holistic_scores),
            "std": np.std(holistic_scores),
            "min": min(holistic_scores),
            "max": max(holistic_scores),
            "judge_scores": holistic_judge_scores,
            "agreement": 1 - (np.std(holistic_scores) / (np.mean(holistic_scores) + 1e-6))
        }

    # 3. Custom Medical Metrics (single judge)
    if primary_judge and MedicalExplainabilityMetric:
        print(f"\nCustom Medical Metrics ({primary_judge.get_model_name()}):")

        custom_metrics = [
            MedicalExplainabilityMetric(model=primary_judge),
            MedicalVeracityMetric(model=primary_judge),
            TransparencyMetric(model=primary_judge)
        ]
        
        # BLEURT metric
        if BLEURTMetric:
            custom_metrics.append(BLEURTMetric())

        
        for metric in custom_metrics:
            try:
                metric.measure(test_case)
                system_result["metrics"][metric.name] = {
                    "score": metric.score,
                    "judge": primary_judge.get_model_name() if hasattr(metric, 'model') else "rule-based"
                }
                print(f"  {metric.name}: {metric.score:.3f}")
            except Exception as e:
                print(f"  {metric.name}: ERROR - {str(e)}")
                system_result["metrics"][metric.name] = {
                    "score": None,
                    "error": str(e),
                    "judge": primary_judge.get_model_name() if hasattr(metric, 'model') else "rule-based"
                }

    # Add to prompt results under the appropriate system
    prompt_results[prompt_id]["systems"][system_name] = system_result

    # For DeepEval assertion
    if primary_judge and MedicalExplainabilityMetric:
        custom_assert_metrics = [
            MedicalExplainabilityMetric(model=primary_judge),
            MedicalVeracityMetric(model=primary_judge),
            TransparencyMetric(model=primary_judge)
        ]
        #assert_test(test_case, custom_assert_metrics, run_async=False)


def test_generate_summary():
    """Generate comprehensive summary after all tests"""

    if not prompt_results:
        pytest.skip("No results to summarize")

    # Export raw results with unified format
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"unified_evaluation_{timestamp}.jsonl"

    with open(filename, 'w') as f:
        for prompt_id, result in prompt_results.items():
            f.write(json.dumps(result) + '\n')

    print(f"\n\nResults exported to: {filename}")

    # Flatten results for analysis
    all_results = []
    for prompt_id, prompt_data in prompt_results.items():
        for system_name, system_data in prompt_data["systems"].items():
            flat_result = {
                "prompt_id": prompt_id,
                "system": system_name,
                "prompt_metadata": prompt_data["metadata"],
                "metrics": system_data["metrics"],
                "response_length": system_data["response_length"]
            }
            all_results.append(flat_result)
    
    # Group results by system
    system_results = {"DiabetesDecisionSupport": [], "Baseline": []}
    for result in all_results:
        system_results[result["system"]].append(result)

    # Print comprehensive summary
    print("\n" + "=" * 120)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 120)

    # 1. All Metrics Summary
    print("\n--- METRICS SUMMARY ---")
    print(f"Judges: {', '.join([j.get_model_name() for j in judges])}")
    
    # Collect all metric names
    all_metric_names = set()
    for result in all_results:
        all_metric_names.update(result["metrics"].keys())
    
    print(f"\n{'Metric':<35} {'DiabetesDecisionSupport':<25} {'Baseline':<25}")
    print("-" * 85)

    for metric_name in sorted(all_metric_names):
        row = f"{metric_name[:35]:<35}"

        for system in ["DiabetesDecisionSupport", "Baseline"]:
            scores = []
            for result in system_results[system]:
                if metric_name in result["metrics"]:
                    metric_data = result["metrics"][metric_name]
                    if isinstance(metric_data.get("score"), (int, float)):
                        scores.append(metric_data["score"])

            if scores:
                avg = np.mean(scores)
                std = np.std(scores)
                row += f"{avg:.3f} (±{std:.3f})"
                row += " " * (25 - len(f"{avg:.3f} (±{std:.3f})"))
            else:
                row += "N/A" + " " * 22

        print(row)

    # 2. Metadata Analysis
    print("\n--- PROMPT METADATA ANALYSIS ---")
    metadata_stats = {}
    for prompt_data in prompt_results.values():
        metadata = prompt_data.get("metadata", {})
        for key, value in metadata.items():
            if key not in metadata_stats:
                metadata_stats[key] = {}
            if value not in metadata_stats[key]:
                metadata_stats[key][value] = 0
            metadata_stats[key][value] += 1
    
    for key, values in metadata_stats.items():
        print(f"\n{key.capitalize()}:")
        for value, count in values.items():
            print(f"  {value}: {count} prompts")

    # 3. Judge Agreement Analysis (for Holistic metric)
    print("\n--- JUDGE AGREEMENT ANALYSIS (Holistic Clinical Quality) ---")
    agreements = []
    for prompt_data in prompt_results.values():
        for system_name, system_data in prompt_data["systems"].items():
            if "Holistic Clinical Quality" in system_data["metrics"]:
                metric_data = system_data["metrics"]["Holistic Clinical Quality"]
                if "agreement" in metric_data:
                    agreements.append(metric_data["agreement"])
    
    if agreements:
        avg_agreement = np.mean(agreements)
        print(f"Average Agreement: {avg_agreement:.3f}")
        if avg_agreement > 0.8:
            print("Interpretation: High consensus among judges")
        elif avg_agreement > 0.6:
            print("Interpretation: Moderate consensus among judges")
        else:
            print("Interpretation: Low consensus among judges")

    # 4. Response Length Analysis
    print("\n--- RESPONSE LENGTH ANALYSIS ---")
    for system in ["DiabetesDecisionSupport", "Baseline"]:
        lengths = [r["response_length"] for r in system_results[system]]
        if lengths:
            print(f"{system}: {np.mean(lengths):.0f} chars (avg), {np.std(lengths):.0f} (std)")

    # Calculate comparison
    system_lengths = [r["response_length"] for r in system_results["DiabetesDecisionSupport"] if "response_length" in r]
    baseline_lengths = [r["response_length"] for r in system_results["Baseline"] if "response_length" in r]
    if system_lengths and baseline_lengths:
        diff = (np.mean(system_lengths) / np.mean(baseline_lengths) - 1) * 100
        if diff > 0:
            print(f"\nDiabetesDecisionSupport is {diff:.1f}% longer than Baseline")
        else:
            print(f"\nDiabetesDecisionSupport is {abs(diff):.1f}% shorter than Baseline")

    # 5. Overall Winner Analysis
    print("\n--- OVERALL PERFORMANCE SUMMARY ---")
    
    # Count wins per metric
    wins = {"DiabetesDecisionSupport": 0, "Baseline": 0, "Tie": 0}
    
    for metric_name in sorted(all_metric_names):
        diabetes_scores = []
        baseline_scores = []
        
        for result in system_results["DiabetesDecisionSupport"]:
            if metric_name in result["metrics"] and isinstance(result["metrics"][metric_name].get("score"), (int, float)):
                diabetes_scores.append(result["metrics"][metric_name]["score"])
        
        for result in system_results["Baseline"]:
            if metric_name in result["metrics"] and isinstance(result["metrics"][metric_name].get("score"), (int, float)):
                baseline_scores.append(result["metrics"][metric_name]["score"])
        
        if diabetes_scores and baseline_scores:
            diabetes_avg = np.mean(diabetes_scores)
            baseline_avg = np.mean(baseline_scores)
            
            if abs(diabetes_avg - baseline_avg) < 0.01:  # Consider as tie if very close
                wins["Tie"] += 1
            elif diabetes_avg > baseline_avg:
                wins["DiabetesDecisionSupport"] += 1
            else:
                wins["Baseline"] += 1

    print("\nMetric Wins:")
    for system, count in wins.items():
        print(f"  {system}: {count} metrics")
    



if __name__ == "__main__":
    import subprocess
    subprocess.run(["deepeval", "test", "run", __file__])