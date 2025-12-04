#main_pipeline2.py
from pathlib import Path
import re
import os
import torch
import yaml
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import Counter
import math

from mistralai import Mistral
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_core.language_models import LLM
from pydantic import Field

from neo4j_graphrag.retrievers import HybridCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem
from neo4j_graphrag.embeddings import Embedder
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI

from diabetes_ltn_runtime import DiabetesLTNRuntime

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


#----------------Config------------
def load_config(path: str = "/data/config.yaml") -> Dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

###############################################################################
# Neo4j Retrieval
###############################################################################

class MistralEmbedderWrapper(Embedder):
    """Wrapper to make LangChain embeddings compatible with neo4j-graphrag."""
    
    def __init__(self, api_key: str, model: str = "mistral-embed"):
        self._embeddings = MistralAIEmbeddings(
            model=model,
            mistral_api_key=api_key
        )
    
    def embed_query(self, text: str) -> list[float]:
        return self._embeddings.embed_query(text)
        
        
class Neo4jDiabetesRetriever:
    """Neo4j GraphRAG retrieval"""

    def __init__(self, neo4j_uri: str, username: str, password: str, api_key: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(username, password))
        self.embeddings = MistralEmbedderWrapper(api_key=api_key)
        self._langchain_embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key
        )

        self.retriever = HybridCypherRetriever(
            driver=self.driver,
            vector_index_name="entities",
            fulltext_index_name="entityEnFulltextIndex",
            embedder=self.embeddings,
            retrieval_query="""
            MATCH (node)-[r]-(related:Entity)
            WHERE elementId(node) < elementId(related)
            RETURN 
              coalesce(node.entity_en, node.canonical_text, node.conceptual_id)
              + " --[" + type(r) + "]--> "
              + coalesce(related.entity_en, related.canonical_text, related.conceptual_id) AS text,
              {
                source_id:    elementId(node),
                relation_type: type(r),
                target_id:    elementId(related),
                source_text:  coalesce(node.entity_en, node.canonical_text, node.conceptual_id),
                target_text:  coalesce(related.entity_en, related.canonical_text, related.conceptual_id),
                score: score
              } AS metadata
            """,
        )

    def retrieve(self, query: str, top_k: int = 20) -> List[RetrieverResultItem]:
        items: List[RetrieverResultItem] = []
    
        safe_query = self._sanitize_query(query)
        remaining_slots = top_k - len(items)
        if remaining_slots > 0:
            pop = self.retriever.search(query_text=safe_query, top_k=remaining_slots)
            items.extend(pop.items)
    
        return items

    
    def _sanitize_query(self, query: str) -> str:
        """Minimal sanitization - only escape Lucene special chars."""
        # Lucene special chars: + - && || ! ( ) { } [ ] ^ " ~ * ? : \ /
        # Only escape those that commonly break queries
        lucene_special = r'([+\-&|!(){}[\]^"~*?:\\/])'
        sanitized = re.sub(lucene_special, r'\\\1', query)
        return sanitized.strip()

class DiabetesDecisionSupport:
    """Main pipeline combining Neo4j GraphRAG + LTN for diabetes decision support"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.retriever = Neo4jDiabetesRetriever(
            config["neo4j_uri"],
            config["neo4j_username"],
            config["neo4j_password"],
            config["mistral_api_key"]
        )
        self.ltn = self._initialize_ltn()
        self.llm = ChatMistralAI(
            model="mistral-large-latest",
            mistral_api_key=config["mistral_api_key"],
            temperature=0
        )
        
        self._ltn_coverage_stats = self._compute_ltn_coverage()
        
    def _compute_ltn_coverage(self) -> Dict[str, Any]:
        """
        Compute coverage statistics between Neo4j entities/relations and LTN vocabulary.
        Returns dict with coverage percentages and missing items.
        """
        if not self.ltn:
            return {"entity_coverage": 0.0, "relation_coverage": 0.0, "available": False}
        
        try:
            with self.retriever.driver.session() as session:
                # Get unique entities from Neo4j
                ent_result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.entity_en IS NOT NULL
                    RETURN COLLECT(DISTINCT e.entity_en) AS entities
                """)
                neo4j_entities = set(ent_result.single()["entities"])
                
                # Get unique relation types from Neo4j
                rel_result = session.run("""
                    MATCH ()-[r]-()
                    RETURN COLLECT(DISTINCT type(r)) AS relations
                """)
                neo4j_relations = set(rel_result.single()["relations"])
            
            ltn_entities = set(self.ltn.ent2id.keys())
            ltn_relations = set(self.ltn.rel2id.keys())
            
            # Compute overlaps
            common_entities = neo4j_entities & ltn_entities
            common_relations = neo4j_relations & ltn_relations
            missing_entities = neo4j_entities - ltn_entities
            missing_relations = neo4j_relations - ltn_relations
            
            entity_coverage = len(common_entities) / len(neo4j_entities) * 100 if neo4j_entities else 0
            relation_coverage = len(common_relations) / len(neo4j_relations) * 100 if neo4j_relations else 0
            
            stats = {
                "available": True,
                "entity_coverage": round(entity_coverage, 1),
                "relation_coverage": round(relation_coverage, 1),
                "neo4j_entity_count": len(neo4j_entities),
                "neo4j_relation_count": len(neo4j_relations),
                "ltn_entity_count": len(ltn_entities),
                "ltn_relation_count": len(ltn_relations),
                "common_entities": len(common_entities),
                "common_relations": len(common_relations),
                "missing_entities": missing_entities,
                "missing_relations": missing_relations,
            }
            
            """print(f"\n{'='*60}")
            print(f"LTN COVERAGE STATISTICS")
            print(f"{'='*60}")
            print(f"Entities:  {stats['common_entities']:,}/{stats['neo4j_entity_count']:,} "
                  f"({stats['entity_coverage']:.1f}% coverage)")
            print(f"Relations: {stats['common_relations']:,}/{stats['neo4j_relation_count']:,} "
                  f"({stats['relation_coverage']:.1f}% coverage)")
            
            if missing_relations:
                print(f"\nMissing relations in LTN: {missing_relations}")
            
            if stats['entity_coverage'] < 50:
                print(f"\nWARNING: Low entity coverage ({stats['entity_coverage']:.1f}%)")
                print(f"   LTN validation will be skipped for many retrieved triples.")
                print(f"   Consider retraining LTN with expanded entity vocabulary.")
            
            print(f"{'='*60}\n")"""
            
            return stats
            
        except Exception as e:
            print(f"[WARNING] Could not compute LTN coverage: {e}")
            return {"entity_coverage": 0.0, "relation_coverage": 0.0, "available": False}


    def _initialize_ltn(self) -> DiabetesLTNRuntime | None:
        try:
            ckpt_dir = Path(self.config["ltn_checkpoint_dir"])
            return DiabetesLTNRuntime(ckpt_dir)
        except Exception as e:
            print(f"LTN initialisation failed: {e}")
        return None

    def _build_structured_prompt(
        self,
        user_input: Union[str, Dict],
        docs: List[RetrieverResultItem],
        ltn_result: Optional[Dict],
        mode: str,
    ) -> str:
        """
        Builds a concise, clinically-focused prompt optimized for readability and efficiency.
        """

        # Collect facts
        retrieved_facts = []
        if docs:
            for i, doc in enumerate(docs[:20]):
                if hasattr(doc, "content"):
                    md = getattr(doc, "metadata", {}) or {}
                    src_id = md.get("source_id", "NA")
                    rel    = md.get("relation_type", "NA")
                    tgt_id = md.get("target_id", "NA")
                    retrieved_facts.append(
                        f"[F{i}] (KG src={src_id}, rel={rel}, tgt={tgt_id}) {doc.content}"
                    )
        
        ltn_facts = []
        if ltn_result and ltn_result.get("predicted_facts"):
            for i, (h, r, t, mu) in enumerate(ltn_result.get("predicted_facts", [])[:20]):
                ltn_facts.append(
                    f"[L{i}] (LTN triple: {h} --[{r}]--> {t}, conf={mu:.2f})"
                )

        # Streamlined system prompt
        system_frame = """You are DiabetesDSS, a clinical decision support tool for healthcare professionals.

Provide clear, explainable output that uses the available evidence and it is understandable from healthcare professionals.


**EVIDENCE CITATION PROTOCOL:**
BEFORE responding, verify that EVERY claim references a valid citation from the evidence list.

The metadata strings like "(KG src=4:abc123...)" are system identifiers - do NOT modify or create them.

___________________________________________________________________________

**MANDATORY OUTPUT STRUCTURE:**


**Clinical Summary**
Direct answer to the query with inline citations [F#][L#], and brief aetiology. Be brief and direct. "

**Risk Assessment**: [HIGH/MODERATE/LOW]
One-sentence rationale with citations.

**Recommended Actions** (bulleted list, ANSWER EACH ONLY IF AVAILABLE AND REQUIRED!!)
• Immediate: [Most urgent action] [evidence]
• Diagnostic: [Key tests needed] [evidence] 
• Treatment: [Primary intervention, IF NEEDED ONLY] [evidence]
• Monitoring: [Critical parameters] [evidence]

**Clinical Reasoning** (3-4 sentences max)
Synthesize evidence into coherent and interpretable explanation.

**Uncertainties**: [What remains unclear or needs specialist input]

_____________________________________________________________________


**QUALITY CONTROL CHECKLIST** (internal - do NOT print this section):
Before finalizing response, verify:
- All citations exist in the provided evidence list
- Uncertainties explicitly acknowledged

**RESPONSE RULES:**
1. Cite EVERY medical fact using [F#] or [L#] that appears in "Available Evidence"
2. BE BRIEF AND TO THE POINT

**FORBIDDEN:**
- Creating citations not in the evidence list
- use of emojis
- printing notes or any other remarks, unrelated to the user's query strictly
"""

        # Task-specific instructions
        if mode == "general":
            task_instruction = "Answer this diabetes query with actionable clinical guidance:"
            user_context = f"Query: {user_input}"

        # Evidence section - condensed
        graph_facts_text = (
            "\n".join(retrieved_facts)
            if retrieved_facts
            else "No relevant facts found"
        )

        ltn_facts_text = (
            "\n".join(ltn_facts)
            if ltn_facts
            else "No validated predictions"
        )


        evidence_section = f"""**Available Evidence:**
            Knowledge Graph Facts:
            {graph_facts_text}
            
            High-Confidence Predictions:
            {ltn_facts_text}

            """


        # Combine sections
        full_prompt = f"""{system_frame}

            {task_instruction}
            
            {evidence_section}
            
            {user_context}
            
            Remember: Be concise, cite all claims, focus on actionable guidance."""

        return full_prompt

    def _post_process_response(self, response: str) -> str:
        """
        Post-process LLM response to ensure clinical relevance and readability.
        """
        
        # Highlight critical information
        critical_patterns = [
            (r'(HIGH risk|URGENT|EMERGENT|CRITICAL)', r'⚠️ \1'),
            (r'(Contraindicated|Do not|Avoid)', r'❌ \1'),
            (r'(Monitor closely|Caution)', r'⚡ \1'),
        ]
        
        for pattern, replacement in critical_patterns:
            response = re.sub(pattern, replacement, response, flags=re.IGNORECASE)
        
        # Add clinical summary if not present
        if "Clinical Summary" not in response and len(response) > 500:
            # Extract key points for summary
            lines = response.split('\n')
            key_lines = [line for line in lines if any(
                keyword in line.lower() for keyword in 
                ['recommend', 'urgent', 'immediate', 'monitor', 'risk', 'diagnosis']
            )][:3]
            
            if key_lines:
                summary = "\n**Quick Summary:**\n" + "\n".join(f"• {line.strip()}" for line in key_lines)
                response = summary + "\n\n" + response
        
        return response

    def query(self, user_input: Union[str, Dict], mode: str = "general", temperature: float = 0.5) -> Dict:
      """
      Optimized query-processing pipeline with improved response formatting.
      """
      user_query_for_retrieval = ""
  
      # Process input based on mode
      if mode == "general":
          if not isinstance(user_input, str):
              raise ValueError("For 'general' mode user_input must be str.")
          user_query_for_retrieval = user_input
  
      # -------- 1) GraphRAG retrieval --------------------------------
      retrieved_docs = self.retriever.retrieve(
          user_query_for_retrieval,
          top_k=self.config.get("top_k", 20)
      )
  
      # -------- 2) Rerank with hybrid scores + LTN + semantic sim ----
      reranked_docs, ltn_result = self._rerank_documents(
          query=user_query_for_retrieval,
          retrieved_docs=retrieved_docs,
          alpha=self.config.get("rerank_alpha", 0.4),
          beta=self.config.get("rerank_beta", 0.4),
          gamma=self.config.get("rerank_gamma", 0.2),
          ltn_threshold=self.config.get("ltn_threshold", 0.75),
          max_facts=self.config.get("max_ltn_facts", 20),
      )
  
      # -------- 3) Build prompt with reranked docs -------------------
      prompt = self._build_structured_prompt(
          user_input,
          reranked_docs,  # Use reranked docs
          ltn_result,
          mode,
      )
  
      # -------- 4) LLM generation -----------------------------------
      llm_with_temp = self.llm.bind(temperature=temperature)
      response = llm_with_temp.invoke(prompt).content
  
      # -------- 5) Post-process for readability ----------------------
      response = self._post_process_response(response)
  
      # -------- 6) Verify citations ---------------------------------
      response_checked, citations_ok = self._verify_citations(
          response,
          reranked_docs,
          ltn_result,
      )
  
      # -------- 7) Build reference index -----------------------------
      kg_refs = []
      for i, doc in enumerate(reranked_docs[:20] if reranked_docs else []):
          md = getattr(doc, "metadata", {}) or {}
          
          # Include scoring info in references
          doc_score_info = {}
          if ltn_result.get("doc_components") and i < len(ltn_result["doc_components"]):
              doc_score_info = ltn_result["doc_components"][i]
          
          kg_refs.append({
              "label": f"F{i}",
              "source_id": md.get("source_id"),
              "relation_type": md.get("relation_type"),
              "target_id": md.get("target_id"),
              "text": doc.content if hasattr(doc, "content") else str(doc),
              "scores": {
                  "hybrid": doc_score_info.get("hybrid_score"),
                  "ltn": doc_score_info.get("ltn_confidence"),
                  "semantic": doc_score_info.get("semantic_similarity"),
                  "combined": doc_score_info.get("combined_score"),
              }
          })
  
      ltn_refs = []
      for i, fact_info in enumerate(ltn_result.get("detailed_facts", [])):
          ltn_refs.append({
              "label": f"L{i}",
              "head": fact_info["head"],
              "relation": fact_info["relation"],
              "tail": fact_info["tail"],
              "confidence": fact_info["ltn_confidence"],
              "semantic_similarity": fact_info["semantic_similarity"],
              "combined_score": fact_info["combined_score"],
          })
  
      # -------- 8) Final structured output ---------------------------
      # Clear embedding cache after query to manage memory
      self.clear_embedding_cache()
      
      return {
          "response": response_checked,
          "evidence_summary": {
              "graph_facts": len([d for d in reranked_docs if hasattr(d, "content")]),
              "validated_facts": len(ltn_result.get("predicted_facts", [])),
              "confidence_range": [
                  f["ltn_confidence"] 
                  for f in ltn_result.get("detailed_facts", [])
              ] if ltn_result.get("detailed_facts") else [],
              "scoring_weights": ltn_result.get("scoring_weights", {}),
              "ltn_coverage": ltn_result.get("ltn_coverage", {}),
          },
          "citations_verified": citations_ok,
          "reference_index": {
              "kg": kg_refs,
              "ltn": ltn_refs,
          },
          "full_evidence": {
              "retrieved_facts": [
                  doc.content for doc in reranked_docs[:20]
              ] if reranked_docs else [],
              "ltn_reasoning": ltn_result,
              "doc_scores": ltn_result.get("doc_scores", []),
          } if self.config.get("include_full_evidence", False) else None,
      }



    def _rerank_documents(
        self,
        query: str,
        retrieved_docs: List[RetrieverResultItem],
        alpha: float = 0.4,
        beta: float = 0.4,
        gamma: float = 0.2,
        ltn_threshold: float = 0.75,
        max_facts: int = 20,
    ) -> Tuple[List[RetrieverResultItem], Dict[str, Any]]:
        """
        Rerank retrieved documents using hybrid search scores, LTN validation, 
        and semantic similarity.
        """
        if not retrieved_docs:
            return [], {"predicted_facts": [], "doc_scores": []}
        
        num_docs = len(retrieved_docs)
        doc_scores = np.zeros(num_docs, dtype=float)
        doc_components = [{} for _ in range(num_docs)]
        scored_facts: List[Tuple[str, str, str, float, float, float, int]] = []
        
        # Track coverage statistics for this query
        ltn_validation_stats = {
            "total_triples": 0,
            "ltn_validated": 0,
            "missing_head": 0,
            "missing_tail": 0,
            "missing_relation": 0,
            "missing_items": [],  
        }
        

        texts_to_embed = [query]
        triple_texts = []
        
        for doc in retrieved_docs:
            md = getattr(doc, "metadata", {}) or {}
            h = md.get("source_text", "")
            r = md.get("relation_type", "")
            t = md.get("target_text", "")
            triple_text = f"{h} {r} {t}" if h and r and t else ""
            triple_texts.append(triple_text)
            if triple_text and triple_text not in texts_to_embed:
                texts_to_embed.append(triple_text)
        
        # Single batched API call for all embeddings
        triple_embedding_map = {}
        query_vec = None
        
        MAX_BATCH_SIZE = 50  # Mistral API limit
        MAX_TEXT_LENGTH = 2000  # Truncate long texts
        
        # Filter and truncate texts
        texts_to_embed_clean = []
        for text in texts_to_embed:
            if text and text.strip():
                truncated = text[:MAX_TEXT_LENGTH] if len(text) > MAX_TEXT_LENGTH else text
                texts_to_embed_clean.append(truncated)
        
        if texts_to_embed_clean:
            try:
                all_embeddings = []
                # Process in batches
                for i in range(0, len(texts_to_embed_clean), MAX_BATCH_SIZE):
                    batch = texts_to_embed_clean[i:i + MAX_BATCH_SIZE]
                    batch_embeddings = self.retriever._langchain_embeddings.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                
                # First embedding is query
                query_vec = np.array(all_embeddings[0])
                
                # Map remaining embeddings to their texts
                for i, text in enumerate(texts_to_embed_clean[1:], 1):
                    if i < len(all_embeddings):
                        triple_embedding_map[text] = np.array(all_embeddings[i])
                        
            except Exception as e:
                print(f"[WARNING] Batch embedding failed: {e}")
        
        # Normalize hybrid scores to [0, 1] range
        hybrid_scores = []
        for doc in retrieved_docs:
            md = getattr(doc, "metadata", {}) or {}
            hybrid_scores.append(float(md.get("score", 0.0)))
        
        max_hybrid = max(hybrid_scores) if hybrid_scores else 1.0
        min_hybrid = min(hybrid_scores) if hybrid_scores else 0.0
        hybrid_range = max_hybrid - min_hybrid if max_hybrid != min_hybrid else 1.0
        
        # ─────────────────────────────────────────────────────────────
        # Score each document
        # ─────────────────────────────────────────────────────────────
        for doc_idx, doc in enumerate(retrieved_docs):
            md = getattr(doc, "metadata", {}) or {}
            
            # Component 1: Normalized Hybrid Search Score
            raw_hybrid = float(md.get("score", 0.0))
            normalized_hybrid = (raw_hybrid - min_hybrid) / hybrid_range
            
            # Component 2: LTN Confidence
            h = md.get("source_text")
            r = md.get("relation_type")
            t = md.get("target_text")
            
            ltn_confidence = None
            has_valid_triple = False
            validation_failure_reason = None
            
            if self.ltn and h and r and t:
                ltn_validation_stats["total_triples"] += 1
                
                h_known = h in self.ltn.ent2id
                t_known = t in self.ltn.ent2id
                r_known = r in self.ltn.rel2id
                
                if h_known and t_known and r_known:
                    ltn_confidence = self.ltn.triple_confidence(h, r, t)
                    has_valid_triple = ltn_confidence is not None
                    if has_valid_triple:
                        ltn_validation_stats["ltn_validated"] += 1
                else:
                    # Track WHY validation failed
                    if not h_known:
                        ltn_validation_stats["missing_head"] += 1
                        ltn_validation_stats["missing_items"].append(("entity", h))
                    if not t_known:
                        ltn_validation_stats["missing_tail"] += 1
                        ltn_validation_stats["missing_items"].append(("entity", t))
                    if not r_known:
                        ltn_validation_stats["missing_relation"] += 1
                        ltn_validation_stats["missing_items"].append(("relation", r))
                    
                    validation_failure_reason = []
                    if not h_known: validation_failure_reason.append(f"head '{h}' unknown")
                    if not t_known: validation_failure_reason.append(f"tail '{t}' unknown")
                    if not r_known: validation_failure_reason.append(f"relation '{r}' unknown")
            
            # Component 3: Semantic Similarity (from batched embeddings)
            semantic_sim = 0.0
            triple_text = triple_texts[doc_idx]
            if query_vec is not None and triple_text in triple_embedding_map:
                triple_vec = triple_embedding_map[triple_text]
                q_norm = np.linalg.norm(query_vec)
                t_norm = np.linalg.norm(triple_vec)
                if q_norm > 0 and t_norm > 0:
                    semantic_sim = max(0.0, float(np.dot(query_vec, triple_vec) / (q_norm * t_norm)))
            
            # Combine Scores
            if has_valid_triple and ltn_confidence is not None:
                combined = (
                    alpha * normalized_hybrid +
                    beta * float(ltn_confidence) +
                    gamma * semantic_sim
                )
                if ltn_confidence >= ltn_threshold:
                    scored_facts.append((
                        h, r, t, 
                        float(ltn_confidence), 
                        semantic_sim, 
                        combined,
                        doc_idx
                    ))
            else:
                # Fallback: redistribute beta weight
                combined = (
                    (alpha + beta * 0.5) * normalized_hybrid +
                    (gamma + beta * 0.5) * semantic_sim
                )
            
            doc_scores[doc_idx] = combined
            doc_components[doc_idx] = {
                "hybrid_score": normalized_hybrid,
                "raw_hybrid": raw_hybrid,
                "ltn_confidence": ltn_confidence,
                "semantic_similarity": semantic_sim,
                "combined_score": combined,
                "has_ltn_validation": has_valid_triple,
                "ltn_failure_reason": validation_failure_reason,
            }
        
        # Sort facts by combined score
        scored_facts.sort(key=lambda x: x[5], reverse=True)
        top_facts = scored_facts[:max_facts]
        
        # Rerank documents by score
        indexed_docs = list(enumerate(retrieved_docs))
        indexed_docs.sort(key=lambda pair: doc_scores[pair[0]], reverse=True)
        reranked_docs = [doc for _, doc in indexed_docs]
        reranked_indices = [idx for idx, _ in indexed_docs]
        reranked_components = [doc_components[idx] for idx in reranked_indices]
        
        # Compute coverage rate for this query
        query_coverage_rate = (
            ltn_validation_stats["ltn_validated"] / ltn_validation_stats["total_triples"] * 100
            if ltn_validation_stats["total_triples"] > 0 else 0.0
        )
        
        # Deduplicate missing items
        unique_missing = {}
        for item_type, item_value in ltn_validation_stats["missing_items"]:
            if item_value not in unique_missing:
                unique_missing[item_value] = item_type
        
        ltn_result = {
            "predicted_facts": [
                (h, r, t, ltn_conf) 
                for h, r, t, ltn_conf, sem_sim, combined, doc_idx in top_facts
            ],
            "detailed_facts": [
                {
                    "head": h,
                    "relation": r,
                    "tail": t,
                    "ltn_confidence": ltn_conf,
                    "semantic_similarity": sem_sim,
                    "combined_score": combined,
                    "source_doc_idx": doc_idx,
                }
                for h, r, t, ltn_conf, sem_sim, combined, doc_idx in top_facts
            ],
            "doc_scores": doc_scores[reranked_indices].tolist(),
            "doc_components": reranked_components,
            "scoring_weights": {"alpha": alpha, "beta": beta, "gamma": gamma},
            # NEW: Coverage statistics for this query
            "ltn_coverage": {
                "total_triples": ltn_validation_stats["total_triples"],
                "validated": ltn_validation_stats["ltn_validated"],
                "coverage_rate": round(query_coverage_rate, 1),
                "failures": {
                    "missing_head": ltn_validation_stats["missing_head"],
                    "missing_tail": ltn_validation_stats["missing_tail"],
                    "missing_relation": ltn_validation_stats["missing_relation"],
                },
                "unique_missing_items": unique_missing,
            },
        }
        
        return reranked_docs, ltn_result
    
    
    def _calculate_cosine_similarity(
        self, 
        query: str, 
        h: str, 
        r: str, 
        t: str,
        use_cache: bool = True,
    ) -> float:
        """
        Calculate cosine similarity between query and triple using embeddings.
        
        Args:
            query: User query string
            h: Head entity text
            r: Relation type
            t: Tail entity text
            use_cache: Whether to cache embeddings (default True)
        
        Returns:
            Cosine similarity score in [0, 1]
        """
        # Initialize cache if not exists
        if not hasattr(self, "_embedding_cache"):
            self._embedding_cache = {}
        
        triple_text = f"{h} {r} {t}"
        
        try:
            # Get query embedding (with caching)
            if use_cache and query in self._embedding_cache:
                query_vec = self._embedding_cache[query]
            else:
                query_embedding = self.retriever.embeddings.embed_query(query)
                query_vec = np.array(query_embedding, dtype=float)
                if use_cache:
                    self._embedding_cache[query] = query_vec
            
            # Get triple embedding (with caching)
            if use_cache and triple_text in self._embedding_cache:
                triple_vec = self._embedding_cache[triple_text]
            else:
                triple_embedding = self.retriever.embeddings.embed_query(triple_text)
                triple_vec = np.array(triple_embedding, dtype=float)
                if use_cache:
                    self._embedding_cache[triple_text] = triple_vec
            
            # Compute cosine similarity
            query_norm = np.linalg.norm(query_vec)
            triple_norm = np.linalg.norm(triple_vec)
            
            if query_norm == 0.0 or triple_norm == 0.0:
                return 0.0
            
            cosine_sim = float(np.dot(query_vec, triple_vec) / (query_norm * triple_norm))
            
            # Clamp to [0, 1] (cosine can be negative for opposing vectors)
            return max(0.0, min(1.0, cosine_sim))
        
        except Exception as e:
            print(f"[WARNING] Embedding similarity calculation failed: {e}")
            return 0.0
    
    
    def clear_embedding_cache(self) -> None:
        """Clear the embedding cache to free memory."""
        if hasattr(self, "_embedding_cache"):
            self._embedding_cache.clear()
            
            
    def _verify_citations(
        self,
        response: str,
        docs: List[RetrieverResultItem],
        ltn_result: Dict,
    ) -> Tuple[str, bool]:
        """
        Verify that all citations in the response reference valid evidence.
        
        Args:
            response: The LLM-generated response text
            docs: List of retrieved documents
            ltn_result: Dictionary containing LTN predictions
        
        Returns:
            Tuple of (cleaned_response, all_citations_valid)
        """
        # ---------------- Validate citation indices -----------------
        all_valid = True
        citation_pattern = re.compile(r"\[(F|L)(\d+)\]")
        max_f_idx = len(docs) - 1 if docs else -1
        max_l_idx = len(ltn_result.get("predicted_facts", [])) - 1
    
        def check_citation(match: re.Match) -> str:
            nonlocal all_valid
            tag_type, idx_str = match.groups()
            idx = int(idx_str)
    
            if tag_type == "F" and (idx < 0 or idx > max_f_idx):
                all_valid = False
                return "[citation removed]"
            elif tag_type == "L" and (idx < 0 or idx > max_l_idx):
                all_valid = False
                return "[citation removed]"
            else:
                return match.group(0)
    
        response_checked = citation_pattern.sub(check_citation, response)
    
        return response_checked, all_valid
        
        
###############################################################################
# Usage
###############################################################################
def main():
    config = load_config()
    system = DiabetesDecisionSupport(config)

    while True:
        user_query = input("\nEnter your diabetes query (or 'exit' to quit): ")
        if user_query.lower() == 'exit':
            break
        if not user_query:
            print("Query cannot be empty.")
            continue
        
        print("\nProcessing general query...")
        result = system.query(user_query, mode="general")
        print("\n--- General Query Result ---")
        print(result['response'])
        

        # Evidence summary
        print("\n--- Evidence Summary ---")
        summary = result['evidence_summary']
        print(f"Graph facts used: {summary['graph_facts']}")
        print(f"Validated facts: {summary['validated_facts']}")
        if summary['confidence_range']:
            print(f"Confidence range: {min(summary['confidence_range']):.2f} - {max(summary['confidence_range']):.2f}")

        
            
            
        """ltn_cov = summary.get('ltn_coverage', {})
        if ltn_cov:
            print(f"\n--- LTN Validation Coverage ---")
            print(f"Triples validated: {ltn_cov.get('validated', 0)}/{ltn_cov.get('total_triples', 0)} "
                  f"({ltn_cov.get('coverage_rate', 0):.1f}%)")
            
            failures = ltn_cov.get('failures', {})
            if any(failures.values()):
                print(f"Validation failures:")
                if failures.get('missing_head', 0) > 0:
                    print(f"  - Unknown head entities: {failures['missing_head']}")
                if failures.get('missing_tail', 0) > 0:
                    print(f"  - Unknown tail entities: {failures['missing_tail']}")
                if failures.get('missing_relation', 0) > 0:
                    print(f"  - Unknown relations: {failures['missing_relation']}")
            
            missing = ltn_cov.get('unique_missing_items', {})
            if missing and len(missing) <= 10:  # Only show if not too many
                print(f"Missing items: {list(missing.keys())}")"""    
        # Reference index: map [F#] and [L#] to IDs/triples
        ref_idx = result.get("reference_index", {})
        kg_refs = ref_idx.get("kg", [])
        ltn_refs = ref_idx.get("ltn", [])

        """if kg_refs or ltn_refs:
            print("\n--- Reference Index (traceable IDs) ---")
            for ref in kg_refs:
                print(
                    f"[{ref['label']}] KG src={ref['source_id']}, rel={ref['relation_type']}, "
                    f"tgt={ref['target_id']} | {ref['text']}"
                )
            for ref in ltn_refs:
                print(
                    f"[{ref['label']}] LTN {ref['head']} --[{ref['relation']}]--> {ref['tail']} "
                    f"(conf={ref['confidence']:.2f})"
                )"""


if __name__ == "__main__":
    main()