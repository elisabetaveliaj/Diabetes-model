# main_train.py

import os
import random
import json
from pathlib import Path
import numpy as np
import torch
from torch.amp import GradScaler, autocast


import time
from contextlib import contextmanager

@contextmanager
def timer(name, enabled=True):
    """Context manager for timing code blocks."""
    if enabled:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
    yield
    if enabled:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        print(f"[TIMER] {name}: {elapsed_ms:.1f}ms")


# reusable helpers
from ltn_core import (
    load_rules, load_triples, BilinearLTN, 
    ltn_loss, validate, sample_batch, RuleBank, rule_metrics, Rule
)

import collections
try:
    import wandb                                 
    wandb.init(project="ltn_alpha_sweep", reinit=True)
except ImportError:
    wandb = None


# ─────────────────── Reproducibility ───────────────────
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True, warn_only=True)

device   = 'cuda' if torch.cuda.is_available() else 'cpu'
#scaler = GradScaler() if device == 'cuda' else None
DATA_DIR = Path("/data/")
METRICS_PATH = DATA_DIR / "ltn_metrics.json"
TEST_METRICS_PATH = DATA_DIR / "ltn_test_metrics.json"

# ─────────────────── 1.  Load entity / relation maps ───────────────────
ent2id = json.load(open(DATA_DIR/"entities.json"))
rel2id = json.load(open(DATA_DIR/"relations.json"))
next_ent = max(ent2id.values()) + 1
next_rel = max(rel2id.values()) + 1


# ─────────────────── 2.  KG triples ───────────────────
train_triples = load_triples(DATA_DIR/"train_triples.json", ent2id, rel2id)
valid_triples = load_triples(DATA_DIR/"valid_triples.json", ent2id, rel2id)
test_triples  = load_triples(DATA_DIR/"test_triples.json" , ent2id, rel2id)
KG = torch.as_tensor(train_triples, dtype=torch.long)       # (N,3)

# ─────────────────── 3.  Rules (multi‑var) ─────────────
rules, next_ent, next_rel = load_rules(DATA_DIR, ent2id, rel2id,
                                       next_ent, next_rel)

# give every rule a learnable log‑λ
for r in rules:
    r.log_lam = torch.nn.Parameter(torch.log(r.lam.clone().to(device)))
rule_bank = RuleBank(rules)    

# ─────────────────── 4.  Build model & optimiser ───────
model = BilinearLTN(len(ent2id), len(rel2id), dim=256).to(device)

#if hasattr(torch, 'compile'):
    #model = torch.compile(model, mode="reduce-overhead")

optim = torch.optim.AdamW(
    list(model.parameters()) + list(rule_bank.parameters()),
    lr=1e-3, weight_decay=1e-5
)
sched  = torch.optim.lr_scheduler.StepLR(optim, step_size=50, gamma=0.5)
scaler = GradScaler() if device == 'cuda' else None

# helpers
entity_ids = torch.arange(len(ent2id), device=device)
rel_ids    = torch.arange(len(rel2id), device=device)

# ─────────────────── 5.  Training loop ────────────────

METRICS_PATH = DATA_DIR / "ltn_metrics.json"
TEST_METRICS_PATH = DATA_DIR / "ltn_test_metrics.json"

# ─────────────────── 6.  Training loop ────────────────

def train(
    alphas=(2.0,),
    *,
    epochs: int = 70,
    batch_size: int = 64,
    λ_kg: float = 50.0,                 # strength of KG hard constraint
):
    """
    Train the LTN model for each α in *alphas*.
    Writes incremental snapshots to ``ltn_metrics.json`` 
    """
    metrics: list[dict] = []
    initial_model_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
    initial_rule_state  = {k: v.clone().detach() for k, v in rule_bank.state_dict().items()}


    for α in alphas:
        print(f"\n=== α = {α} ===")
            # --- reset model & rules ------------------------------------------
        model.load_state_dict(initial_model_state, strict=True)
        rule_bank.load_state_dict(initial_rule_state, strict=True)

        num_batches = max(1, KG.size(0) // batch_size)
        
        optim = torch.optim.AdamW(
            list(model.parameters()) + list(rule_bank.parameters()),
            lr=1e-3,
            weight_decay=1e-5,
        )
        sched = torch.optim.lr_scheduler.StepLR(
            optim,
            step_size=50,  
            gamma=0.5        
        )

        for epoch in range(epochs):
            epoch_bce, epoch_reg, epoch_kg = [], [], []
            tot = 0.0

            for batch_idx in range(num_batches):
                triples, labels, kg_mask = sample_batch(
                    batch_size, KG, entity_ids, device, neg_per_pos=1, rules=rules
                )

                optim.zero_grad()

                if scaler is not None:
                    with autocast(device_type="cuda", dtype=torch.float16):
                        bce, reg, kg_pen = ltn_loss(
                            triples,
                            labels,
                            model,
                            rules,
                            entity_ids,
                            kg_mask=kg_mask,
                            λ_kg=λ_kg,           
                        )
                        loss = bce + λ_kg * kg_pen + α * reg
                        scaler.scale(loss).backward()
                        scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(rule_bank.parameters()), 1.0)
                        scaler.step(optim)
                        scaler.update()
                else:
                    bce, reg, kg_pen = ltn_loss(
                        triples,
                        labels,
                        model,
                        rules,
                        entity_ids,
                        kg_mask=kg_mask,
                        λ_kg=λ_kg,           
                    )
                    loss = bce + λ_kg * kg_pen + α * reg                
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(rule_bank.parameters()), 1.0)
                    optim.step()
                    


                # -------- bookkeeping per batch ----------
                epoch_bce.append(bce.item())
                epoch_reg.append(reg.item())
                epoch_kg.append(kg_pen.item())
                
            tot += loss.item()
            
        
            # ---- end epoch -------------------------------------------
            sched.step()
            mean_bce = float(np.mean(epoch_bce))
            mean_reg = float(np.mean(epoch_reg))
            mean_kg  = float(np.mean(epoch_kg))
            total_loss = mean_bce + λ_kg * mean_kg + α * mean_reg
            rule_ratio = (α * mean_reg) / (total_loss + 1e-9)

            print(
                f"Epoch {epoch:03d}  "
                f"loss={loss:.4f}  "
                f"Rule={α*mean_reg:.4f}  "
                f"KG={λ_kg*mean_kg:.4f}  "
                f"(rule/total={rule_ratio:.2%})"
            )

            if wandb is not None:
                wandb.log(
                    {
                        "alpha": α,
                        "epoch": epoch,
                        "loss_bce": mean_bce,
                        "loss_rule": α * mean_reg,
                        "loss_kg": λ_kg * mean_kg,
                        "ratio_rule_to_total": rule_ratio,
                    }
                )

            # --- quick dev-set check every 10 epochs ------------------
            if epoch % 10 == 0:
                stats = validate(
                    model,
                    valid_triples,
                    entity_ids,
                    device,
                    num_samples=50,
                    hits_ks=(1, 3, 10),
                )
                
                mean_sat, wr_cov = rule_metrics(
                    model,
                    torch.tensor(valid_triples, device=device),
                    rules,
                    entity_ids,
                    device,
                )
                stats["MeanSatisfaction"]     = mean_sat
                stats["WeightedRuleCoverage"] = wr_cov
                print("  → VAL:", {k: f"{v:.3f}" for k, v in stats.items()})

                snapshot = {
                    "alpha": α,
                    "epoch": epoch,
                    "loss": round(mean_bce + α * mean_reg + λ_kg * mean_kg, 4),
                    **{k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()},
                }
                metrics.append(snapshot)

                # persist after every validation step
                with open(METRICS_PATH, "w") as f:
                    json.dump(metrics, f, indent=2)
                    
    torch.cuda.empty_cache()

    # ------------- after all α runs: save final state -------------------
    def serialise_rule(rule: Rule) -> dict:
      return {
          "num_vars": rule.num_vars,
          "head": vars(rule.head),                    # -> dict with 5 fields
          "body": [vars(a) for a in rule.body],
          "lambda": rule.lam.item(),                 # positive weight
      }

    rules_serialised = [serialise_rule(r) for r in rules]
    
    ckpt = {
        "model": model.state_dict(),
        "rule_bank": rule_bank.state_dict(),          # λ in log space (kept for gradients)
        "rules_serialised": rules_serialised,
        "hyper": {"dim": model.dim},
    }
    torch.save(ckpt, DATA_DIR / "ltn_final.pt")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics



# ─────────────────── 7.  Entry point ──────────────────
if __name__ == "__main__":
    combined = np.vstack([train_triples, valid_triples])
    KG = torch.as_tensor(combined, dtype=torch.long).to(device)
    KG = KG.to(device)

    train()
    with open(DATA_DIR / "entities_after.json", "w") as f:
        json.dump(ent2id, f, indent=2)

    with open(DATA_DIR / "relations_after.json", "w") as f:
        json.dump(rel2id, f, indent=2)

    print("Saved regenerated entities.json and relations.json")


    print("\n=== TEST RESULTS ===")
    test_stats = validate(
        model,
        test_triples,
        entity_ids,
        device,
        num_samples=50,
        hits_ks=(1, 3, 10),
    )
    mean_sat, wr_cov = rule_metrics(
                    model,
                    torch.tensor(test_triples, device=device),
                    rules,
                    entity_ids,
                    device,
                )
    test_stats["MeanSatisfaction"]     = mean_sat
    test_stats["WeightedRuleCoverage"] = wr_cov
    test_stats_rounded = {k: round(v, 4) for k, v in test_stats.items()}

    # Persist to separate JSON for clarity -------------------------------
    with open(TEST_METRICS_PATH, "w") as f:
        json.dump(test_stats_rounded, f, indent=2)

    # Pretty console printout --------------------------------------------
    print("\n=== TEST RESULTS ===")
    for k, v in test_stats_rounded.items():
        print(f"{k:8s}: {v:.4f}")
