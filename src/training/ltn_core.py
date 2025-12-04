# ltn_core.py

import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import List, Dict, Tuple
from dataclasses import dataclass
from typing import List
import json, csv, math, statistics, random, re
from pathlib import Path
import re





# ---------- dataclasses ----------
@dataclass
class Atom:
    s_kind: str
    s_id: int
    r_id: int
    o_kind: str
    o_id: int


@dataclass
class Rule:
    num_vars: int
    head: Atom
    body: List[Atom]

    # one learnable parameter per rule, initialised later
    log_lam: torch.nn.Parameter = None       # to be filled in after construction

    @property
    def lam(self) -> torch.Tensor:
        """Positive rule weight λ = exp(log_lam)."""
        return self.log_lam.exp()
    
    

# ---------- parsing / loading ----------

def _split_fields(s: str) -> list[str]:
    """Return AMIE fields separated by ≥2 whitespace (or tabs)."""
    return re.split(r'\s{2,}|\t', s.strip())

def _fields_to_triples(fields: list[str]) -> list[tuple[str, str, str]]:
    if len(fields) % 3:
        raise ValueError(f"Bad AMIE rule – #fields ≠ 3 × n: {fields}")
    return [
        (fields[i], fields[i + 1], fields[i + 2])
        for i in range(0, len(fields), 3)
    ]


def _tok_to_kind_id(tok: str,
                    var2idx: Dict[str, int],
                    ent2id: Dict[str, int],
                    next_ent: int
                    ) -> Tuple[str, int, int]:
    """
    Returns (kind,id,next_ent).
      kind ∈ {'var','const'}
      id   = var‑idx  or  entity‑id
    """
    if tok.startswith('?'):
        if tok not in var2idx:
            var2idx[tok] = len(var2idx)
        return 'var', var2idx[tok], next_ent
    else:
        if tok not in ent2id:
            ent2id[tok] = next_ent
            next_ent += 1
        return 'const', ent2id[tok], next_ent



def load_rules(data_dir: str,
               ent2id: Dict[str, int],
               rel2id: Dict[str, int],
               next_ent: int,
               next_rel: int
               ) -> Tuple[List[Rule], int, int]:
    """Parse empirical + AMIE rules → List[Rule]."""
    rules: List[Rule] = []

    # --- AMIE CSV ------------------------------------------------------
    amie_path = Path(data_dir) / "AMIE-rules.csv"
    
    with open(amie_path) as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        pca_idx = hdr.index("PCA Confidence")
    
        for row in rdr:
            rule_txt   = row[0].strip()
            confidence = float(row[pca_idx])
            lam = -math.log(1 - confidence + 1e-6)
    
            body_txt, head_txt = (x.strip() for x in rule_txt.split("=>"))
    
            # ---- HEAD ------------------------------------------------------
            head_fields = _split_fields(head_txt)
            head_s, head_r, head_o = _fields_to_triples(head_fields)[0]
    
            var2idx = {}
            kind_s, idx_s, next_ent = _tok_to_kind_id(head_s, var2idx, ent2id, next_ent)
            kind_o, idx_o, next_ent = _tok_to_kind_id(head_o, var2idx, ent2id, next_ent)
    
            if head_r not in rel2id:
                rel2id[head_r], next_rel = next_rel, next_rel + 1
            head = Atom(kind_s, idx_s, rel2id[head_r], kind_o, idx_o)
    
            # ---- BODY ------------------------------------------------------
            body_fields = _split_fields(body_txt)
            body_atoms  = []
            for s_tok, r_tok, o_tok in _fields_to_triples(body_fields):
                if r_tok not in rel2id:
                    rel2id[r_tok], next_rel = next_rel, next_rel + 1
                kind_s, idx_s, next_ent = _tok_to_kind_id(s_tok, var2idx, ent2id, next_ent)
                kind_o, idx_o, next_ent = _tok_to_kind_id(o_tok, var2idx, ent2id, next_ent)
                body_atoms.append(Atom(kind_s, idx_s, rel2id[r_tok], kind_o, idx_o))
    
            rule = Rule(len(var2idx), head, body_atoms)
            rule.log_lam = torch.nn.Parameter(torch.log(torch.tensor(lam, dtype=torch.float32)))
            rules.append(rule)
        

    med_amie = statistics.median([r.lam for r in rules if len(r.body) > 1])
    for r in rules:
        if r.num_vars == 1:
            r.log_lam.data = torch.log((2 * med_amie).clone())
            r.log_lam.requires_grad_(False)

    return rules, next_ent, next_rel

def load_triples(path, ent2id, rel2id):
    """
    Load a JSON array of either
      • dicts with keys 'h','r','t'
      • lists/tuples [h,r,t]
    and return a list of (h_id, r_id, t_id) tuples.
    """
    raw = json.load(open(path))
    out = []
    for item in raw:
        if isinstance(item, dict):
            h, r, t = item['h'], item['r'], item['t']
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            h, r, t = item
        else:
            raise ValueError(f"Unrecognized triple format: {item!r}")
        out.append((ent2id[h], rel2id[r], ent2id[t]))
    return out



# ---------- rule evaluation ----------

def _eval_atom(model, atom, bound, h_batch, t_batch, all_ent_emb, device):
    """
    Fast atom evaluation using pre-cached entity embeddings.
    
    Args:
        model: BilinearLTN model
        atom: Atom dataclass instance
        bound: dict mapping var_id -> tensor of bound entity indices
        h_batch: (B,) head entity indices from triples
        t_batch: (B,) tail entity indices from triples
        all_ent_emb: (E, D) pre-cached entity embeddings
        device: torch device
        
    Returns:
        torch.Tensor: (B,) fuzzy truth values in [0, 1]
    """
    B = h_batch.size(0)
    E, D = all_ent_emb.shape
    
    # Get relation matrix once
    W = model.rel.weight[atom.r_id].view(D, D)  # (D, D)
    
    # ─────────────────────────────────────────────────────────
    # Determine subject binding
    # ─────────────────────────────────────────────────────────
    if atom.s_kind == 'var' and atom.s_id in bound:
        # Variable is bound to batch of entity indices
        h_emb = model.ent(bound[atom.s_id])  # (B, D)
        subj_existential = False
    elif atom.s_kind == 'var':
        # Unbound variable - existential quantification needed
        subj_existential = True
        h_emb = None
    else:
        # Constant - same entity for all batch elements
        h_emb = all_ent_emb[atom.s_id].unsqueeze(0).expand(B, -1)  # (B, D)
        subj_existential = False
    
    # ─────────────────────────────────────────────────────────
    # Determine object binding
    # ─────────────────────────────────────────────────────────
    if atom.o_kind == 'var' and atom.o_id in bound:
        t_emb = model.ent(bound[atom.o_id])  # (B, D)
        obj_existential = False
    elif atom.o_kind == 'var':
        obj_existential = True
        t_emb = None
    else:
        t_emb = all_ent_emb[atom.o_id].unsqueeze(0).expand(B, -1)  # (B, D)
        obj_existential = False
    
    # ─────────────────────────────────────────────────────────
    # Case 1: No existential - direct bilinear score
    # ─────────────────────────────────────────────────────────
    if not subj_existential and not obj_existential:
        scores = torch.einsum('bd,de,be->b', h_emb, W, t_emb)
        return torch.sigmoid(scores)
    
    # ─────────────────────────────────────────────────────────
    # Case 2: Existential on object only (most common case)
    # max over all possible tail entities: max_t (h @ W @ t)
    # ─────────────────────────────────────────────────────────
    if not subj_existential and obj_existential:
        # h_emb @ W gives (B, D), then @ all_ent_emb.T gives (B, E)
        hW = torch.einsum('bd,de->be', h_emb, W)  # (B, D)
        scores = hW @ all_ent_emb.T  # (B, E)
        return torch.sigmoid(scores.max(dim=1).values)
    
    # ─────────────────────────────────────────────────────────
    # Case 3: Existential on subject only
    # max over all possible head entities: max_h (h @ W @ t)
    # ─────────────────────────────────────────────────────────
    if subj_existential and not obj_existential:
        # W @ t_emb.T gives (D, B), then all_ent_emb @ that gives (E, B)
        Wt = torch.einsum('de,be->db', W, t_emb)  # (D, B)
        scores = all_ent_emb @ Wt  # (E, B)
        return torch.sigmoid(scores.max(dim=0).values)
    
    # ─────────────────────────────────────────────────────────
    # Case 4: Both existential (rare, use sampling approximation)
    # Full E×E is O(E²) which is too expensive for large KGs
    # ─────────────────────────────────────────────────────────
    k = min(256, E)  # Sample size for approximation
    sample_idx = torch.randperm(E, device=device)[:k]
    sampled_emb = all_ent_emb[sample_idx]  # (k, D)
    
    # Compute max over sampled (h, t) pairs
    hW = sampled_emb @ W  # (k, D)
    scores = hW @ sampled_emb.T  # (k, k)
    return torch.sigmoid(scores.max().expand(B))

def evaluate_rules(model, triples, rules, entity_ids, *, chunk=128, return_groundings=False):
    """
    Compute μ_rule for each rule (no λ yet).
    
    OPTIMIZED VERSION: Uses cached embeddings and vectorized operations.

    Args
    ----
    model        : BilinearLTN with score(h,r,t) → logits
    triples      : (B,3) batch of (h,r,t)   — used to bind head-vars
    rules        : list[Rule]
    entity_ids   : (|E|,) tensor with all entity ids
    chunk        : (kept for API compatibility, not used in optimized version)
    return_groundings: bool, if True, return groundings for the body

    Returns
    -------
    μ : (len(rules), B) tensor with fuzzy truth-values of every rule
    groundings: if return_groundings is True, a list of groundings for each rule
    """
    device = triples.device
    h, _, t = triples.t()
    B = h.size(0)
    
    # ─────────────────────────────────────────────────────────────
    # OPTIMIZATION: Cache all entity embeddings once per batch
    # ─────────────────────────────────────────────────────────────
    all_ent_emb = model.get_all_entity_embeddings()  # (E, D) - no copy
    
    outs = []
    groundings = [] if return_groundings else None

    for rule in rules:
        # ─────────────────────────────────────────────────────────
        # HEAD: bind variables from the input triples
        # ─────────────────────────────────────────────────────────
        bound = {}
        
        def bind(kind, idx, default_vec):
            if kind == 'var':
                bound[idx] = default_vec
                return default_vec
            return torch.full_like(default_vec, idx)

        Hh = bind(rule.head.s_kind, rule.head.s_id, h)
        Rh = torch.full_like(Hh, rule.head.r_id)
        Th = bind(rule.head.o_kind, rule.head.o_id, t)
        μ_head = torch.sigmoid(model.score(Hh, Rh, Th))

        # ─────────────────────────────────────────────────────────
        # BODY: evaluate each atom using optimized function
        # ─────────────────────────────────────────────────────────
        μ_body = torch.ones_like(μ_head)
        rule_groundings = [] if return_groundings else None

        for atom in rule.body:
            # Use optimized atom evaluation
            μ_atom = _eval_atom(
                model, atom, bound, h, t, all_ent_emb, device
            )
            
            # Łukasiewicz t-norm: T(a,b) = max(0, a + b - 1)
            μ_body = torch.clamp(μ_body + μ_atom - 1.0, min=0.)
            
            if return_groundings:
                # Simplified grounding info (detailed tracking removed for speed)
                rule_groundings.append({
                    "atom": atom,
                    "grounding": "optimized_eval"
                })

        # Łukasiewicz implication: body → head = min(1, 1 - body + head)
        μ_rule = torch.clamp(1.0 - μ_body + μ_head, max=1.0)
        outs.append(μ_rule)
        
        if return_groundings:
            groundings.append(rule_groundings)

    if return_groundings:
        return torch.stack(outs), groundings
    return torch.stack(outs)

class RuleBank(nn.Module):
    """
    Thin container so torch.optim can see rule.log_lam parameters.
    """
    def __init__(self, rules: List[Rule]):
        super().__init__()
        # register each log_lam as a sub‑parameter
        for i, r in enumerate(rules):
            self.register_parameter(f"log_lam_{i}", r.log_lam)

    def forward(self):
        raise RuntimeError("RuleBank is a parameter holder only.")

def validate(model,
             valid_triples,
             ent_ids,
             device,
             num_samples: int,
             hits_ks):
    """
    Link-prediction evaluation (binary).

    Returns
    -------
    stats : dict with keys
        'AUC', 'Accuracy', 'MRR', 'Hits@k' for every k in hits_ks.
    """
    model.eval()
    with torch.no_grad():

        # -------- positives ----------
        pos = torch.tensor(valid_triples, device=device)        # (N,3)
        N   = pos.size(0)
        pos_logit = model.score(pos[:, 0], pos[:, 1], pos[:, 2])  # (N,)

        # -------- negatives ----------
        K = num_samples
        tails = torch.randint(0, len(ent_ids), (N, K), device=device)
        neg = pos.unsqueeze(1).expand(N, K, 3).clone()
        neg[:, :, 2] = tails
        neg_logit = model.score(neg[:, :, 0].flatten(),
                                neg[:, :, 1].flatten(),
                                neg[:, :, 2].flatten()).view(N, K)        # (N,K)

        # -------- ranks for MRR / Hits@k ----------
        all_scores = torch.cat([pos_logit.unsqueeze(1), neg_logit], 1)    # (N,K+1)
        ranks = (all_scores > pos_logit.unsqueeze(1)).sum(1) + 1          # (N,)

        # -------- flat vectors for AUC / Accuracy ----------
        pos_sig = torch.sigmoid(pos_logit)           # (N,)
        neg_sig = torch.sigmoid(neg_logit.flatten()) # (N·K,)

        y_scores = torch.cat([pos_sig, neg_sig])                         # 1-D
        y_true   = torch.cat([torch.ones_like(pos_sig),
                              torch.zeros_like(neg_sig)])               # 1-D

        auc  = roc_auc_score(y_true.cpu(), y_scores.cpu())
        acc  = accuracy_score(y_true.cpu(), (y_scores > 0.5).cpu())

        stats = {
            'AUC'      : auc,
            'Accuracy' : acc,
            'MRR'      : (1.0 / ranks.float()).mean().item(),
            **{f'Hits@{k}': (ranks <= k).float().mean().item() for k in hits_ks}
        }

    model.train()
    return stats
    
    


#  RULE-AWARE METRICS  (mean satisfaction & weighted rule coverage)
# ----------------------------------------------------------------------
def rule_metrics(model,
                 triples: torch.Tensor,     # (N,3)  validation positives
                 rules:   List[Rule],
                 entity_ids: torch.Tensor,
                 device: torch.device):
    """
    Returns a pair:
        mean_sat   : average sigmoid(score) over *triples*
        wr_cov     : Σ_g λ_g μ̄_g / Σ_g λ_g   (λ truncated to match μ̄ len)

    If evaluate_rules() returns fewer rows than `rules` (because some rules
    cannot be grounded) we truncate λ to the common prefix to avoid a size
    mismatch.
    """
    model.eval()
    with torch.no_grad():
        # -- mean satisfaction of plain KG triples -----------------------
        h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
        mean_sat = torch.sigmoid(model.score(h, r, t)).mean().item()

        # -- rule coverage ----------------------------------------------
        
        max_rule_triples = 512
        if triples.size(0) > max_rule_triples:
            perm = torch.randperm(triples.size(0), device=device)
            triples_sampled = triples[perm[:max_rule_triples]]
        else:
            triples_sampled = triples
            
        μ = evaluate_rules(model, triples, rules, entity_ids)  # (R′, N)
        μ_bar = μ.mean(dim=1)                                 # (R′,)

        λ_all = torch.stack([
            torch.exp(r.log_lam) if r.log_lam is not None else
            torch.tensor(r.lam, device=device)
            for r in rules
        ])

        R_eff = min(len(λ_all), len(μ_bar))       # common prefix length
        if R_eff == 0:
            return mean_sat, 0.0                  # no evaluable rules

        λ = λ_all[:R_eff]
        μ_bar = μ_bar[:R_eff]

        wr_cov = (λ * μ_bar).sum() / λ.sum()

    model.train()
    return mean_sat, wr_cov.item()

    
    
# ---------- model ----------
class BilinearLTN(nn.Module):
    def __init__(self, n_ent, n_rel, dim=256):
        super().__init__()
        self.dim = dim
        self.ent = nn.Embedding(n_ent, dim)
        self.rel = nn.Embedding(n_rel, dim * dim)
        nn.init.xavier_uniform_(self.ent.weight)
        nn.init.xavier_uniform_(self.rel.weight.view(-1, dim, dim))
        

   
    def score(self, h, r, t):
      """
      Vectorised triple scoring.
  
      Args
      ----
      h, r, t : LongTensor of shape (B,)
          Indices of head entity, relation and tail entity.
  
      Returns
      -------
      Tensor of shape (B,) – higher means the triple is considered true.
      """
  
      # ---------- bilinear score for *all* triples ----------
      W      = self.rel(r).view(-1, self.dim, self.dim)      # (B,D,D)
      h_emb  = self.ent(h).unsqueeze(1)                      # (B,1,D)
      t_emb  = self.ent(t).unsqueeze(2)                      # (B,D,1)
      scores = (h_emb @ W @ t_emb).squeeze(-1).squeeze(-1)   # (B,)
  

      return scores
      
    def get_all_entity_embeddings(self):
        """Cache all entity embeddings for existential quantification."""
        return self.ent.weight  # (E, D) - no copy, just reference
    
    def score_with_cached_emb(self, h_emb, r, t_emb):
        """Score using pre-fetched embeddings."""
        W = self.rel(r).view(-1, self.dim, self.dim)
        return torch.einsum('bd,bde,be->b', h_emb, W, t_emb)
    

# ---------- ltn loss -----------------------------------------------------
def ltn_loss(triples, labels, model, rules, entity_ids,
             rule_sample_frac: float = 0.3,
             kg_mask: torch.Tensor | None = None,
             λ_kg: float = 100.0):
    """
    Return
        bce, regularization, kg_penalty
    (three tensors, each carries .grad_fn so they stay in the graph)
    """
    h, r, t = triples[:, 0], triples[:, 1], triples[:, 2]
    device = triples.device
    
    # --- 1) supervised BCE ----------------------------------------------
    logits = model.score(h, r, t)
    bce = F.binary_cross_entropy_with_logits(logits, labels)

    # --- 2) KG hard constraint  (not added to bce – handled in train loop)
    if kg_mask is not None and kg_mask.any():
        pos_logits = torch.sigmoid(logits[kg_mask])
        kg_pen = ((1.0 - pos_logits) ** 2).mean()
    else:
        kg_pen = torch.tensor(0.0, device=bce.device)

    # --- 3) fuzzy-rule regulariser  (no α scaling here) ------------------
    k = max(1, int(rule_sample_frac * len(rules)))
    if k < len(rules):
        sample_indices = torch.randperm(len(rules), device=device)[:k]
        sampled = [rules[i] for i in sample_indices.tolist()]
    else:
        sampled = rules
    

    μ = evaluate_rules(model, triples, sampled, entity_ids)      # (k,B)
    μ_mean = μ.mean(dim=1)                                       # universal ≈ mean

    lams = torch.stack([
        torch.exp(r_obj.log_lam) if r_obj.log_lam is not None 
        else torch.tensor(r_obj.lam, device=device, dtype=torch.float32)
        for r_obj in sampled
    ])  # (k,)
    regularization = (lams * (1.0 - μ_mean) ** 2).mean()

    return bce, regularization, kg_pen



#------------------------SAMPLES--------------------------

def sample_rule_triples(rules, entity_ids, device, k_pos=4, k_neg=4):
    """
    For a random subset of rules:
        • choose concrete entity IDs for every variable
        • emit the HEAD triple as a *positive*
        • flip the tail entity to a random different one → *negative*
    Returns two tensors: rule_pos, rule_neg  (may be empty)
    """
    if len(rules) == 0:
        return None, None

    chosen = random.sample(rules, min(len(rules), k_pos))
    pos_rows, neg_rows = [], []
    E = entity_ids

    for rule in chosen:
        # 1) pick random entities for each variable
        var_bind = {i: int(E[torch.randint(0, len(E), (1,))]) for i in range(rule.num_vars)}
        def _get_id(kind, idx):
            return var_bind[idx] if kind == "var" else idx

        # 2) build HEAD triple
        h = _get_id(rule.head.s_kind, rule.head.s_id)
        r = rule.head.r_id
        t = _get_id(rule.head.o_kind, rule.head.o_id)
        pos_rows.append((h, r, t))

        # 3) negative: corrupt tail (ensure different)
        t_neg = t
        while t_neg == t:
            t_neg = int(E[torch.randint(0, len(E), (1,))])
        neg_rows.append((h, r, t_neg))

    if not pos_rows:
        return None, None
    return (torch.tensor(pos_rows, device=device, dtype=torch.long),
            torch.tensor(neg_rows, device=device, dtype=torch.long))


def sample_batch(batch_size: int,
                 KG: torch.Tensor,
                 entity_ids: torch.Tensor,
                 device: torch.device,
                 rules: List[Rule] | None = None,
                 neg_per_pos: int = 1):
    """
    Returns
        triples : (B,3) LongTensor
        labels  : (B,)  FloatTensor  (1 = positive, 0 = negative)
        kg_mask : (B,)  BoolTensor   True where row is a KG positive
    """

    # -------- 1. how many positives / negatives ------------------------
    n_pos = batch_size // (1 + neg_per_pos)      
    n_neg = batch_size - n_pos
    n_entities = len(entity_ids)
    
    triples = torch.empty((batch_size, 3), dtype=torch.long, device=device)
    labels = torch.empty(batch_size, dtype=torch.float, device=device)
    kg_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)

    # -------- 2. sample KG positives -----------------------------
    idx_kg  = torch.randint(0, KG.size(0),  (n_pos,),  device=device)
    triples[:n_pos] = KG[idx_kg]
    labels[:n_pos] = 1.0
    kg_mask[:n_pos] = True
    n_kg = n_pos

    # -------- 3. sample random negatives -------------------------------
    # Repeat positive indices to match negative count
    repeats_needed = (n_neg + n_pos - 1) // n_pos
    neg_base_idx = idx_kg.repeat(repeats_needed)[:n_neg]
    
    triples[n_pos:batch_size] = KG[neg_base_idx]
    # Corrupt tails with random entities
    triples[n_pos:batch_size, 2] = torch.randint(0, n_entities, (n_neg,), device=device)
    labels[n_pos:batch_size] = 0.0

    # -------- 4. add rule-derived examples ------------------
    if rules is not None:
        rule_pos, rule_neg = sample_rule_triples(rules, entity_ids, device)
        if rule_pos is not None:
            n_rule_pos = rule_pos.size(0)
            n_rule_neg = rule_neg.size(0)
            
            triples = torch.cat([triples, rule_pos, rule_neg], dim=0)
            labels = torch.cat([
                labels,
                torch.ones(n_rule_pos, device=device),
                torch.zeros(n_rule_neg, device=device)
            ])
            kg_mask = torch.cat([
                kg_mask,
                torch.zeros(n_rule_pos + n_rule_neg, dtype=torch.bool, device=device)
            ])         


    # -------- 5. random shuffle ----------------------------------------
    perm = torch.randperm(triples.size(0), device=device)
    return triples[perm], labels[perm], kg_mask[perm]