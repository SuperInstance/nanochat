# INTEGRATION.md — nanochat × SuperInstance Ecosystem

This document describes how nanochat wires into the SuperInstance ecosystem of crates, forming a closed loop from application code to locally-trained models and back.

---

## Architecture Overview

```
┌─────────────┐     CAPABILITY.toml     ┌───────────────────┐
│  Rust Crates │ ──────────────────────→ │ synthetic data    │
│  (openmind)  │   cellular decomposition│ generation        │
└─────────────┘                          └────────┬──────────┘
                                                  │ JSONL
                                                  ▼
┌─────────────┐     training config    ┌───────────────────┐
│   autoclaw   │ ──────────────────────→│    nanochat        │
│  (orchestr.) │                        │  train → evaluate  │
└──────┬──────┘                         └────────┬──────────┘
       │                                         │
       │  model ready                            │ metrics
       ▼                                         ▼
┌─────────────┐     embeddings        ┌───────────────────┐
│  Application │ ←───────────────────  │   open_vectors    │
│  (inference) │   similarity search   │   (storage)       │
└─────────────┘                        └───────────────────┘
```

---

## 1. nanochat + autoclaw: Automated Training on Single-GPU

autoclaw discovers what needs training and hands the configuration to nanochat. nanochat doesn't care *why* it's training — it just trains.

### How It Works

1. **autoclaw** scans the SuperInstance crate graph and identifies capability gaps
2. It generates a training manifest (dataset path, depth, stages, eval targets)
3. nanochat reads the manifest and executes the full pipeline
4. The trained model is deposited for local inference

### Example: autoclaw Generates a Training Run

```python
# autoclaw creates this manifest, nanochat consumes it
import json
from pathlib import Path

manifest = {
    "run_id": "si-conservation-law-v1",
    "stages": ["pretrain", "sft"],
    "depth": 12,  # small model for domain-specific knowledge
    "dataset": {
        "pretrain": "data/conservation_law_synthetic.jsonl",
        "sft": "data/conservation_law_qa.jsonl",
    },
    "eval": {
        "benchmarks": ["humaneval"],
        "custom_tasks": ["tasks/conservation_eval.py"],
    },
    "budget": {
        "max_gpu_hours": 4.0,
        "gamma_target": 0.85,  # conservation-law: 85% productive compute
    },
    "output": "runs/conservation-law-v1/",
}

Path("runs/autoclaw_manifests/conservation_law_v1.json").write_text(
    json.dumps(manifest, indent=2)
)
```

### Running It

```bash
# autoclaw discovers the gap and queues the run
python -m autoclaw scan --crates ./crates --queue-missing

# nanochat picks up the manifest and trains
python -m scripts.base_train --manifest runs/autoclaw_manifests/conservation_law_v1.json
python -m scripts.chat_sft --manifest runs/autoclaw_manifests/conservation_law_v1.json

# The trained model is now available for local inference
python -m scripts.chat_web --checkpoint runs/conservation-law-v1/ckpt.pt
```

---

## 2. nanochat + openmind: Codebase → Synthetic Training Data

openmind decomposes codebases into cellular units (each described by a CAPABILITY.toml). Those cells become the raw material for synthetic training data.

### The Pipeline

```
Rust crate (e.g., conservation-law)
    → openmind decomposes into cells
    → Each cell has a CAPABILITY.toml
    → synthetic_from_capability.py reads them
    → Generates Q&A conversations about the crate's domain
    → Outputs JSONL for nanochat's CustomJSON task
    → nanochat trains a domain specialist model
```

### Example: From CAPABILITY.toml to Training Data

Given `crates/conservation-law/CAPABILITY.toml`:
```toml
[crate]
name = "conservation-law"
description = "Implements γ+H=C conservation: productive + wasted = total compute"
layer = "core"
```

`synthetic_from_capability.py` produces:
```jsonl
{"messages": [{"role": "user", "content": "What does the conservation-law crate do?"}, {"role": "assistant", "content": "The conservation-law crate implements the γ+H=C conservation principle..."}]}
{"messages": [{"role": "user", "content": "How do you calculate wasted compute (H)?"}, {"role": "assistant", "content": "Wasted compute H is calculated as H = C - γ, where C is total budget..."}]}
```

### Using the Data

```bash
# Generate synthetic data from all crate CAPABILITY.toml files
python si_integration/synthetic_from_capability.py \
    --crates-dir ../crates \
    --output data/si_synthetic.jsonl \
    --conversations-per-crate 50

# Train with nanochat using the CustomJSON task
python -m scripts.chat_sft \
    --task custom_json \
    --data-path data/si_synthetic.jsonl \
    --depth 12
```

---

## 3. The Application-First Loop

The core insight: **don't train first and hope the model is useful. Build the app first, then train the model to serve it.**

### The Loop

```
1. Agent builds application (using expensive inference API)
   ↓
2. Application works, generates real user interactions
   ↓
3. Capture those interactions + synthetic augmentation
   ↓
4. nanochat trains a local specialist model
   ↓
5. Replace expensive inference with local model
   ↓
6. Local model serves the application → more interactions → better training data
   ↓
   (repeat)
```

### Code Example: Closing the Loop

```python
# Step 1: Application captures real interactions
import json
from pathlib import Path

interactions = []
def on_user_interaction(query, response, metadata):
    """Captured from the live application."""
    interactions.append({
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": response},
        ],
        "metadata": metadata,  # latency, model used, etc.
    })

# Step 2: Periodically export for training
def export_training_data():
    path = Path("data/app_interactions.jsonl")
    with open(path, "a") as f:
        for interaction in interactions:
            f.write(json.dumps(interaction) + "\n")
    interactions.clear()
    return path

# Step 3: autoclaw triggers retraining when enough data accumulates
# autoclaw monitors data/app_interactions.jsonl size
# When threshold met → nanochat fine-tunes → model redeployed

# Step 4: Local model replaces API inference
# nanochat serves the model via chat_web:
#   python -m scripts.chat_web --checkpoint runs/app-specialist/ckpt.pt
# Application points to localhost:8000 instead of api.openai.com
```

### Conservation-Aware Budget

The training loop respects conservation law (γ + H = C):

```python
from si_integration.training_budget import TrainingBudget

budget = TrainingBudget(total_gpu_hours=8.0, gamma_target=0.80)

# Before each training run
run_cost = budget.estimate_run(depth=18, dataset_size=50_000)
if budget.can_afford(run_cost):
    # Train
    budget.record_run(gpu_hours=run_cost, productive_ratio=0.85)
else:
    print(f"Budget exhausted: γ={budget.gamma:.1f}h / C={budget.total:.1f}h")
```

---

## Quick Start: Full Integration

```bash
# 1. Clone and setup
git clone https://github.com/SuperInstance/nanochat.git
cd nanochat && uv sync --extra gpu

# 2. Generate synthetic data from SuperInstance crates
python si_integration/synthetic_from_capability.py \
    --crates-dir /path/to/superinstance/crates \
    --output data/si_synthetic.jsonl

# 3. Train a domain specialist
python -m scripts.base_train --depth 12 --data data/si_synthetic.jsonl
python -m scripts.chat_sft --depth 12 --task custom_json --data-path data/si_synthetic.jsonl

# 4. Serve locally
python -m scripts.chat_web --checkpoint runs/si-specialist/ckpt.pt
```

---

## Crate Integration Reference

| SuperInstance Crate | nanochat Role | Data Flow |
|---|---|---|
| autoclaw | Orchestrator → nanochat | Manifests, configs, schedules |
| openmind | Data source → nanochat | CAPABILITY.toml → synthetic JSONL |
| open_vectors | Storage ← nanochat | Training embeddings, metrics |
| conservation-law | Budget governor | γ + H = C for GPU hour tracking |
