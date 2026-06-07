# AGENTS_AS_APPS.md — nanochat IS the Training Application

## The Thesis

nanochat is not a tool that agents *use*. nanochat is an **application that agents ARE**.

In the SuperInstance ecosystem, the agent doesn't call a training API and wait. The agent **decides what to train**, and nanochat handles the mechanics. autoclaw automates the loop. The result is a local model that replaces expensive inference.

---

## The Loop

```
                    ┌──────────────────────┐
                    │    Agent (autoclaw)    │
                    │  Decides WHAT to train │
                    └───────┬──────────────┘
                            │
                    ┌───────▼──────────────┐
                    │   Application Code    │
                    │  (uses API inference) │
                    └───────┬──────────────┘
                            │ captures interactions
                    ┌───────▼──────────────┐
                    │   Synthetic Data Gen  │
                    │  + Real Interactions   │
                    └───────┬──────────────┘
                            │ JSONL
                    ┌───────▼──────────────┐
                    │      nanochat         │
                    │  train → evaluate     │
                    └───────┬──────────────┘
                            │ trained model
                    ┌───────▼──────────────┐
                    │   Local Inference     │
                    │  replaces API calls   │
                    └───────┬──────────────┘
                            │ better/cheaper
                            └─── back to Application Code
```

This is not theoretical. Every step runs on a single GPU.

---

## Why Application-First?

The traditional ML approach:

```
Collect data → Clean data → Train model → Build app → Hope it works
```

Problems:
- You don't know what the model needs to know until the app exists
- Synthetic data is generic, not tuned to actual usage patterns
- Expensive iteration cycle

The SuperInstance approach:

```
Build app (with API inference) → Capture real usage → Train local model → Replace API → Improve
```

Advantages:
- **Real data first**: The model learns from actual user interactions
- **Cheap iteration**: Single-GPU, nanochat's compute-optimal defaults
- **Conservation-aware**: γ+H=C tracks productive vs. wasted compute
- **Immediate ROI**: Every training run directly serves the application

---

## How It Works in Practice

### Step 1: Agent Builds the Application

The agent (driven by autoclaw) builds an application that uses external inference. This is the "bootstrap" phase — expensive but necessary to learn what the model actually needs.

```python
# The app talks to GPT-4 initially
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": user_query}]
)
# But we capture everything
capture_interaction(user_query, response.choices[0].message.content)
```

### Step 2: Application Works, Interactions Accumulate

The app serves real users. Every interaction is logged. Synthetic data from openmind's cellular decomposition augments the real data.

```bash
# openmind decomposes the codebase into cells
# Each cell → CAPABILITY.toml → synthetic Q&A conversations
python si_integration/synthetic_from_capability.py \
    --crates-dir ./crates \
    --output data/synthetic.jsonl

# Real interactions are also collected
cat data/app_interactions.jsonl data/synthetic.jsonl > data/training.jsonl
```

### Step 3: nanochat Trains the Local Model

When enough data accumulates (or on a schedule), nanochat trains:

```bash
# autoclaw triggers the training run
python -m scripts.base_train \
    --depth 18 \
    --data data/training.jsonl \
    --output runs/app-specialist-v1/

# Fine-tune on the Q&A data
python -m scripts.chat_sft \
    --depth 18 \
    --task custom_json \
    --data-path data/training.jsonl \
    --output runs/app-specialist-v1/
```

### Step 4: Local Model Replaces API Inference

```python
# Before: expensive external API
response = openai.chat.completions.create(model="gpt-4", messages=...)

# After: free local inference
import requests
response = requests.post("http://localhost:8000/chat", json={
    "messages": messages
})
```

### Step 5: Loop

The local model serves the app. Interactions continue to accumulate. Periodically, nanochat retrains with the expanded dataset. Each iteration makes the local model better at exactly what the application needs.

---

## Cost Model

Conservation law applies: **γ + H = C** (productive + wasted = total compute budget)

| Phase | γ (productive) | H (wasted) | C (total) |
|-------|----------------|------------|-----------|
| Bootstrap (API inference) | High (learning) | API cost $ | Variable |
| First training run | 70-80% | 20-30% (overfit) | ~2-4 GPU hours |
| Retraining (more data) | 80-90% | 10-20% | ~1-2 GPU hours |
| Steady state | 90%+ | <10% | <1 GPU hour |

A single H100 hour costs ~$3. Training a domain specialist for your application: **under $10**.

---

## The Agent's Decision Framework

autoclaw decides when to train based on:

1. **Data volume threshold**: Enough interactions accumulated?
2. **Performance gap**: Is the local model worse than the API?
3. **Budget conservation**: Is there GPU budget remaining (γ < C)?
4. **Spectral signal**: Are loss eigenvalues indicating overfitting (H rising)?

When all conditions align → trigger nanochat training → deploy new model → loop.

---

## Summary

| Concept | Implementation |
|---------|---------------|
| Agent decides WHAT | autoclaw scans crate graph, identifies gaps |
| nanochat does the training | Single-GPU, compute-optimal, all stages |
| Data comes from the app | Real interactions + synthetic from openmind |
| Budget is conserved | γ+H=C tracking via training_budget.py |
| Local replaces API | chat_web serves the trained model |
| Loop is automatic | autoclaw monitors, triggers, deploys |

**The application is the training data. The training serves the application. The agent orchestrates. nanochat executes.**
