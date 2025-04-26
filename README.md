# Confidence‑Aware Retrieval‑Augmented Generation (CA‑RAG)

> **A minimal, fully‑reproducible reference pipeline** that combines confidence‑weighted retrieval with LoRA‑fine‑tuned Llama‑3 8B to reduce hallucinations in scientific question‑answering.

---

## ✨ Key Features

| Stage | What it does | Where to look |
|-------|--------------|---------------|
| **0 · Config & logging** | Centralised hyper‑parameters, reproducible seeding | `Config` dataclass in `ca_rag_pipeline.py` |
| **1 · Dataset builder** | Pulls 200 PubMed abstracts, slices them into *problem / approach* pairs, builds an 80/10/10 split | `fetch_pubmed_abstracts()` & `build_dataset()` |
| **2 · Confidence‑aware retrieval** | BM25 overlap + journal credibility + recency → single score | `confidence_score()` & `Retriever` class |
| **3 · Model fine‑tuning** | LoRA adapters on **Meta‑Llama‑3‑8B‑Instruct** with hyper‑params from Table 2 | `finetune()` |
| **4 · Evaluation** | ROUGE‑L + BERTScore automatic metrics (mirrors the report) | `evaluate()` |

---

## 🛠 Requirements

```bash
python >= 3.10
CUDA‑enabled GPU (≈12 GB VRAM for model + LoRA fine‑tuning)
```

Install Python dependencies:

```bash
pip install transformers accelerate peft datasets rank-bm25 \
            scikit-learn rouge-score bert-score biopython tqdm
```

---

## 🚀 Quick Start

```bash
# 1 ▸ Clone this repo & cd in
python ca_rag_pipeline.py            # one‑shot: build ➜ train ➜ evaluate

# 2 ▸ Need help?
python ca_rag_pipeline.py --help      # list CLI flags
```

Artifacts are written to `./artifacts/` (dataset cache, LoRA checkpoints, metrics logs).

---

## 📂 Project Layout

```
.
├── ca_rag_pipeline.py   # single‑file pipeline (you’re here 📜)
├── README.md            # project overview (this file)
```

---


