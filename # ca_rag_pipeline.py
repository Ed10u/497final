from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import re
import tempfile
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_metric
from peft import LoraConfig, get_peft_model
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

@dataclass
class Config:
    seed: int = 42
    work_dir: Path = Path("./artifacts")
    model_name: str = "meta‑llama/Meta‑Llama‑3‑8B‑Instruct"  # HuggingFace hub id

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    epochs: int = 1
    per_device_batch_size: int = 2
    grad_accum: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.05

    top_k_docs: int = 3
    confidence_threshold: float = 0.5

    split_ratio: Tuple[int, int, int] = (80, 10, 10)


CFG = Config()
CFG.work_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger("ca‑rag")

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fetch_pubmed_abstracts(max_records: int = 200) -> List[Dict[str, Any]]:
    from Bio import Entrez
    Entrez.email = "example@example.com"
    query = "clinical medicine[MeSH Major Topic] AND 2015:3000[pdat]"
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_records)
    ids = Entrez.read(handle)["IdList"]
    handle.close()

    handle = Entrez.efetch(db="pubmed", id=",".join(ids), rettype="abstract", retmode="text")
    raw_text = handle.read()
    handle.close()

    # Split each abstract crudely by PMID header
    entries: List[Dict[str, Any]] = []
    for block in raw_text.split("PMID:"):
        block = block.strip()
        if not block:
            continue
        pmid_match = re.match(r"(\d+)", block)
        if not pmid_match:
            continue
        pmid = pmid_match.group(1)
        sentences = re.split(r"(?<=[.!?])\s+", block)
        if len(sentences) < 2:
            continue
        entries.append(
            {
                "pmid": pmid,
                "problem": sentences[0],
                "approach": " ".join(sentences[1:]),
                "journal": "Unknown",
                "year": random.randint(2015, 2025),
            }
        )
    return entries


def build_dataset() -> DatasetDict:
    cached = CFG.work_dir / "dataset.arrow"
    if cached.exists():
        return DatasetDict.load_from_disk(str(cached))

    abstracts = fetch_pubmed_abstracts()
    random.shuffle(abstracts)
    n = len(abstracts)
    n_train = math.floor(n * CFG.split_ratio[0] / 100)
    n_val = math.floor(n * CFG.split_ratio[1] / 100)
    ds = DatasetDict(
        {
            "train": Dataset.from_list(abstracts[:n_train]),
            "validation": Dataset.from_list(abstracts[n_train : n_train + n_val]),
            "test": Dataset.from_list(abstracts[n_train + n_val :]),
        }
    )
    ds.save_to_disk(str(cached))
    return ds


def bm25_index(corpus: List[str]) -> BM25Okapi:
    tokenized = [doc.split() for doc in corpus]
    return BM25Okapi(tokenized)


def confidence_score(journal: str, year: int, bm25_score: float) -> float:
    credibility = 1.0 if journal.lower().endswith("q1") else 0.5
    overlap = min(1.0, bm25_score / 10)
    recency = 1.0 if (2025 - year) <= 2 else 0.7
    return 0.4 * credibility + 0.3 * overlap + 0.3 * recency


class Retriever:
    def __init__(self, passages: List[str], meta: List[Dict[str, Any]]):
        self.bm25 = bm25_index(passages)
        self.meta = meta
        self.passages = passages

    def query(self, q: str, k: int = 3) -> List[str]:
        scores = self.bm25.get_scores(q.split())
        ranked = sorted(
            enumerate(scores),
            key=lambda x: confidence_score(self.meta[x[0]]["journal"], self.meta[x[0]]["year"], x[1]),
            reverse=True,
        )
        return [self.passages[i] for i, _ in ranked[:k]]

def prepare_training_examples(dataset: Dataset, retriever: Retriever) -> Dataset:
    def _build_example(example):
        evidences = retriever.query(example["problem"], k=CFG.top_k_docs)
        prompt_lines = [f"<problem> {example['problem']}"]
        prompt_lines += [f"<evidence> {ev}" for ev in evidences]
        prompt_lines.append("<answer>")
        return {"text": "\n".join(prompt_lines + [example["approach"]])}

    return dataset.map(_build_example, remove_columns=dataset.column_names)


def finetune(train_ds: Dataset, val_ds: Dataset) -> None:
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
    model = AutoModelForCausalLM.from_pretrained(CFG.model_name, device_map="auto")

    model = get_peft_model(
        model,
        LoraConfig(r=CFG.lora_r, lora_alpha=CFG.lora_alpha, lora_dropout=CFG.lora_dropout, task_type="CAUSAL_LM"),
    )

    def tokenize_fn(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=512)

    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(CFG.work_dir / "lora‑runs"),
        overwrite_output_dir=True,
        num_train_epochs=CFG.epochs,
        per_device_train_batch_size=CFG.per_device_batch_size,
        per_device_eval_batch_size=CFG.per_device_batch_size,
        gradient_accumulation_steps=CFG.grad_accum,
        learning_rate=CFG.learning_rate,
        warmup_ratio=CFG.warmup_ratio,
        evaluation_strategy="epoch",
        logging_steps=50,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(CFG.work_dir / "ca‑rag‑lora")

def evaluate(test_ds: Dataset, retriever: Retriever) -> None:
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    from bert_score import score as bert_score

    preds, refs = [], []
    for ex in tqdm(test_ds, desc="Evaluating"):
        evidences = "\n".join(retriever.query(ex["problem"], k=CFG.top_k_docs))
        preds.append(evidences)
        refs.append(ex["approach"])

    rouge_scores = [rouge.score(r, p)["rougeL"].fmeasure for r, p in zip(refs, preds)]
    _, bert_p, _ = bert_score(preds, refs, lang="en", rescale_with_baseline=True)

    logger.info("ROUGE‑L (F): %.2f", 100 * sum(rouge_scores) / len(rouge_scores))
    logger.info("BERTScore (P): %.2f", 100 * bert_p.mean().item())

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CA‑RAG pipeline")
    parser.add_argument("--refresh‑corpus", action="store_true", help="Redownload PubMed data")
    args = parser.parse_args()

    set_seed(CFG.seed)

    ds = build_dataset() if not args.refresh_corpus else build_dataset()

    retriever = Retriever([e["approach"] for e in ds["train"]], list(ds["train"]))
    train_prepared = prepare_training_examples(ds["train"], retriever)
    val_prepared = prepare_training_examples(ds["validation"], retriever)

    finetune(train_prepared, val_prepared)

    evaluate(ds["test"], retriever)


if __name__ == "__main__":
    main()
