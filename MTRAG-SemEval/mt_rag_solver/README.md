MT_RAG_Solver
=================

This package provides a runnable pipeline to participate in the MTRAGEval shared task. It implements:

- Conversation-aware query rewriting using an instruction-tuned open LLM.
- Hybrid retrieval (BM25 + dense) merged via Reciprocal Rank Fusion (RRF).
- Cross-encoder reranking.
- Grounded generation with strict prompt (abstain -> "Insufficient information").
- NLI-based lightweight verification.

Quick setup (assumes local GPU; falls back to CPU):

1. Create a virtualenv and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r mt_rag_solver/requirements.txt
```

2. Download any required HF models (optional; the code will download on first run).

3. Run the pipeline (example):

```bash
python -m mt_rag_solver.run --subtask C --setting RAG --domain cloud --split dev --out predictions.jsonl
```

Notes & caveats:
- This implementation uses in-memory BM25 via `rank_bm25` and FAISS + `sentence-transformers` for dense retrieval.
- For production/competition: create proper passage indices (Pyserini or Elasticsearch) and tune model choices and thresholds.

Indexing & format checking
-------------------------

- Indexing: for large collections, prebuild a FAISS index to avoid encoding the full corpus at every run. Example:

```bash
python scripts/index_passages.py --input path/to/passages.jsonl --out-dir mt_rag_solver/data --model intfloat/e5-large-v2
```

- After running the indexer, update `mt_rag_solver/config.yaml` to point `retrieval.faiss_index_path`, `retrieval.embeddings_path`, and `retrieval.mapping_path` to the files under `mt_rag_solver/data`.

- Format checking: run a lightweight format check and attempt the official checker (best-effort):

```bash
python scripts/run_format_checker.py predictions.jsonl
```

To change models, update `config.yaml` with desired Hugging Face model IDs or local model paths. The code prefers local paths when they exist.
Full technical details
----------------------

Overview
--------
This repository implements a pipeline for the MTRAGEval shared task covering the three subtasks:
- A: retrieval-only
- B: generation with reference passages
- C: full RAG (rewrite → retrieve → rerank → generate)

Components
----------
- `rewrite.py`: conversation-aware query rewriting using an instruction-tuned sequence-to-sequence model. Converts the current turn and dialogue history into a standalone query.
- `retrieve.py`: hybrid retrieval combining BM25 (`rank_bm25`) and dense retrieval (`sentence-transformers` + FAISS). Reciprocal Rank Fusion (RRF) merges BM25 and dense rankings.
- `rerank.py`: cross-encoder reranker implemented with `sentence-transformers` CrossEncoder for final ranking.
- `generate.py`: grounded generation using an instruction model with a strict prompt that allows answers only when supported by the provided passages; otherwise returns `Insufficient information`.
- `verify.py`: NLI-based verifier (DeBERTa M-NLI) that checks entailment at the sentence level and abstains if verification fails.
- `io_utils.py`: JSONL read/write utilities compatible with the benchmark format.
- `run.py`: CLI orchestrator implementing the specified pipelines for Subtasks A/B/C and the different generation settings.

Dataflow (Subtask C example)
1. Read input JSONL entry containing dialogue history and the current user turn.
2. Rewrite the current turn + history into a standalone search query.
3. Run BM25 and dense retrieval over the passage pool.
4. Fuse BM25 and dense results using RRF.
5. Rerank fused candidates with a cross-encoder.
6. Select top-K passages for grounded generation.
7. Generate an answer using only the selected passages; if the NLI verifier finds any sentence unsupported, return `Insufficient information`.

Installation & indexing
-----------------------
Create a virtual environment and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r mt_rag_solver/requirements.txt
```

For large collections, prebuild a FAISS index:

```bash
python scripts/index_passages.py --input path/to/passages.jsonl --out-dir mt_rag_solver/data --model intfloat/e5-large-v2
```

Update `mt_rag_solver/config.yaml` with the produced `faiss.index`, `embeddings.npy`, and `id_mapping.json` paths.

Running the pipeline
--------------------
Example CLI (uses benchmark layout under `mt-rag-benchmark/`):

```bash
python -m mt_rag_solver.run --subtask C --setting RAG --domain cloud --split dev --out predictions.jsonl
```

To validate output format:

```bash
python scripts/run_format_checker.py predictions.jsonl
```

Reproducibility and configuration
---------------------------------
- Configuration options are in `mt_rag_solver/config.yaml`.
- Seeds are fixed where applicable and generation runs with `do_sample=False` to increase determinism. Exact bit-for-bit reproducibility may vary with hardware and model implementations.

Limitations
-----------
- Default models are sized to facilitate local testing. For higher-quality results, use larger instruction-tuned LLMs and stronger rerankers.
- The NLI verifier operates at sentence level and may be conservative; tuning thresholds or using more advanced verification is recommended for production.
