# RFP Advanced RAG

**English** | [한국어](README.ko.md)

> A question-answering system over **100 Korean public-sector RFP documents (~7,500 PDF pages)**, with text, table, and image parsing and a hybrid Dense + BM25 retriever.
>
> **Status:** a post-project audit found that the reported "Advanced RAG" evaluation ran on the *Naive* index, so the table and image parsing were never measured, and the grader was too lenient. The numbers below are reported as they are, with the audit next to them. A clean ablation is in progress (see [Evaluation audit](#evaluation-audit)).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## Results (v1, as originally reported)

Benchmark: 30 hand-written questions, each with a gold answer and a source page (team-labeled as 9 text, 12 table, and 9 image questions). Scores come from [`evaluation/evaluate_final.py`](evaluation/evaluate_final.py), where `gpt-5-mini` marks each answer PASS or FAIL.

| System as actually evaluated | Index | Retriever | Correct (out of 30) |
|---|---|---|:---:|
| Naive RAG | text-only pages | Dense, k = 4 | 21 |
| "Advanced RAG" | **same text-only index** | Dense k = 12 + BM25 k = 12 | 25 |

Per-question logs: [`result_1.txt`](evaluation/results/result_1.txt) (Naive), [`result_2.txt`](evaluation/results/result_2.txt) (Advanced).

The only real difference between the two runs is **retrieval breadth** (4 chunks vs. up to 24, plus BM25). Of the 4 changed questions, 2 are clear wins that fit this explanation (Q15 and Q21: "not found" became correct). Q29 lost a hallucinated detail, and Q28 was a grader false positive.

The 5 questions both systems miss (Q08, Q12, Q14, Q24, Q26) all have their answers **inside tables or forms**, which is exactly what the unevaluated table parser was built for.

## Evaluation audit

I re-read the notebooks, the evaluation code, and every scored answer after the project ended.

**What was evaluated** ([`3. AdvancedRAG.ipynb`](notebooks/RAGsystem/))
- **The "Advanced" run reused the Naive index.** Cell 23 loads `faiss_openai` + `split_documents.pkl`, which the Naive notebook saved. The cells that merge tables and images and build a new index have no execution count, so they never ran in that session.
- **Tables and images never reached the index.** The 7,547 documents equal 7,576 PDF pages minus 29 empty ones, which means text pages only. The 12,971 parsed tables were not included.
- The full text + table + image pipeline exists in [`src/pipeline.py`](src/pipeline.py) (used by the Streamlit app), but it was never benchmarked.

**Answer grading**
- **Lenient rubric.** The prompt passes answers that cover "half or more" of the gold answer. Q28 passed without the key fact (tablet PC), and Q03 passed with 400W where the gold answer is 100W, even though the rubric says numeric mismatches must fail.
- **Same model generates and grades** (`gpt-5-mini`), so self-preference bias is possible.
- **Single, non-deterministic run.** The generator uses `temperature=1` and the eval ran once. A 4-question difference on n=30 is not statistically significant (McNemar exact p ≈ 0.125).
- **Unused judge.** `LLM_as_a_judge.py` (`gpt-4o-mini`) was passed the *retrieved context* instead of the gold answer, and it was not used for the reported numbers.

**Retrieval metrics** ([`exp_parameters.ipynb`](notebooks/experiments/exp_parameters.ipynb))
- **6 of 30 questions have no gold page** (`page: null`), which caps Hit@k at 0.80.
- **Exact-page matching** counts facts that also appear on other pages as misses. That explains a Hit@10 of about 0.27 alongside 70–83% answer accuracy.
- **"Best k = 12" is an artifact.** The ensemble returns up to 2k chunks and MRR has no rank cutoff, so MRR rises with k automatically. Hit@1/3/5/10 are identical for k = 8, 10 and 12.
- **No held-out set.** The same 30 questions were used both to tune and to report.

**v2 plan: a clean ablation**

| Step | Change |
|---|---|
| ① | Naive (text, dense) |
| ② | + hybrid retrieval |
| ③ | + table parsing |
| ④ | + image parsing |

Each step is scored per question type (text / table / image), first by retrieval hit rate (no LLM needed) and then by a strict gold-based judge from a different model family, averaged over several runs. The 6 missing gold pages will be completed and MRR capped at a fixed cutoff.

---

## Problem

RFP (Request for Proposal) documents are long, inconsistently formatted PDFs. Key facts such as budgets, schedules, and hardware specs often live in **tables and images**, not plain text, so a naive text-only RAG misses or garbles them.

## Architecture

The full pipeline implemented in `src/`. Steps marked † were not part of the v1 benchmark.

```
PDF (100 files)
 ├─ Text   → PyMuPDF
 ├─ Tables → pdfplumber → Markdown        †
 └─ Images → GPT Vision → text summary    †
        │
        ▼
 Merge + preprocess → chunk (700 / 70) → text-embedding-3-small → FAISS
        │
        ▼
 Hybrid Retriever  (Dense 0.7 + BM25 0.3, k = 12)
        │
        ▼
 gpt-5-mini + custom prompt (LangChain LCEL) → answer with source metadata
        │
        ▼
 Streamlit UI  ·  LLM-as-a-Judge evaluation
```

| Component | Choice |
|---|---|
| Embedding | `text-embedding-3-small` |
| Vector store | FAISS |
| Retriever | `EnsembleRetriever` (Dense 70% + BM25 30%), k = 12 |
| Generator | `gpt-5-mini` via LangChain LCEL |
| Grader (v1) | `gpt-5-mini` PASS/FAIL |

---

## My Contributions

This was a 5-person team project (Feb 2026). I wrote about 79% of the non-notebook code in the final repository, including:

- **Retriever experiments**: compared naive, dense, hybrid, and reranker retrievers ([`retrievers/`](retrievers/)) and chose the final hybrid setup
- **Advanced RAG implementation**: [`notebooks/RAGsystem/3. AdvancedRAG.ipynb`](notebooks/RAGsystem/)
- **Parsers**: text, advanced text, and GPT-Vision image parsing ([`parsers/`](parsers/))
- **Production pipeline**: [`src/pipeline.py`](src/pipeline.py), [`loader.py`](src/loader.py), [`generator.py`](src/generator.py) (prompt design)
- **Evaluation**: chunking grid search, the Naive vs Advanced benchmark runs, and the post-project [evaluation audit](#evaluation-audit) that found the issues above
- **Streamlit UI**: [`src/streamlit.py`](src/streamlit.py)

---

## Quick Start

```bash
conda create -p .conda python=3.10 -y
conda activate ./.conda
pip install -r requirements.txt

# Add your key to a .env file in the project root
echo "OPENAI_API_KEY=sk-..." > .env

streamlit run src/streamlit.py
```

If `data/vectorstore/faiss_advanced/` does not exist, it is built on the first run, which takes a while.

Programmatic use:

```python
from src.pipeline import build_pipeline
from src.generator import ask

chain = build_pipeline()   # loads the FAISS index, or builds it if missing
print(ask(chain, "What is the total budget of the Bonghwa-gun disaster system project?"))
```

## Project Structure

```
├── parsers/       # text / table / image parsers
├── src/           # production pipeline (loader → preprocessor → embedding → retriever → generator) + Streamlit
├── retrievers/    # retriever variants used in experiments
├── evaluation/    # test set (30 Q&A), LLM judge, results
├── notebooks/     # EDA, experiments (parsers, chunking, retrievers, prompts), Naive/Advanced RAG
└── config.yaml
```

## Limitations and Next Steps

- The benchmark has only 30 questions, so one question equals 3.3 percentage points. See [Evaluation audit](#evaluation-audit) for the v1 measurement issues and the v2 plan.
- Once retrieval metrics are fixed, a reranker and query rewriting are the next levers to test.
- The documents and questions are in Korean. The pipeline itself is language-agnostic.

## Team

[juyoung-song](https://github.com/juyoung-song) · [choimuyeong](https://github.com/choimuyeong) · [Yumin-Hwang046](https://github.com/Yumin-Hwang046) · [rnrudwns123-design](https://github.com/rnrudwns123-design) · [sfun1993-bit](https://github.com/sfun1993-bit)

Final report (Korean): [Team2_RAG_Report.pdf](https://github.com/user-attachments/files/25600246/Team2_RAG_.pdf)
