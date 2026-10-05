# RFP Advanced RAG

**English** | [한국어](README.ko.md)

> A question-answering system over **100 Korean public-sector RFP documents (~7,500 PDF pages)**.
> It parses text, tables, and images, retrieves with a hybrid Dense + BM25 retriever, and is evaluated with an LLM-as-a-Judge benchmark.
> Advanced RAG fixed questions that Naive RAG could not answer (21 → 25 of 30 under the v1 grader). An audit of that grader found it too lenient, and a stricter v2 evaluation is in progress (see [Evaluation audit](#evaluation-audit)).

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## Results

Benchmark: 30 hand-written questions, each with a gold answer and a source page. v1 scores come from [`evaluation/evaluate_final.py`](evaluation/evaluate_final.py), where `gpt-5-mini` marks each answer PASS or FAIL against the gold answer.

| System | Correct (v1 grader, out of 30) |
|---|:---:|
| Naive RAG (text only, dense retrieval) | 21 |
| Advanced RAG (improved text parsing + hybrid Dense/BM25 retrieval) | 25 |

Per-question logs: [`result_1.txt`](evaluation/results/result_1.txt) (Naive), [`result_2.txt`](evaluation/results/result_2.txt) (Advanced).

All 4 changed questions moved from wrong to right, and none regressed. On review, though, only 2 of the 4 are clear wins (Q15 and Q21: "not found" became the correct answer). Q29 lost a hallucinated detail, and Q28 was a grader false positive.

## Evaluation audit

I re-read every scored answer and the evaluation code after the project ended. Issues found in v1:

**Answer grading**
- **Lenient rubric.** The prompt passes answers that cover "half or more" of the gold answer. Q28 passed without mentioning the key fact (tablet PC), and Q03 passed with 400W where the gold answer is 100W, even though the rubric says numeric mismatches must fail.
- **Same model generates and grades** (`gpt-5-mini`), so self-preference bias is possible.
- **Single, non-deterministic run.** The generator uses `temperature=1` and the eval ran once. A 4-question difference on n=30 is not statistically significant (McNemar exact p ≈ 0.125).
- **Unused judge.** `LLM_as_a_judge.py` (`gpt-4o-mini`) was passed the *retrieved context* instead of the gold answer, so it measured consistency with the context rather than correctness. It was not used for the reported numbers.

**Retrieval metrics** ([`exp_parameters.ipynb`](notebooks/experiments/exp_parameters.ipynb))
- **6 of 30 questions have no gold page** (`page: null`). They can never count as a hit, which caps Hit@k at 0.80.
- **Exact-page matching.** Facts that also appear on summary pages count as misses, so the Hit@10 of about 0.27 says more about labels than about retrieval, given that 70–83% of answers were correct.
- **"Best k = 12" is an artifact.** The ensemble returns up to 2k chunks and MRR has no rank cutoff, so MRR rises with k automatically. Hit@1/3/5/10 are identical for k = 8, 10 and 12.
- **No held-out set.** The same 30 questions were used both to tune and to report.

**v2 plan:** a strict gold-based judge from a different model family that outputs a reason with its score, mean ± std over several runs, completed page labels with file-level and page-level hit rates, MRR@10 at a fixed k, and a tuning/test split.

---

## Problem

RFP (Request for Proposal) documents are long, inconsistently formatted PDFs. Key facts such as budgets, schedules, and hardware specs often live in **tables and images**, not plain text, so a naive text-only RAG misses or garbles them.

## Architecture

```
PDF (100 files)
 ├─ Text   → PyMuPDF
 ├─ Tables → pdfplumber → Markdown
 └─ Images → GPT Vision → text summary
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
- **Evaluation**: chunking grid search, the Naive vs Advanced benchmark runs, and the post-project [evaluation audit](#evaluation-audit)
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
