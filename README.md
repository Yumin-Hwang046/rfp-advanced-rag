# RFP Advanced RAG

**English** | [한국어](README.ko.md)

> A question-answering system over **100 Korean public-sector RFP documents (~7,500 PDF pages)**.
> It parses text, tables, and images, retrieves with a hybrid Dense + BM25 retriever, and is evaluated with an LLM-as-a-Judge benchmark.
> Moving from Naive RAG to Advanced RAG raised answer accuracy from **70% to 83%**.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=chainlink&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-0467DF?style=flat-square&logo=meta&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)

---

## Results

Benchmark: 30 hand-written questions with gold answers. Each answer is graded correct or incorrect by `gpt-4o-mini` acting as a judge ([`evaluation/LLM_as_a_judge.py`](evaluation/LLM_as_a_judge.py)).

| System | Correct (out of 30) | Accuracy |
|---|:---:|:---:|
| Naive RAG (text only, dense retrieval) | 21 | 70% |
| **Advanced RAG** (improved text parsing + hybrid Dense/BM25 retrieval) | **25** | **83%** |

Full per-question logs: [`result_1.txt`](evaluation/results/result_1.txt) (Naive), [`result_2.txt`](evaluation/results/result_2.txt) (Advanced).

### Chunking grid search (retrieval quality)

10 configurations were compared in two stages ([`gridsearch_stage2.csv`](evaluation/results/gridsearch_stage2.csv)). The best one is used in production:

| chunk_size | overlap | k | MRR | Hit@10 |
|:---:|:---:|:---:|:---:|:---:|
| **700** | **70** | **12** | **0.134** | 0.267 |
| 700 | 70 | 10 | 0.132 | 0.267 |
| 900 | 90 | 10 | 0.131 | 0.300 |
| 700 | 70 | 4 | 0.122 | 0.200 |

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
| Judge | `gpt-4o-mini` |

---

## My Contributions

This was a 5-person team project (Feb 2026). I wrote about 79% of the non-notebook code in the final repository, including:

- **Retriever experiments**: compared naive, dense, hybrid, and reranker retrievers ([`retrievers/`](retrievers/)) and chose the final hybrid setup
- **Advanced RAG implementation**: [`notebooks/RAGsystem/3. AdvancedRAG.ipynb`](notebooks/RAGsystem/)
- **Parsers**: text, advanced text, and GPT-Vision image parsing ([`parsers/`](parsers/))
- **Production pipeline**: [`src/pipeline.py`](src/pipeline.py), [`loader.py`](src/loader.py), [`generator.py`](src/generator.py) (prompt design)
- **Evaluation**: chunking grid search and the Naive vs Advanced benchmark runs
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

- The benchmark has only 30 questions, so one question equals 3.3 percentage points. A larger test set would give tighter estimates.
- Retrieval MRR (~0.13) is low even though answer accuracy is high. The generator often recovers from imperfect ranking, which makes a reranker or query rewriting the clearest next lever.
- The documents and questions are in Korean. The pipeline itself is language-agnostic.

## Team

[juyoung-song](https://github.com/juyoung-song) · [choimuyeong](https://github.com/choimuyeong) · [Yumin-Hwang046](https://github.com/Yumin-Hwang046) · [rnrudwns123-design](https://github.com/rnrudwns123-design) · [sfun1993-bit](https://github.com/sfun1993-bit)

Final report (Korean): [Team2_RAG_Report.pdf](https://github.com/user-attachments/files/25600246/Team2_RAG_.pdf)
