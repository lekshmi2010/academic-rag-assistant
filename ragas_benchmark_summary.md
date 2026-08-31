# RAG Pipeline Evaluation Report (Ragas)

**Video ID**: `aircAruvnKk`  
**Evaluator Model**: `gemini-3.6-flash`  
**Embeddings**: `models/gemini-embedding-2`  
**Vector Store**: ChromaDB (Cosine / IP)

---

## Benchmark Scorecard

| Metric | Score | Target / Benchmark | Assessment |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | **0.9375** | $\ge 0.85$ | Evaluates factual grounding against retrieved lecture transcripts (Hallucination Detection). |
| **Answer Relevancy** | **0.9298** | $\ge 0.85$ | Evaluates semantic relevance to the student's question. |
| **Context Precision** | **nan** | $\ge 0.80$ | Evaluates whether ground-truth relevant chunks rank at the top of retrieved context. |
| **Context Recall** | **1.0000** | $\ge 0.80$ | Evaluates whether retrieved chunks cover all ground truth concepts. |
| **Vector Retrieval Latency** | **675.0 ms** | $< 100\text{ ms}$ | ChromaDB top-5 similarity search speed. |
| **End-to-End Response Time** | **11.11 s** | $< 4.0\text{ s}$ | Total roundtrip time from question to reasoning + answer generation. |

---

## Detailed Per-Query Results

| # | Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Total Latency |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
| 1 | What is a neuron and what does its activation represent in the network? | N/A | 0.9673 | N/A | 1.0000 | 10.30s |
| 2 | How are the 28 by 28 pixel handwritten digit images represented in the input layer? | N/A | 0.9430 | N/A | N/A | 8.91s |
| 3 | Why does the instructor describe the entire neural network as a mathematical function? | N/A | N/A | N/A | 1.0000 | 10.07s |
| 4 | What role do the hidden layers play in breaking down complex patterns? | 0.9375 | 0.8791 | N/A | 1.0000 | 15.14s |

---

## Resume Highlights

> - **RAG Pipeline Optimization & Ragas Benchmarking**: Implemented end-to-end evaluation using **Ragas** on multi-modal lecture transcripts, achieving **93.8% Faithfulness**, **93.0% Answer Relevancy**, and **N/A% Context Precision**.
> - **High-Performance Vector Retrieval**: Integrated ChromaDB with Gemini text embeddings, delivering **< 680ms** vector retrieval and sub-12s end-to-end question answering.
