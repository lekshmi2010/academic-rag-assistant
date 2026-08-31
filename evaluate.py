import os
import sys
import time
import warnings
import pandas as pd
import numpy as np

# Ensure root workspace is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Ensure standard stdout encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Suppress minor non-critical third-party deprecation notices during eval
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import config
from database import get_vector_store
from services.chat import ask_lecture_question
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from datasets import Dataset
from ragas import evaluate, EvaluationDataset
from ragas.run_config import RunConfig
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


def get_ragas_evaluators():
    """Initializes the LLM and Embeddings wrappers for Ragas evaluation using Gemini."""
    eval_llm = ChatGoogleGenerativeAI(
        model=config.CHAT_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
        temperature=0.0,
        request_timeout=120,
    )
    eval_embeddings = GoogleGenerativeAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        google_api_key=config.GOOGLE_API_KEY,
    )
    return LangchainLLMWrapper(eval_llm), LangchainEmbeddingsWrapper(eval_embeddings)


def run_ragas_evaluation(
    video_id: str,
    test_cases: list[dict],
    top_k: int = 5,
    output_csv: str = "ragas_benchmark_results.csv",
    output_summary_md: str = "ragas_benchmark_summary.md",
):
    """
    Executes an end-to-end Ragas evaluation for a given ingested lecture video.

    Measures:
      - Faithfulness (Hallucination Detection)
      - Answer Relevancy (Question-Response Pertinence)
      - Context Precision (Retrieval Ranking Quality)
      - Context Recall (Ground Truth Coverage)
      - Vector Retrieval Latency (ms)
      - End-to-End Response Time (s)
    """
    print("\n" + "=" * 80)
    print(f"[+] INITIATING RAGAS BENCHMARK SUITE (Video ID: {video_id})")
    print(f"[+] Evaluator LLM : {config.CHAT_MODEL}")
    print(f"[+] Embedding Model: {config.EMBEDDING_MODEL}")
    print(f"[+] Test Set Size : {len(test_cases)} query pairs")
    print("=" * 80)

    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k, "filter": {"video_id": video_id}},
    )

    questions = []
    generated_answers = []
    retrieved_contexts = []
    ground_truths = []
    retrieval_latencies_ms = []
    total_latencies_sec = []

    print("\n[Step 1/3] Querying RAG Pipeline & Capturing Retrieved Contexts...")
    for idx, item in enumerate(test_cases, 1):
        q = item["question"]
        ref = item["ground_truth"]
        session_id = f"eval-session-{idx}-{int(time.time())}"

        print(f"\n  [{idx}/{len(test_cases)}] Question: \"{q}\"")

        # 1. Measure Vector Retrieval Latency & extract context chunks
        t_ret_start = time.perf_counter()
        docs = retriever.invoke(q)
        t_ret_end = time.perf_counter()
        ret_latency_ms = (t_ret_end - t_ret_start) * 1000
        retrieval_latencies_ms.append(ret_latency_ms)

        chunk_texts = [d.page_content for d in docs if d.page_content]
        retrieved_contexts.append(chunk_texts)

        # 2. Measure End-to-End Query Generation Time with retry pacing
        t_gen_start = time.perf_counter()
        answer = ask_lecture_question(video_id, q, session_id)
        t_gen_end = time.perf_counter()
        tot_latency_sec = t_gen_end - t_gen_start
        total_latencies_sec.append(tot_latency_sec)

        questions.append(q)
        generated_answers.append(answer)
        ground_truths.append(ref)

        print(f"      [+] Retrieval Latency : {ret_latency_ms:.1f} ms")
        print(f"      [+] End-to-End Time   : {tot_latency_sec:.2f} s")
        print(f"      [+] Retrieved Chunks  : {len(chunk_texts)}")

        # Pacing delay between queries
        if idx < len(test_cases):
            time.sleep(2.0)

    # Construct Ragas Evaluation Dataset
    eval_dict = {
        "user_input": questions,
        "response": generated_answers,
        "retrieved_contexts": retrieved_contexts,
        "reference": ground_truths,
    }
    hf_dataset = Dataset.from_dict(eval_dict)
    eval_dataset = EvaluationDataset.from_hf_dataset(hf_dataset)

    # Run Ragas Evaluation with rate-limit friendly RunConfig
    print("\n[Step 2/3] Computing RAG Triad Metrics via Ragas...")
    ragas_llm, ragas_embeddings = get_ragas_evaluators()
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    eval_run_config = RunConfig(
        timeout=180,
        max_retries=10,
        max_wait=60,
        max_workers=1,
    )

    eval_result = evaluate(
        dataset=eval_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=eval_run_config,
    )

    # Convert results into structured DataFrame
    df = eval_result.to_pandas()
    df["retrieval_latency_ms"] = retrieval_latencies_ms
    df["total_latency_sec"] = total_latencies_sec

    # Save detailed CSV
    df.to_csv(output_csv, index=False)
    print(f"\n[Step 3/3] Exported detailed evaluation metrics to '{output_csv}'.")

    # Aggregate metric summary
    avg_faithfulness = df["faithfulness"].mean() if "faithfulness" in df else np.nan
    avg_relevancy = df["answer_relevancy"].mean() if "answer_relevancy" in df else np.nan
    avg_precision = df["context_precision"].mean() if "context_precision" in df else np.nan
    avg_recall = df["context_recall"].mean() if "context_recall" in df else np.nan
    avg_ret_ms = np.mean(retrieval_latencies_ms)
    avg_tot_sec = np.mean(total_latencies_sec)

    # Print Final Terminal Report
    print("\n" + "=" * 80)
    print(" FINAL RAGAS BENCHMARK SUMMARY")
    print("=" * 80)
    print(f" {'#':<3} | {'Question Snippet':<32} | {'Faithful':<9} | {'Relevance':<9} | {'Precision':<9} | {'Recall':<7}")
    print("-" * 80)
    for i, row in df.iterrows():
        q_snip = (row["user_input"][:29] + "...") if len(row["user_input"]) > 32 else row["user_input"]
        f_val = f"{row.get('faithfulness', 0):.2f}" if pd.notna(row.get('faithfulness')) else "N/A"
        ar_val = f"{row.get('answer_relevancy', 0):.2f}" if pd.notna(row.get('answer_relevancy')) else "N/A"
        cp_val = f"{row.get('context_precision', 0):.2f}" if pd.notna(row.get('context_precision')) else "N/A"
        cr_val = f"{row.get('context_recall', 0):.2f}" if pd.notna(row.get('context_recall')) else "N/A"
        print(f" {i+1:<3} | {q_snip:<32} | {f_val:<9} | {ar_val:<9} | {cp_val:<9} | {cr_val:<7}")

    print("-" * 80)
    print(f" [*] AVERAGE FAITHFULNESS     : {avg_faithfulness:.4f} ({avg_faithfulness*100:.1f}%)")
    print(f" [*] AVERAGE ANSWER RELEVANCY : {avg_relevancy:.4f} ({avg_relevancy*100:.1f}%)")
    print(f" [*] AVERAGE CONTEXT PRECISION: {avg_precision:.4f} ({avg_precision*100:.1f}%)")
    print(f" [*] AVERAGE CONTEXT RECALL   : {avg_recall:.4f} ({avg_recall*100:.1f}%)")
    print(f" [*] AVG RETRIEVAL LATENCY    : {avg_ret_ms:.1f} ms")
    print(f" [*] AVG END-TO-END LATENCY   : {avg_tot_sec:.2f} seconds")
    print("=" * 80)

    # Write Markdown summary for portfolio/resume
    summary_md = f"""# RAG Pipeline Evaluation Report (Ragas)

**Video ID**: `{video_id}`  
**Evaluator Model**: `{config.CHAT_MODEL}`  
**Embeddings**: `{config.EMBEDDING_MODEL}`  
**Vector Store**: ChromaDB (Cosine / IP)

---

## Benchmark Scorecard

| Metric | Score | Target / Benchmark | Assessment |
| :--- | :---: | :---: | :--- |
| **Faithfulness** | **{avg_faithfulness:.4f}** | $\\ge 0.85$ | Evaluates factual grounding against retrieved lecture transcripts (Hallucination Detection). |
| **Answer Relevancy** | **{avg_relevancy:.4f}** | $\\ge 0.85$ | Evaluates semantic relevance to the student's question. |
| **Context Precision** | **{avg_precision:.4f}** | $\\ge 0.80$ | Evaluates whether ground-truth relevant chunks rank at the top of retrieved context. |
| **Context Recall** | **{avg_recall:.4f}** | $\\ge 0.80$ | Evaluates whether retrieved chunks cover all ground truth concepts. |
| **Vector Retrieval Latency** | **{avg_ret_ms:.1f} ms** | $< 100\\text{{ ms}}$ | ChromaDB top-5 similarity search speed. |
| **End-to-End Response Time** | **{avg_tot_sec:.2f} s** | $< 4.0\\text{{ s}}$ | Total roundtrip time from question to reasoning + answer generation. |

---

## Detailed Per-Query Results

| # | Question | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Total Latency |
| :-: | :--- | :-: | :-: | :-: | :-: | :-: |
"""
    for i, row in df.iterrows():
        f_val = f"{row.get('faithfulness', 0):.4f}" if pd.notna(row.get('faithfulness')) else "N/A"
        ar_val = f"{row.get('answer_relevancy', 0):.4f}" if pd.notna(row.get('answer_relevancy')) else "N/A"
        cp_val = f"{row.get('context_precision', 0):.4f}" if pd.notna(row.get('context_precision')) else "N/A"
        cr_val = f"{row.get('context_recall', 0):.4f}" if pd.notna(row.get('context_recall')) else "N/A"
        lat = f"{row.get('total_latency_sec', 0):.2f}s"
        summary_md += f"| {i+1} | {row['user_input']} | {f_val} | {ar_val} | {cp_val} | {cr_val} | {lat} |\n"

    summary_md += """
---

## Resume Highlights

> - **RAG Pipeline Optimization & Ragas Benchmarking**: Implemented end-to-end evaluation using **Ragas** on multi-modal lecture transcripts, achieving **{faithfulness_pct}% Faithfulness**, **{relevancy_pct}% Answer Relevancy**, and **{precision_pct}% Context Precision**.
> - **High-Performance Vector Retrieval**: Integrated ChromaDB with Gemini text embeddings, delivering **< {ret_ms_round}ms** vector retrieval and sub-{tot_sec_round}s end-to-end question answering.
""".format(
        faithfulness_pct=f"{avg_faithfulness*100:.1f}" if pd.notna(avg_faithfulness) else "N/A",
        relevancy_pct=f"{avg_relevancy*100:.1f}" if pd.notna(avg_relevancy) else "N/A",
        precision_pct=f"{avg_precision*100:.1f}" if pd.notna(avg_precision) else "N/A",
        ret_ms_round=int(np.ceil(avg_ret_ms / 10.0) * 10),
        tot_sec_round=int(np.ceil(avg_tot_sec)),
    )

    with open(output_summary_md, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"[+] Exported benchmark scorecard summary to '{output_summary_md}'.\n")

    return df


if __name__ == "__main__":
    # Test suite for 3Blue1Brown Neural Networks lecture (video_id: aircAruvnKk)
    benchmark_video_id = "aircAruvnKk"
    benchmark_test_cases = [
        {
            "question": "What is a neuron and what does its activation represent in the network?",
            "ground_truth": "A neuron is a basic unit or node that holds a number called an activation between 0 and 1, representing how brightly lit a particular pixel or subcomponent is."
        },
        {
            "question": "How are the 28 by 28 pixel handwritten digit images represented in the input layer?",
            "ground_truth": "The 28x28 pixel image contains 784 pixels in total, each corresponding to one of the 784 neurons in the input layer with an activation value between 0 (black) and 1 (white)."
        },
        {
            "question": "Why does the instructor describe the entire neural network as a mathematical function?",
            "ground_truth": "The network takes 784 numbers as input and outputs 10 numbers representing the probabilities of digits 0 through 9, operating as a function parameterized by weights and biases."
        },
        {
            "question": "What role do the hidden layers play in breaking down complex patterns?",
            "ground_truth": "The hidden layers build up higher-level representations hierarchically, where early layers detect small edges and components, and later layers combine them into loops and digits."
        }
    ]

    run_ragas_evaluation(benchmark_video_id, benchmark_test_cases)
