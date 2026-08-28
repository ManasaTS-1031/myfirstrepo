# 🩺 Medical Adaptive RAG Chatbot

A medical question-answering chatbot built using **Adaptive RAG**, **ChromaDB**, **Tavily**, **Gemini**, and **Gradio**.

The system first searches the local medical knowledge base. If the retrieved information has low similarity, it automatically falls back to web search using Tavily. The retrieved information is then passed to Gemini to generate a context-grounded answer with sources.

> ⚠️ **Medical Disclaimer:** This project is intended for educational and research purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment.

---

## 📸 Application

<p align="center">
  <img
   src="https://drive.google.com/uc?export=view&id=1ZiSzti3W_lEjD3AuOVBQj07-j8171WlY"
   alt="Medical Adaptive RAG Chatbot"
   width="850"
  >
</p>

---

## 🏗️ Architecture

```text
                         User Question
                              ↓
                           Gradio
                              ↓
                    Adaptive RAG Pipeline
                              ↓
                         ChromaDB
                              ↓
                    Similarity Check
                       /          \
                     High          Low
                      ↓             ↓
                   Local         Tavily
                 Knowledge      Web Search
                      \             /
                       \           /
                        \         /
                          Gemini
                            ↓
                    Answer + Sources
```

---

## 🔄 Adaptive RAG Workflow

The chatbot uses an adaptive retrieval approach instead of relying exclusively on either local documents or web search.

### 1. User Query

The user submits a medical question through the **Gradio** interface.

### 2. Local Knowledge Retrieval

The system first searches the locally stored medical documents using **ChromaDB**.

### 3. Similarity Check

The retrieved documents are evaluated based on their similarity/relevance to the user's query.

### 4. Adaptive Routing

If the retrieved local context is sufficiently relevant, it is used for answer generation.

If the local retrieval quality is low, the system automatically performs a **Tavily web search** to obtain additional relevant information.

### 5. Answer Generation

The retrieved context from the local knowledge base and/or Tavily is passed to **Gemini**.

Gemini generates the final answer using the available context.

### 6. Sources

The system provides the available sources along with the generated response.

---

## ✨ Key Features

* 🩺 Medical question answering
* 🧠 Adaptive Retrieval-Augmented Generation
* 📚 Local medical knowledge base
* 🗄️ ChromaDB vector database
* 🔎 Similarity-based retrieval
* 🌐 Tavily web-search fallback
* 🤖 Gemini-powered response generation
* 📖 Context-grounded answers
* 🔗 Source-aware responses
* 💬 Gradio user interface
* 📊 Ragas-based RAG evaluation
* 📈 Retrieval and answer-quality evaluation

---

## 🧩 Technology Stack

| Technology                | Purpose                                                  |
| ------------------------- | -------------------------------------------------------- |
| **Python**                | Core programming language                                |
| **ChromaDB**              | Vector database for local document retrieval             |
| **Tavily**                | Web-search fallback when local retrieval is insufficient |
| **Gemini**                | Large language model for answer generation               |
| **Gradio**                | Interactive chatbot interface                            |
| **Ragas**                 | RAG evaluation framework                                 |
| **Sentence Transformers** | Evaluation embeddings                                    |

---

# 📊 RAG Evaluation

The Medical Adaptive RAG pipeline was evaluated using **Ragas** on a small, manually curated medical evaluation dataset containing **3 queries** related to the common cold.

The evaluation measures both:

* **Retrieval quality** — whether the relevant information was retrieved from the knowledge base.
* **Answer quality** — whether the generated answer is faithful to and semantically similar to the expected answer.

---

## ⚙️ Evaluation Configuration

| Parameter             | Value                                    |
| --------------------- | ---------------------------------------- |
| Evaluation queries    | **3**                                    |
| Top-K                 | **5**                                    |
| Evaluation framework  | **Ragas**                                |
| Evaluation LLM        | **Gemini**                               |
| Evaluation embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Output file           | `EvaluationScores.csv`                   |

---

# 🧪 Evaluation Queries

The evaluation dataset contains three manually created questions and corresponding ground-truth answers.

---

## Query 1 — Common Cold Definition

### Question

> What is the definition of the common cold and what symptoms characterize it?

### Ground Truth

> The common cold is a mild, self-limiting upper respiratory tract infection characterized by nasal stuffiness and discharge, sneezing, sore throat, and cough. It is a syndrome rather than a specific disease because many respiratory viruses can cause it.

---

## Query 2 — Cause of the Common Cold

### Question

> Which virus is the most common cause of the common cold?

### Ground Truth

> Rhinoviruses are the most common cause of the common cold.

---

## Query 3 — Sleep and Common Cold Susceptibility

### Question

> How does sleep duration affect a person's susceptibility to developing the common cold?

### Ground Truth

> Shorter average sleep duration before rhinovirus exposure is associated with greater susceptibility to developing common cold symptoms.

---

# 📈 Ragas Metrics

The following Ragas metrics were used to evaluate the Medical Adaptive RAG pipeline.

| Metric                    | Purpose                                                                                         |
| ------------------------- | ----------------------------------------------------------------------------------------------- |
| **Faithfulness**          | Measures whether the generated answer is supported by the retrieved context                     |
| **Answer Relevancy**      | Measures how relevant the generated answer is to the user's question                            |
| **Context Precision**     | Measures whether relevant retrieved contexts are ranked above irrelevant contexts               |
| **Context Recall**        | Measures whether the retrieved context contains the information required to answer the question |
| **Context Entity Recall** | Measures how well important entities from the reference are retrieved                           |
| **Answer Similarity**     | Measures semantic similarity between the generated answer and ground truth                      |
| **Answer Correctness**    | Measures how well the generated answer matches the reference answer                             |

### Metrics Used in the Implementation

```python
METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness,
]
```

---

# 📊 Evaluation Results

The current evaluation was performed using **3 medical queries** with **Top-K = 5**.

| Metric                    |     Score | Interpretation              |
| ------------------------- | --------: | --------------------------- |
| **Faithfulness**          |  **0.90** | 🟢 Very Good                |
| **Context Recall**        |  **0.75** | 🟢 Good                     |
| **Context Entity Recall** |  **1.00** | 🟢 Excellent                |
| **Answer Similarity**     | **0.917** | 🟢 Very Good                |
| Answer Relevancy          |   **N/A** | ⚪ Not successfully computed |
| Context Precision         |   **N/A** | ⚪ Not successfully computed |
| Answer Correctness        |   **N/A** | ⚪ Not successfully computed |

> **Note:** The evaluation dataset contains only three questions, so these results should be considered an initial evaluation rather than a statistically comprehensive benchmark.

---

# 🔍 Metric Interpretation

## 1. Faithfulness — 0.90

**Faithfulness** measures whether the generated answer is supported by the retrieved context.

The system achieved a score of **0.90**, indicating that the generated responses were largely grounded in the retrieved medical information.

This is particularly important for a medical RAG system because answers should be based on retrieved evidence rather than unsupported model-generated information.

**Result: Very Good 🟢**

---

## 2. Context Recall — 0.75

**Context Recall** measures whether the retrieved context contains the information required to answer the question.

The system achieved a score of **0.75**.

This indicates that the retrieval system was able to retrieve most of the information required by the evaluation questions, but some relevant information may not always have been retrieved.

Possible areas for improvement include:

* Document chunking
* Embedding models
* Similarity thresholds
* Top-K tuning
* Hybrid retrieval
* Query reformulation
* Retrieval ranking

**Result: Good 🟢**

---

## 3. Context Entity Recall — 1.00

**Context Entity Recall** measures whether important entities and concepts from the reference answer are present in the retrieved context.

The system achieved a score of **1.00**.

This indicates that the important concepts and entities needed for the evaluated questions were successfully retrieved.

**Result: Excellent 🟢**

---

## 4. Answer Similarity — 0.917

**Answer Similarity** measures the semantic similarity between the generated answer and the ground-truth answer.

The system achieved a score of approximately **0.917**.

This indicates that the generated answers were highly similar in meaning to the manually prepared reference answers.

**Result: Very Good 🟢**

---

# ⚪ Metrics Not Successfully Computed

The following metrics returned `NaN` during the current evaluation:

* **Answer Relevancy**
* **Context Precision**
* **Answer Correctness**

These metrics are therefore reported as **N/A**, rather than zero.

`N/A` means that a valid numerical score was not produced during the current evaluation run. It does **not** mean that the system received a score of zero.

The other four metrics successfully produced numerical values and are used for the current performance summary.

---

# 📋 Overall Evaluation Summary

Based on the current three-query evaluation:

```text
Metric                    Score
------------------------------------------------
Faithfulness              0.900
Context Recall            0.750
Context Entity Recall     1.000
Answer Similarity         0.917
```

The current evaluation indicates that the Medical Adaptive RAG pipeline demonstrates:

* ✅ Strong answer grounding
* ✅ High semantic similarity to the expected answers
* ✅ Excellent retrieval of important entities
* ✅ Good retrieval coverage
* 🔧 Potential for improvement in context recall

The strongest result is **Context Entity Recall (1.00)**, indicating that important concepts required by the evaluation questions were successfully retrieved.

The **Faithfulness score of 0.90** also indicates strong grounding of generated responses in the retrieved context.

The **Answer Similarity score of 0.917** demonstrates that the generated answers are generally close in meaning to the manually created ground truths.

The main retrieval area for improvement is **Context Recall (0.75)**.

---

# 🧪 Example Evaluation Output

After running the evaluation, the system produces per-question scores and an average score summary.

Example:

```text
======================================================================
AVERAGE RAGAS SCORES
======================================================================

faithfulness             0.900000
context_recall           0.750000
context_entity_recall    1.000000
answer_similarity        0.916533
```

The detailed evaluation results are saved to:

```text
medical_rag/EvaluationScores.csv
```

---

# 📁 Evaluation Output

The generated CSV contains the evaluation information for each question, including the generated response and the available metric scores.

Example structure:

```text
user_input
response
retrieved_contexts
reference
faithfulness
answer_relevancy
context_precision
context_recall
context_entity_recall
answer_similarity
answer_correctness
```

---

# 🚀 Running the Evaluation

To run the evaluation with five retrieved chunks per question:

```bash
python medical_rag/src/evaluation.py --top-k 5
```

To explicitly evaluate the three questions:

```bash
python medical_rag/src/evaluation.py --limit 3 --top-k 5
```

To evaluate using a specific document:

```bash
python medical_rag/src/evaluation.py --limit 3 --top-k 5 --docname <document-name>
```

The evaluation results are saved automatically to:

```text
medical_rag/EvaluationScores.csv
```

---

# 🔬 Evaluation Methodology

The evaluation pipeline follows this process:

```text
Evaluation Query
       ↓
   ChromaDB
       ↓
Retrieve Top-K Contexts
       ↓
   Retrieved Context
       ↓
      Gemini
       ↓
Generated Response
       ↓
Compare with Ground Truth
       ↓
      Ragas
       ↓
Evaluation Metrics
```

For each evaluation query:

1. The query is sent to the retrieval pipeline.
2. The top **5** relevant chunks are retrieved.
3. The retrieved chunks are provided to the answer-generation model.
4. Gemini generates the final response.
5. The generated response is compared against the ground-truth answer.
6. Ragas calculates the available retrieval and generation metrics.
7. Per-question results are stored in `EvaluationScores.csv`.

---

# 📈 Future Improvements

The current evaluation contains only **3 manually created questions**. A larger evaluation dataset would provide a more reliable measurement of system performance.

Future improvements include:

* Increase the evaluation dataset from 3 questions to 20–50+ questions
* Include multiple medical topics
* Add factual questions
* Add comparative questions
* Add multi-step medical questions
* Add questions requiring multiple retrieved chunks
* Evaluate local retrieval separately from Tavily fallback retrieval
* Compare different Top-K values
* Experiment with different embedding models
* Tune the Adaptive RAG similarity threshold
* Improve document chunking
* Investigate hybrid dense + sparse retrieval
* Enable reliable scoring for the currently unavailable Ragas metrics
* Track evaluation scores across different versions of the system

---

# 🛠️ Potential Retrieval Improvements

The current **Context Recall score of 0.75** suggests that retrieval is the main area where the system can be improved.

Potential approaches include:

### Better Chunking

Experiment with:

* Chunk size
* Chunk overlap
* Semantic chunking
* Section-aware chunking

### Better Embeddings

Evaluate different embedding models to determine whether they improve medical-document retrieval.

### Top-K Optimization

Compare different values:

```text
Top-K = 3
Top-K = 5
Top-K = 7
Top-K = 10
```

and evaluate which configuration provides the best balance between retrieval quality and context size.

### Hybrid Retrieval

Combine:

```text
Dense Retrieval
       +
Sparse Retrieval
       ↓
Better Candidate Retrieval
```

This can help when medical terminology or exact keywords are important.

---

# 🔐 Medical Safety

This project is designed for **educational and research purposes**.

The chatbot should not be considered a medical professional and should not be used as a replacement for qualified healthcare advice.

Users should not rely on the chatbot for:

* Medical diagnosis
* Emergency decisions
* Prescription decisions
* Treatment decisions
* Individualized medical advice

For medical concerns, users should consult a qualified healthcare professional.

---

# 📌 Project Summary

This project demonstrates a **Medical Adaptive RAG** architecture that combines local medical knowledge retrieval with web-search fallback.

The overall pipeline is:

```text
                  User Question
                        ↓
                     Gradio
                        ↓
                 Adaptive RAG
                        ↓
                   ChromaDB
                        ↓
               Similarity Check
                  /         \
                 /           \
              High            Low
               ↓               ↓
          Local Context      Tavily
               \               /
                \             /
                 \           /
                    Gemini
                       ↓
                 Final Answer
                       ↓
               Answer + Sources
```

The initial Ragas evaluation demonstrates:

```text
Faithfulness              0.900
Context Recall            0.750
Context Entity Recall     1.000
Answer Similarity         0.917
```

These results indicate **strong answer grounding, high semantic similarity, and excellent entity retrieval**, while also identifying **context recall** as an area for further optimization.

---

## 👩‍💻 Project Status

**Current Status:** 🟢 Working

**Evaluation Status:** 🟢 Initial evaluation completed

**Evaluation Dataset:** 3 medical queries

**Retrieval:** ChromaDB + Adaptive routing

**Web Fallback:** Tavily

**LLM:** Gemini

**Interface:** Gradio

**Evaluation:** Ragas
