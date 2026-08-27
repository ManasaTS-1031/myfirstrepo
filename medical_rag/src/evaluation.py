import argparse
import os
from pathlib import Path

from datasets import Dataset
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from ragas import evaluate

from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness,
)

from llm_response import llm_chat
from retrieval import retrieve


# ============================================================
# EVALUATION QUERIES
# ============================================================

QUERIES = [
    "What is the definition of the common cold and what symptoms characterize it?",

    "Which virus is the most common cause of the common cold?",

    "How does sleep duration affect a person's susceptibility to developing the common cold?",

]


# ============================================================
# GROUND TRUTHS
# ============================================================

GROUND_TRUTHS = [
    (
        "The common cold is a mild, self-limiting upper respiratory tract "
        "infection characterized by nasal stuffiness and discharge, sneezing, "
        "sore throat, and cough. It is a syndrome rather than a specific "
        "disease because many respiratory viruses can cause it."
    ),

    (
        "Rhinoviruses are the most common cause of the common cold."
    ),

    (
        "Shorter average sleep duration before rhinovirus exposure is "
        "associated with greater susceptibility to developing common "
        "cold symptoms."
    ),

]


# ============================================================
# RAGAS METRICS
# ============================================================

# IMPORTANT:
# These are metric objects from the legacy Ragas API.
# Do NOT use Faithfulness(), AnswerRelevancy(), etc.
#
# ChatGoogleGenerativeAI is passed to evaluate() below.

METRICS = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_entity_recall,
    answer_similarity,
    answer_correctness,
]


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(limit=None, top_k=5, docname_filter=None):
    """
    Retrieve contexts, generate answers, and construct
    a HuggingFace Dataset for Ragas evaluation.
    """

    selected_queries = QUERIES[:limit] if limit else QUERIES

    rows = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    total_queries = len(selected_queries)

    for index, query in enumerate(selected_queries, start=1):

        print()
        print("=" * 70)
        print(f"Processing query {index}/{total_queries}")
        print(f"Query: {query}")
        print("=" * 70)

        # ----------------------------------------------------
        # Retrieve relevant chunks
        # ----------------------------------------------------

        matches = retrieve(
            query,
            top_k=top_k,
            docname_filter=docname_filter,
        )

        contexts = [
            match["text"]
            for match in matches
            if match.get("text")
        ]

        print(f"Retrieved contexts: {len(contexts)}")

        # ----------------------------------------------------
        # Combine retrieved contexts
        # ----------------------------------------------------

        context = "\n\n".join(contexts)

        # ----------------------------------------------------
        # Generate RAG response
        # ----------------------------------------------------

        print("Generating answer...")

        response = llm_chat(
            context=context,
            user_query=query,
        )

        print("Response:")
        print(response)

        # ----------------------------------------------------
        # Save row
        # ----------------------------------------------------

        rows["user_input"].append(query)
        rows["response"].append(response)
        rows["retrieved_contexts"].append(contexts)

        # Use the current loop index rather than
        # QUERIES.index(query), which is safer if
        # duplicate queries are ever added.
        rows["reference"].append(
            GROUND_TRUTHS[index - 1]
        )

    return Dataset.from_dict(rows)


# ============================================================
# RUN RAGAS EVALUATION
# ============================================================

def run_evaluation(dataset):
    """
    Run Ragas evaluation using Gemini as the evaluator LLM
    and Sentence Transformer embeddings for embedding-based
    metrics.
    """

    # --------------------------------------------------------
    # Load environment variables
    # --------------------------------------------------------

    load_dotenv()

    gemini_key = os.getenv("GEMINI_KEY")

    if not gemini_key:
        raise ValueError(
            "GEMINI_KEY is missing from your .env file"
        )

    # --------------------------------------------------------
    # Configure Google API key
    # --------------------------------------------------------

    os.environ.setdefault(
        "GOOGLE_API_KEY",
        gemini_key,
    )

    # --------------------------------------------------------
    # Evaluation model
    # --------------------------------------------------------

    evaluator_model = os.getenv(
        "GEMINI_EVAL_MODEL",
        "gemini-3.6-flash",
    )

    print()
    print("=" * 70)
    print("Loading Ragas evaluation LLM...")
    print(f"Evaluation model: {evaluator_model}")
    print("=" * 70)

    evaluator_llm = ChatGoogleGenerativeAI(
        model=evaluator_model,
        temperature=0,
        google_api_key=gemini_key,
    )

    # --------------------------------------------------------
    # Evaluation embeddings
    # --------------------------------------------------------

    print("Loading evaluation embedding model...")

    evaluator_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # --------------------------------------------------------
    # Run Ragas
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("Running Ragas evaluation...")
    print("=" * 70)

    result = evaluate(
        dataset,
        metrics=METRICS,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the medical RAG pipeline "
            "using Ragas."
        )
    )

    # --------------------------------------------------------
    # Number of queries
    # --------------------------------------------------------

    parser.add_argument(
        "--limit",
        type=int,
        help=(
            "Evaluate only the first N questions."
        ),
    )

    # --------------------------------------------------------
    # Top-K
    # --------------------------------------------------------

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help=(
            "Number of retrieved chunks per question. "
            "Default: 5"
        ),
    )

    # --------------------------------------------------------
    # Document filter
    # --------------------------------------------------------

    parser.add_argument(
        "--docname",
        help=(
            "Restrict retrieval to one ingested "
            "document name."
        ),
    )

    # --------------------------------------------------------
    # Output CSV
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "EvaluationScores.csv"
        ),
        help=(
            "CSV path for per-question scores."
        ),
    )

    args = parser.parse_args()

    # ========================================================
    # VALIDATION
    # ========================================================

    if args.limit is not None:

        if not 1 <= args.limit <= len(QUERIES):

            parser.error(
                f"--limit must be between 1 and "
                f"{len(QUERIES)}"
            )

    if args.top_k < 1:

        parser.error(
            "--top-k must be at least 1"
        )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    print()
    print("=" * 70)
    print("MEDICAL RAG EVALUATION")
    print("=" * 70)

    print(
        f"Number of queries : "
        f"{args.limit or len(QUERIES)}"
    )

    print(
        f"Top-K             : "
        f"{args.top_k}"
    )

    print(
        f"Document filter   : "
        f"{args.docname or 'None'}"
    )

    print(
        f"Output            : "
        f"{args.output}"
    )

    # ========================================================
    # BUILD DATASET
    # ========================================================

    dataset = build_dataset(
        limit=args.limit,
        top_k=args.top_k,
        docname_filter=args.docname,
    )

    print()
    print("=" * 70)
    print("Dataset created successfully.")
    print(f"Dataset rows: {len(dataset)}")
    print("=" * 70)

    # ========================================================
    # RUN EVALUATION
    # ========================================================

    result = run_evaluation(dataset)

    # ========================================================
    # CONVERT RESULTS TO PANDAS
    # ========================================================

    scores = result.to_pandas()

    # ========================================================
    # SAVE CSV
    # ========================================================

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores.to_csv(
        args.output,
        index=False,
    )

    # ========================================================
    # FIND AVAILABLE METRIC COLUMNS
    # ========================================================

    score_columns = [
        metric.name
        for metric in METRICS
        if metric.name in scores.columns
    ]

    # ========================================================
    # PRINT AVERAGE SCORES
    # ========================================================

    print()
    print("=" * 70)
    print("AVERAGE RAGAS SCORES")
    print("=" * 70)

    if score_columns:

        averages = scores[
            score_columns
        ].mean(numeric_only=True)

        print(
            averages.to_string()
        )

    else:

        print(
            "No metric score columns were found."
        )

    # ========================================================
    # PRINT PER-QUESTION SCORES
    # ========================================================

    print()
    print("=" * 70)
    print("PER-QUESTION SCORES")
    print("=" * 70)

    display_columns = [
        column
        for column in [
            "user_input",
            "response",
            *score_columns,
        ]
        if column in scores.columns
    ]

    print(
        scores[display_columns].to_string(
            index=False
        )
    )

    # ========================================================
    # FINISHED
    # ========================================================

    print()
    print("=" * 70)
    print("Evaluation completed successfully.")
    print(f"Saved results to:")
    print(args.output)
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()