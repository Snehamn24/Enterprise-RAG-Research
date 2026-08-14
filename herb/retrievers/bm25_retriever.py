import json
import os
import re

from rank_bm25 import BM25Okapi


# ============================================================
# PATHS
# ============================================================

# Current folder:
# C:\Enterprise-RAG\herb\retrievers
CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Parent folder:
# C:\Enterprise-RAG\herb
HERB_DIR = os.path.dirname(
    CURRENT_DIR
)

# Metadata generated during vector-store stage
METADATA_PATH = os.path.join(
    HERB_DIR,
    "data",
    "index_metadata.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def tokenize(text):
    """
    Convert text into tokens for BM25.

    Example:

    "Marketing Research Analysts"

    becomes roughly:

    ["marketing", "research", "analysts"]
    """

    # Convert to lowercase
    text = text.lower()

    # "_" should behave like a space.
    #
    # Example:
    # market_research_report
    #
    # becomes:
    # market research report
    text = text.replace("_", " ")

    # Replace hyphens with spaces
    text = text.replace("-", " ")

    # Remove symbols
    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    # Remove repeated spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Split sentence into words
    tokens = text.strip().split()

    return tokens


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BM25 RETRIEVER")
print("=" * 70)

print("\nLoading metadata...")


metadata = []

with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        record = json.loads(line)

        metadata.append(record)


print(
    "Records loaded:",
    len(metadata)
)


# ============================================================
# CREATE BM25 CORPUS
# ============================================================

print("\nPreparing BM25 corpus...")


corpus = []


for record in metadata:

    # For our CLEAN BM25 BASELINE,
    # search only the actual chunk text.
    text = record.get(
        "text",
        ""
    )

    corpus.append(
        tokenize(text)
    )


print(
    "Corpus documents:",
    len(corpus)
)


# ============================================================
# BUILD BM25 INDEX
# ============================================================

print("\nBuilding BM25 index...")


bm25 = BM25Okapi(
    corpus
)


print("BM25 index ready.")


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(query, top_k=TOP_K):
    """
    Retrieve top-k chunks using BM25.
    """

    # Convert query into tokens
    query_tokens = tokenize(
        query
    )


    # BM25 gives one score
    # for every chunk in our corpus.
    scores = bm25.get_scores(
        query_tokens
    )


    # Get indices sorted by score
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )


    results = []


    for rank, idx in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        # Copy metadata for this result
        result = metadata[idx].copy()

        # Add BM25 information
        result["rank"] = rank
        result["bm25_score"] = float(
            scores[idx]
        )

        results.append(
            result
        )


    return results


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results
):

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)


    print("\n" + "=" * 80)
    print("BM25 RESULTS")
    print("=" * 80)


    for result in results:

        print("\n" + "-" * 80)

        print(
            "Rank       :",
            result.get("rank")
        )

        print(
            "BM25 Score :",
            round(
                result.get(
                    "bm25_score",
                    0
                ),
                4
            )
        )

        print(
            "Doc ID     :",
            result.get("doc_id")
        )

        print(
            "Chunk ID   :",
            result.get("chunk_id")
        )

        print(
            "Parent Doc :",
            result.get(
                "parent_doc_id"
            )
        )

        print(
            "Source     :",
            result.get("source")
        )

        print(
            "Product    :",
            result.get("product")
        )

        print("\nTEXT:")

        print(
            result.get(
                "text",
                ""
            )
        )


# ============================================================
# MAIN
# ============================================================

def main():

    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit' to quit): "
        ).strip()


        if query.lower() == "exit":

            print(
                "\nExiting BM25 retriever."
            )

            break


        if not query:

            print(
                "Please enter a question."
            )

            continue


        results = retrieve(
            query,
            TOP_K
        )


        display_results(
            query,
            results
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()


# 41,790 chunks
#      ↓
# Break every chunk into words
#      ↓
# Build BM25 index
#      ↓
# User asks question
#      ↓
# Break question into words
#      ↓
# Compare important words
#      ↓
# Return Top 5

# metadata is not added this will be added afterwards to compare the results