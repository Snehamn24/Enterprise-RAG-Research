import json
import os
import re

import faiss

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

HERB_DIR = os.path.dirname(
    CURRENT_DIR
)


INDEX_PATH = os.path.join(
    HERB_DIR,
    "data",
    "herb.index"
)


METADATA_PATH = os.path.join(
    HERB_DIR,
    "data",
    "index_metadata.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5


# We retrieve more candidates from each method
# before combining them.
CANDIDATE_K = 200


# Standard constant used for
# Reciprocal Rank Fusion.
RRF_K = 60


# ============================================================
# TOKENIZATION FOR BM25
# ============================================================

def tokenize(text):

    # Convert to lowercase
    text = str(text).lower()

    # Treat underscores and hyphens as spaces
    text = text.replace("_", " ")
    text = text.replace("-", " ")

    # Remove special symbols
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

    return text.strip().split()


# ============================================================
# LOAD METADATA
# ============================================================

print("=" * 70)
print("HYBRID RETRIEVER")
print("=" * 70)


print("\nLoading metadata...")


metadata = []


with open(
    METADATA_PATH,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        metadata.append(
            json.loads(line)
        )


print(
    "Metadata records:",
    len(metadata)
)


# ============================================================
# BUILD BM25 INDEX
# ============================================================

print("\nBuilding BM25 index...")


tokenized_corpus = []


for record in metadata:

    # CLEAN BASELINE:
    # BM25 searches only chunk text.
    text = record.get(
        "text",
        ""
    )

    tokenized_corpus.append(
        tokenize(text)
    )


bm25 = BM25Okapi(
    tokenized_corpus
)


print("BM25 index ready.")


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("\nLoading FAISS index...")


index = faiss.read_index(
    INDEX_PATH
)


print(
    "FAISS vectors:",
    index.ntotal
)

print(
    "Vector dimension:",
    index.d
)


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index and metadata "
        "are not aligned."
    )


print(
    "Index and metadata alignment verified."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")


model = SentenceTransformer(
    MODEL_NAME
)


print(
    "Model loaded:",
    MODEL_NAME
)


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def bm25_retrieve(
    query,
    candidate_k=CANDIDATE_K
):

    query_tokens = tokenize(
        query
    )


    scores = bm25.get_scores(
        query_tokens
    )


    # Sort all chunk indices according
    # to BM25 score.
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda idx: scores[idx],
        reverse=True
    )


    results = []


    for rank, idx in enumerate(
        ranked_indices[:candidate_k],
        start=1
    ):

        results.append({
            "index": idx,
            "rank": rank,
            "score": float(
                scores[idx]
            )
        })


    return results


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_retrieve(
    query,
    candidate_k=CANDIDATE_K
):

    # Convert query into embedding
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )


    # Normalize query
    faiss.normalize_L2(
        query_embedding
    )


    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        candidate_k
    )


    results = []


    for rank, (
        score,
        idx
    ) in enumerate(
        zip(
            scores[0],
            indices[0]
        ),
        start=1
    ):

        if idx == -1:
            continue


        results.append({
            "index": int(idx),
            "rank": rank,
            "score": float(score)
        })


    return results


# ============================================================
# RECIPROCAL RANK FUSION
# ============================================================

def reciprocal_rank_fusion(
    bm25_results,
    dense_results
):

    """
    Combine BM25 and Dense rankings.

    RRF formula:

        1 / (RRF_K + rank)

    A document appearing high in BOTH
    rankings receives a stronger score.
    """

    fused = {}


    # --------------------------------------------------------
    # ADD BM25 RANKINGS
    # --------------------------------------------------------

    for item in bm25_results:

        idx = item["index"]


        if idx not in fused:

            fused[idx] = {
                "rrf_score": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "dense_rank": None,
                "dense_score": None
            }


        fused[idx]["bm25_rank"] = (
            item["rank"]
        )

        fused[idx]["bm25_score"] = (
            item["score"]
        )


        fused[idx]["rrf_score"] += (
            1
            /
            (
                RRF_K
                +
                item["rank"]
            )
        )


    # --------------------------------------------------------
    # ADD DENSE RANKINGS
    # --------------------------------------------------------

    for item in dense_results:

        idx = item["index"]


        if idx not in fused:

            fused[idx] = {
                "rrf_score": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "dense_rank": None,
                "dense_score": None
            }


        fused[idx]["dense_rank"] = (
            item["rank"]
        )

        fused[idx]["dense_score"] = (
            item["score"]
        )


        fused[idx]["rrf_score"] += (
            1
            /
            (
                RRF_K
                +
                item["rank"]
            )
        )


    return fused


# ============================================================
# HYBRID RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k=TOP_K
):

    # --------------------------------------------------------
    # STEP 1:
    # BM25 candidates
    # --------------------------------------------------------

    bm25_results = bm25_retrieve(
        query
    )


    # --------------------------------------------------------
    # STEP 2:
    # Dense candidates
    # --------------------------------------------------------

    dense_results = dense_retrieve(
        query
    )


    # --------------------------------------------------------
    # STEP 3:
    # Fuse rankings
    # --------------------------------------------------------

    fused = reciprocal_rank_fusion(
        bm25_results,
        dense_results
    )


    # --------------------------------------------------------
    # STEP 4:
    # Sort according to RRF score
    # --------------------------------------------------------

    ranked_indices = sorted(
        fused.keys(),
        key=lambda idx: fused[idx][
            "rrf_score"
        ],
        reverse=True
    )


    # --------------------------------------------------------
    # STEP 5:
    # Build final results
    # --------------------------------------------------------

    results = []


    for rank, idx in enumerate(
        ranked_indices[:top_k],
        start=1
    ):

        result = metadata[
            idx
        ].copy()


        result["rank"] = rank


        result["rrf_score"] = float(
            fused[idx]["rrf_score"]
        )


        result["bm25_rank"] = (
            fused[idx]["bm25_rank"]
        )


        result["bm25_score"] = (
            fused[idx]["bm25_score"]
        )


        result["dense_rank"] = (
            fused[idx]["dense_rank"]
        )


        result["dense_score"] = (
            fused[idx]["dense_score"]
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
    print("HYBRID BM25 + DENSE RESULTS")
    print("=" * 80)


    for result in results:

        print("\n" + "-" * 80)


        print(
            "Rank        :",
            result.get("rank")
        )


        print(
            "RRF Score   :",
            round(
                result.get(
                    "rrf_score",
                    0
                ),
                6
            )
        )


        print(
            "BM25 Rank   :",
            result.get(
                "bm25_rank"
            )
        )


        print(
            "Dense Rank  :",
            result.get(
                "dense_rank"
            )
        )


        bm25_score = result.get(
            "bm25_score"
        )

        if bm25_score is not None:

            print(
                "BM25 Score  :",
                round(
                    bm25_score,
                    4
                )
            )


        dense_score = result.get(
            "dense_score"
        )

        if dense_score is not None:

            print(
                "Dense Score :",
                round(
                    dense_score,
                    4
                )
            )


        print(
            "Doc ID      :",
            result.get(
                "doc_id"
            )
        )


        print(
            "Chunk ID    :",
            result.get(
                "chunk_id"
            )
        )


        print(
            "Parent Doc  :",
            result.get(
                "parent_doc_id"
            )
        )


        print(
            "Source      :",
            result.get(
                "source"
            )
        )


        print(
            "Product     :",
            result.get(
                "product"
            )
        )


        if result.get("author"):

            print(
                "Author      :",
                result.get(
                    "author"
                )
            )


        if result.get("user_id"):

            print(
                "User ID     :",
                result.get(
                    "user_id"
                )
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

    print(
        "\nBM25 candidates :",
        CANDIDATE_K
    )

    print(
        "Dense candidates:",
        CANDIDATE_K
    )

    print(
        "Final results   :",
        TOP_K
    )


    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit' to quit): "
        ).strip()


        if query.lower() == "exit":

            print(
                "\nExiting Hybrid Retriever."
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
# RUN
# ============================================================

if __name__ == "__main__":
    main()