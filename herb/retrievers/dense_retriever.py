import json
import os

import faiss
from sentence_transformers import SentenceTransformer


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


# ============================================================
# LOAD FAISS INDEX
# ============================================================

print("=" * 70)
print("DENSE RETRIEVER")
print("=" * 70)

print("\nLoading FAISS index...")

index = faiss.read_index(
    INDEX_PATH
)

print(
    "Vectors loaded:",
    index.ntotal
)

print(
    "Vector dimension:",
    index.d
)


# ============================================================
# LOAD METADATA
# ============================================================

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
    "Metadata records:",
    len(metadata)
)


# ============================================================
# VERIFY ALIGNMENT
# ============================================================

if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index and metadata are not aligned.\n"
        f"FAISS vectors: {index.ntotal}\n"
        f"Metadata: {len(metadata)}"
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
# RETRIEVE
# ============================================================

def retrieve(query, top_k=TOP_K):
    """
    Retrieve top-k chunks using dense semantic search.
    """

    # --------------------------------------------------------
    # STEP 1:
    # Convert query into embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )


    # --------------------------------------------------------
    # STEP 2:
    # Normalize query vector
    #
    # Our document embeddings were also normalized.
    # Therefore Inner Product behaves like cosine similarity.
    # --------------------------------------------------------

    faiss.normalize_L2(
        query_embedding
    )


    # --------------------------------------------------------
    # STEP 3:
    # Search FAISS
    # --------------------------------------------------------

    scores, indices = index.search(
        query_embedding,
        top_k
    )


    # --------------------------------------------------------
    # STEP 4:
    # Map FAISS indices back to metadata
    # --------------------------------------------------------

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


        result = metadata[
            idx
        ].copy()


        result["rank"] = rank

        result["dense_score"] = float(
            score
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
    print("DENSE RETRIEVAL RESULTS")
    print("=" * 80)


    for result in results:

        print("\n" + "-" * 80)


        print(
            "Rank        :",
            result.get("rank")
        )


        print(
            "Dense Score :",
            round(
                result.get(
                    "dense_score",
                    0
                ),
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

    while True:

        query = input(
            "\nEnter your question "
            "(or type 'exit' to quit): "
        ).strip()


        if query.lower() == "exit":

            print(
                "\nExiting Dense Retriever."
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

# ksUser question
#      ↓
# SentenceTransformer
#      ↓
# 384-dimensional query vector
#      ↓
# FAISS
#      ↓
# Compare query vector with stored chunk vectors
#      ↓
# Top 5 most semantically similar chunk