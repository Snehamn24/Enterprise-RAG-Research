import json
import os

import faiss
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "embeddings.jsonl"
)

INDEX_FILE = os.path.join(
    BASE_DIR,
    "data",
    "herb.index"
)

METADATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "index_metadata.jsonl"
)


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_DIMENSION = 384


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

def load_embeddings():

    print("=" * 60)
    print("LOADING EMBEDDINGS")
    print("=" * 60)

    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError(
            f"Embeddings file not found:\n{EMBEDDINGS_FILE}"
        )

    vectors = []
    metadata = []

    with open(
        EMBEDDINGS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            record = json.loads(line)

            embedding = record.get("embedding")

            if embedding is None:
                continue

            if len(embedding) != EMBEDDING_DIMENSION:
                raise ValueError(
                    f"Invalid embedding dimension: "
                    f"{len(embedding)}"
                )

            vectors.append(embedding)

            metadata.append(record)

    print("Embeddings loaded:", len(vectors))

    return vectors, metadata


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_index(vectors):

    print("\n" + "=" * 60)
    print("BUILDING FAISS INDEX")
    print("=" * 60)

    vectors = np.asarray(
        vectors,
        dtype="float32"
    )

    print("Vector matrix shape:", vectors.shape)

    # Inner Product works well because
    # embeddings were normalized during encoding.
    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    index.add(vectors)

    print("Vectors indexed:", index.ntotal)

    return index


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(index):

    print("\n" + "=" * 60)
    print("SAVING FAISS INDEX")
    print("=" * 60)

    faiss.write_index(
        index,
        INDEX_FILE
    )

    print("FAISS index saved:")
    print(INDEX_FILE)


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(metadata):

    print("\n" + "=" * 60)
    print("SAVING INDEX METADATA")
    print("=" * 60)

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for record in metadata:

            # We don't need to store the
            # huge embedding again.
            record = {
                key: value
                for key, value in record.items()
                if key != "embedding"
            }

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("Metadata records:", len(metadata))
    print("Metadata saved:")
    print(METADATA_FILE)


# ============================================================
# MAIN
# ============================================================

def main():

    vectors, metadata = load_embeddings()

    if len(vectors) != len(metadata):
        raise ValueError(
            "Number of vectors and metadata records do not match."
        )

    index = build_index(vectors)

    save_index(index)

    save_metadata(metadata)

    print("\n" + "=" * 60)
    print("FAISS VECTOR STORE COMPLETED")
    print("=" * 60)

    print("Total vectors:", index.ntotal)
    print("Dimension:", EMBEDDING_DIMENSION)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()