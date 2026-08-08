import json
import os

from sentence_transformers import SentenceTransformer


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CHUNKS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "chunks.jsonl"
)

EMBEDDINGS_FILE = os.path.join(
    BASE_DIR,
    "data",
    "embeddings.jsonl"
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# LOAD CHUNKS
# ============================================================

def load_chunks():

    print("=" * 60)
    print("LOADING CHUNKS")
    print("=" * 60)

    if not os.path.exists(CHUNKS_FILE):
        raise FileNotFoundError(
            f"Chunks file not found:\n{CHUNKS_FILE}"
        )

    chunks = []

    with open(
        CHUNKS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            chunks.append(json.loads(line))

    print("Chunks loaded:", len(chunks))

    return chunks


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

def load_model():

    print("\n" + "=" * 60)
    print("LOADING EMBEDDING MODEL")
    print("=" * 60)

    print("Model:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    return model


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings(chunks, model):

    print("\n" + "=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    print("Texts to embed:", len(texts))

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    print("\nEmbedding generation completed.")

    print(
        "Embedding shape:",
        embeddings.shape
    )

    return embeddings


# ============================================================
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(chunks, embeddings):

    print("\n" + "=" * 60)
    print("SAVING EMBEDDINGS")
    print("=" * 60)

    with open(
        EMBEDDINGS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        for chunk, embedding in zip(
            chunks,
            embeddings
        ):

            record = {
                "chunk_id": chunk["chunk_id"],
                "parent_doc_id": chunk["parent_doc_id"],
                "doc_id": chunk["doc_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "product": chunk["product"],
                "embedding": embedding.tolist()
            }

            # Preserve additional metadata
            for key, value in chunk.items():

                if key not in record:
                    record[key] = value

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    print("Embeddings saved successfully.")
    print("Output file:", EMBEDDINGS_FILE)
    print("Records saved:", len(chunks))


# ============================================================
# MAIN
# ============================================================

def main():

    chunks = load_chunks()

    model = load_model()

    embeddings = generate_embeddings(
        chunks,
        model
    )

    save_embeddings(
        chunks,
        embeddings
    )

    print("\n" + "=" * 60)
    print("EMBEDDING PIPELINE COMPLETED")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()