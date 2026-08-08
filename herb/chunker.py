import json
import os

# ============================================================
# PATHS
# ============================================================

# Location of the unified dataset created by loader.py
INPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "unified_data.jsonl"
)

# Output file containing the chunks
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "chunks.jsonl"
)


# ============================================================
# CHUNKING SETTINGS
# ============================================================

# Maximum number of characters in one chunk
CHUNK_SIZE = 1000

# Number of characters shared between consecutive chunks
# This helps preserve context between chunks.
CHUNK_OVERLAP = 200


# ============================================================
# TEXT CHUNKING FUNCTION
# ============================================================

def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split a long text into overlapping chunks.

    Example:

        Original text
        ├───────────────┤
                 ↓
        Chunk 1
        ├───────────────┤
                  ├───────────────┤
                  Chunk 2

    The overlap helps prevent important information from
    being lost at chunk boundaries.
    """

    # Ignore empty text
    if not text:
        return []

    text = text.strip()

    # If text is already small enough, return it as one chunk
    if len(text) <= chunk_size:
        return [text]

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        # Extract the chunk
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping overlap
        start = end - overlap

    return chunks


# ============================================================
# LOAD UNIFIED DATASET
# ============================================================

print("=" * 60)
print("LOADING UNIFIED DATASET")
print("=" * 60)

print(f"Input file: {INPUT_FILE}")

records = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:

    for line in f:

        if line.strip():
            records.append(json.loads(line))


print(f"Records loaded: {len(records)}")


# ============================================================
# CREATE CHUNKS
# ============================================================

print("\n" + "=" * 60)
print("CREATING CHUNKS")
print("=" * 60)

chunked_records = []

chunk_id = 0

for record in records:

    text = record.get("text", "")

    chunks = create_chunks(text)

    for chunk in chunks:

        # Copy the original metadata
        chunk_record = record.copy()

        # Replace the text with the chunk
        chunk_record["text"] = chunk

        # Give every chunk a unique ID
        chunk_record["chunk_id"] = f"chunk_{chunk_id}"

        # Keep track of which original document this came from
        chunk_record["parent_doc_id"] = record.get("doc_id")

        chunked_records.append(chunk_record)

        chunk_id += 1


# ============================================================
# SAVE CHUNKS
# ============================================================

print("\n" + "=" * 60)
print("SAVING CHUNKS")
print("=" * 60)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for record in chunked_records:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CHUNKING COMPLETE")
print("=" * 60)

print(f"Original records : {len(records)}")
print(f"Total chunks     : {len(chunked_records)}")
print(f"Chunk size       : {CHUNK_SIZE}")
print(f"Chunk overlap    : {CHUNK_OVERLAP}")
print(f"Output file      : {OUTPUT_FILE}")


# ============================================================
# SAMPLE CHUNK
# ============================================================

if chunked_records:

    print("\n" + "=" * 60)
    print("SAMPLE CHUNK")
    print("=" * 60)

    print(
        json.dumps(
            chunked_records[0],
            indent=2,
            ensure_ascii=False
        )
    )