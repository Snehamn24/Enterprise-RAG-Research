import json
import os


# ============================================================
# HERB DATASET PATH
# ============================================================

# loader.py is located at:
# C:\Enterprise-RAG\herb\loader.py
#
# HERB products are located at:
# C:\Enterprise-RAG\herb\data\HERB\products

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PRODUCTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "HERB",
    "products"
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_json(path):
    """
    Load a JSON file and return its contents.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# CONVERT A RECORD INTO OUR COMMON DOCUMENT FORMAT
# ============================================================

def create_record(
    doc_id,
    text,
    source,
    product,
    metadata=None
):
    """
    Convert different HERB sources into one common format.

    Every retrieval document will have:

        doc_id
        text
        source
        product

    Additional metadata is preserved when available.
    """

    record = {
        "doc_id": str(doc_id),
        "text": str(text),
        "source": source,
        "product": product
    }

    if metadata:
        record.update(metadata)

    return record


# ============================================================
# LOAD ALL PRODUCT DATA
# ============================================================

def load_products():

    if not os.path.exists(PRODUCTS_PATH):
        raise FileNotFoundError(
            f"HERB products directory not found:\n{PRODUCTS_PATH}"
        )

    product_files = [
        file
        for file in os.listdir(PRODUCTS_PATH)
        if file.endswith(".json")
    ]

    print("Products Found :", len(product_files))
    print(product_files[:5])

    return product_files


# ============================================================
# PROCESS ONE PRODUCT
# ============================================================

def process_product(product_file):

    product_path = os.path.join(
        PRODUCTS_PATH,
        product_file
    )

    product_name = os.path.splitext(product_file)[0]

    product = load_json(product_path)

    print("\n")
    print("=" * 60)
    print("Product:", product_name)
    print("=" * 60)

    print("\nAvailable fields:")
    print(product.keys())

    # --------------------------------------------------------
    # Unified retrieval corpus for this product
    # --------------------------------------------------------

    unified_records = []

    # --------------------------------------------------------
    # Evaluation questions
    # --------------------------------------------------------

    answerable_questions = []
    unanswerable_questions = []


    # ========================================================
    # 1. TEAM
    # ========================================================

    team = product.get("team")

    print("\nTEAM")
    print("Type:", type(team))

    if isinstance(team, dict):
        print("Team keys:", list(team.keys())[:10])

        # Team is structured metadata.
        # We DO NOT add it directly to the retrieval corpus.
        #
        # It will later be used for metadata-aware retrieval
        # and employee/entity enrichment.

    elif isinstance(team, list):
        print("Team records:", len(team))

        if team:
            print("First team record:")
            print(team[0])


    # ========================================================
    # 2. CUSTOMERS
    # ========================================================

    customers = product.get("customers")

    print("\nCUSTOMERS")
    print("Type:", type(customers))

    if isinstance(customers, list):

        print("Customer records:", len(customers))

        if customers:
            print("First customer:")
            print(customers[0])

    elif isinstance(customers, dict):

        print("Customer keys:")
        print(list(customers.keys())[:10])


    # ========================================================
    # 3. SLACK
    # ========================================================

    slack = product.get("slack", [])

    print("\nSLACK")
    print("Records:", len(slack))

    for item in slack:

        message = item.get("Message", {})
        user = message.get("User", {})

        text = user.get("text", "")

        doc_id = (
            item.get("id")
            or user.get("utterranceID")
            or "unknown_slack"
        )

        metadata = {
            "channel": item.get("Channel", {}).get("name"),
            "channel_id": item.get("Channel", {}).get("channelID"),
            "user_id": user.get("userId"),
            "timestamp": user.get("timestamp")
        }

        unified_records.append(
            create_record(
                doc_id=doc_id,
                text=text,
                source="slack",
                product=product_name,
                metadata=metadata
            )
        )


    # ========================================================
    # 4. DOCUMENTS
    # ========================================================

    documents = product.get("documents", [])

    print("\nDOCUMENTS")
    print("Records:", len(documents))

    for item in documents:

        doc_id = item.get("id")

        # Some HERB documents use "id", while some may
        # expose another identifier.
        if not doc_id:
            doc_id = item.get(
                "doc_id",
                f"{product_name}_document"
            )

        text = (
            item.get("content")
            or item.get("text")
            or ""
        )

        metadata = {}

        # Preserve useful document metadata
        for key in [
            "title",
            "document_type",
            "date",
            "author",
            "authors"
        ]:
            if key in item:
                metadata[key] = item[key]

        unified_records.append(
            create_record(
                doc_id=doc_id,
                text=text,
                source="document",
                product=product_name,
                metadata=metadata
            )
        )


    # ========================================================
    # 5. MEETING TRANSCRIPTS
    # ========================================================

    meeting_transcripts = product.get(
        "meeting_transcripts",
        []
    )

    print("\nMEETING TRANSCRIPTS")
    print("Records:", len(meeting_transcripts))

    for item in meeting_transcripts:

        doc_id = item.get(
            "id",
            f"{product_name}_meeting"
        )

        text = item.get(
            "transcript",
            item.get("text", "")
        )

        metadata = {
            "date": item.get("date"),
            "document_type": item.get("document_type"),
            "participants": item.get("participants", [])
        }

        unified_records.append(
            create_record(
                doc_id=doc_id,
                text=text,
                source="meeting",
                product=product_name,
                metadata=metadata
            )
        )


    # ========================================================
    # 6. MEETING CHATS
    # ========================================================

    meeting_chats = product.get(
        "meeting_chats",
        []
    )

    print("\nMEETING CHATS")
    print("Records:", len(meeting_chats))

    for item in meeting_chats:

        doc_id = item.get(
            "id",
            f"{product_name}_meeting_chat"
        )

        text = (
            item.get("text")
            or item.get("content")
            or ""
        )

        metadata = {}

        for key in [
            "date",
            "timestamp",
            "channel"
        ]:
            if key in item:
                metadata[key] = item[key]

        unified_records.append(
            create_record(
                doc_id=doc_id,
                text=text,
                source="meeting_chat",
                product=product_name,
                metadata=metadata
            )
        )


    # ========================================================
    # 7. URLS
    # ========================================================

    urls = product.get("urls", [])

    print("\nURLS")
    print("Records:", len(urls))

    for item in urls:

        doc_id = item.get(
            "id",
            f"{product_name}_url"
        )

        description = item.get(
            "description",
            ""
        )

        link = item.get(
            "link",
            ""
        )

        # We combine description + URL because the
        # description contains useful semantic information.
        text = f"{description}\nURL: {link}"

        metadata = {
            "url": link
        }

        unified_records.append(
            create_record(
                doc_id=doc_id,
                text=text,
                source="url",
                product=product_name,
                metadata=metadata
            )
        )


    # ========================================================
    # 8. PULL REQUESTS
    # ========================================================

    prs = product.get("prs", [])

    print("\nPULL REQUESTS")
    print("Records:", len(prs))

    for item in prs:

        doc_id = item.get(
            "id",
            f"{product_name}_pr"
        )

        title = item.get("title", "")
        summary = item.get("summary", "")

        text = (
            f"Title: {title}\n"
            f"Summary: {summary}"
        )

        metadata = {
            "pr_number": item.get("number"),
            "state": item.get("state"),
            "mergeable": item.get("mergeable"),
            "merged": item.get("merged"),
            "author": item.get("user", {}).get("login"),
            "reviews": item.get("reviews", []),
            "link": item.get("link")
        }

        unified_records.append(
            create_record(
                doc_id=doc_id,
                text=text,
                source="pull_request",
                product=product_name,
                metadata=metadata
            )
        )


    # ========================================================
    # 9. ANSWERABLE QUESTIONS
    # ========================================================

    answerable_questions = product.get(
        "answerable_questions",
        []
    )

    print("\nANSWERABLE QUESTIONS")
    print("Records:", len(answerable_questions))

    if answerable_questions:

        print("First answerable question:")
        print(answerable_questions[0])


    # ========================================================
    # 10. UNANSWERABLE QUESTIONS
    # ========================================================

    unanswerable_questions = product.get(
        "unanswerable_questions",
        []
    )

    print("\nUNANSWERABLE QUESTIONS")
    print("Records:", len(unanswerable_questions))

    if unanswerable_questions:

        print("First unanswerable question:")
        print(unanswerable_questions[0])


    # ========================================================
    # RETURN EVERYTHING
    # ========================================================

    return (
        unified_records,
        answerable_questions,
        unanswerable_questions
    )


# ============================================================
# MAIN
# ============================================================

def main():

    product_files = load_products()

    # These will eventually contain data from ALL 30 products.
    all_unified_records = []
    all_answerable_questions = []
    all_unanswerable_questions = []


    # ========================================================
    # PROCESS EVERY PRODUCT
    # ========================================================

    for product_file in product_files:

        (
            unified_records,
            answerable_questions,
            unanswerable_questions
        ) = process_product(product_file)

        all_unified_records.extend(
            unified_records
        )

        all_answerable_questions.extend(
            answerable_questions
        )

        all_unanswerable_questions.extend(
            unanswerable_questions
        )


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL HERB DATASET SUMMARY")
    print("=" * 60)

    print(
        "Total unified retrieval records:",
        len(all_unified_records)
    )

    print(
        "Total answerable questions:",
        len(all_answerable_questions)
    )

    print(
        "Total unanswerable questions:",
        len(all_unanswerable_questions)
    )

    print("\nSources in unified corpus:")

    source_counts = {}

    for record in all_unified_records:

        source = record["source"]

        source_counts[source] = (
            source_counts.get(source, 0) + 1
        )

    for source, count in source_counts.items():

        print(
            f"{source:20s}: {count}"
        )


    # ========================================================
    # SHOW SAMPLE RECORDS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("SAMPLE UNIFIED RECORD")
    print("=" * 60)

    if all_unified_records:

        print(
            json.dumps(
                all_unified_records[0],
                indent=2,
                ensure_ascii=False
            )
        )

    # ============================================================
    # SAVE FINAL UNIFIED DATASET
    # ============================================================

    OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "unified_data.jsonl"
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for record in unified_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("\n" + "=" * 60)
    print("UNIFIED DATASET SAVED")
    print("=" * 60)
    print(f"File: {OUTPUT_FILE}")
    print(f"Records saved: {len(unified_records)}")


    # ========================================================
    # IMPORTANT:
    #
    # We are NOT saving unified_data.jsonl yet.
    #
    # First we verify that all HERB fields are being loaded
    # correctly across all 30 products.
    #
    # After verification, we will create:
    #
    # herb/
    # ├── unified_data.jsonl
    # ├── answerable_questions.jsonl
    # └── unanswerable_questions.jsonl
    #
    # ========================================================


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()