import json
import os
import re
from collections import defaultdict

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

EMPLOYEE_PATH = os.path.join(
    HERB_DIR,
    "data",
    "HERB",
    "metadata",
    "employee.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K = 5

DENSE_K = 100
BM25_K = 100

RRF_K = 60


# ============================================================
# TEXT FUNCTIONS
# ============================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def tokenize(text):

    return normalize_text(
        text
    ).split()


# Small normalization:
# analysts -> analyst
# managers -> manager

def singular(word):

    if word.endswith("ies") and len(word) > 4:

        return word[:-3] + "y"

    if word.endswith("s") and len(word) > 3:

        return word[:-1]

    return word


def token_set(text):

    return {
        singular(word)
        for word in tokenize(text)
    }


# ============================================================
# LOAD CHUNK METADATA
# ============================================================

print("=" * 70)
print("METADATA-AWARE HYBRID RETRIEVER")
print("=" * 70)

print("\nLoading chunk metadata...")


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
# LOAD EMPLOYEE METADATA
# ============================================================

print("\nLoading employee metadata...")


with open(
    EMPLOYEE_PATH,
    "r",
    encoding="utf-8"
) as file:

    employee_data = json.load(file)


employees = {}


if isinstance(employee_data, dict):

    for key, value in employee_data.items():

        if not isinstance(value, dict):
            continue

        employee_id = (
            value.get("employee_id")
            or key
        )

        employees[
            employee_id
        ] = value


elif isinstance(employee_data, list):

    for value in employee_data:

        employee_id = value.get(
            "employee_id"
        )

        if employee_id:

            employees[
                employee_id
            ] = value


print(
    "Employees loaded:",
    len(employees)
)


# ============================================================
# EMPLOYEE HELPERS
# ============================================================

def get_employee_role(employee_id):

    if not employee_id:

        return ""

    employee = employees.get(
        employee_id,
        {}
    )

    return str(
        employee.get(
            "role",
            ""
        )
    )


def employee_description(employee_id):

    if not employee_id:

        return ""

    employee = employees.get(
        employee_id
    )

    if not employee:

        return str(
            employee_id
        )

    return " ".join([
        str(employee_id),

        str(
            employee.get(
                "name",
                ""
            )
        ),

        str(
            employee.get(
                "role",
                ""
            )
        ),

        str(
            employee.get(
                "location",
                ""
            )
        )
    ])


# ============================================================
# BUILD ENRICHED TEXT
# ============================================================

def build_search_text(record):

    """
    BM25 will search not only chunk text,
    but also useful enterprise metadata.
    """

    fields = [

        record.get(
            "text",
            ""
        ),

        record.get(
            "doc_id",
            ""
        ),

        record.get(
            "parent_doc_id",
            ""
        ),

        record.get(
            "product",
            ""
        ),

        record.get(
            "source",
            ""
        ),

        record.get(
            "channel",
            ""
        ),

        employee_description(
            record.get(
                "author"
            )
        ),

        employee_description(
            record.get(
                "user_id"
            )
        )
    ]

    return " ".join(
        str(field)
        for field in fields
        if field
    )


# ============================================================
# CREATE SEARCHABLE CORPUS
# ============================================================

print("\nBuilding enriched corpus...")


search_texts = [
    build_search_text(record)
    for record in metadata
]


tokenized_corpus = [
    tokenize(text)
    for text in search_texts
]


# ============================================================
# BUILD BM25
# ============================================================

print("Building BM25 index...")


bm25 = BM25Okapi(
    tokenized_corpus
)


print("BM25 ready.")


# ============================================================
# BUILD PRODUCT LOOKUP
# ============================================================

product_to_indices = defaultdict(
    list
)

products = set()


for idx, record in enumerate(
    metadata
):

    product = record.get(
        "product"
    )

    if product:

        products.add(
            product
        )

        product_to_indices[
            product
        ].append(
            idx
        )


print(
    "Products detected:",
    len(products)
)


# ============================================================
# LOAD FAISS
# ============================================================

print("\nLoading FAISS index...")


index = faiss.read_index(
    INDEX_PATH
)


print(
    "FAISS vectors:",
    index.ntotal
)


if index.ntotal != len(metadata):

    raise ValueError(
        "FAISS index and metadata "
        "are not aligned."
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
# PRODUCT DETECTION
# ============================================================

def detect_product(query):

    query_normalized = normalize_text(
        query
    )


    for product in sorted(
        products,
        key=len,
        reverse=True
    ):

        if normalize_text(
            product
        ) in query_normalized:

            return product


    return None


# ============================================================
# ROLE MATCHING
# ============================================================

def role_matches_query(
    query,
    record
):

    query_words = token_set(
        query
    )


    employee_ids = [

        record.get(
            "author"
        ),

        record.get(
            "user_id"
        )
    ]


    for employee_id in employee_ids:

        role = get_employee_role(
            employee_id
        )


        if not role:

            continue


        role_words = token_set(
            role
        )


        if not role_words:

            continue


        matched = (
            role_words
            .intersection(
                query_words
            )
        )


        match_ratio = (
            len(matched)
            /
            len(role_words)
        )


        if match_ratio >= 0.80:

            return True


    return False


# ============================================================
# ADMIN NOISE DETECTION
# ============================================================

def is_admin_noise(record):

    if record.get(
        "source"
    ) != "slack":

        return False


    if record.get(
        "user_id"
    ) != "slack_admin_bot":

        return False


    text = normalize_text(
        record.get(
            "text",
            ""
        )
    )


    noise_terms = [

        "joined",

        "created this channel",

        "joined via invite"
    ]


    return any(
        term in text
        for term in noise_terms
    )


# ============================================================
# METADATA BONUS
# ============================================================

def metadata_bonus(
    query,
    record
):

    bonus = 0.0


    # --------------------------------------------------------
    # Employee-role match
    # --------------------------------------------------------

    if role_matches_query(
        query,
        record
    ):

        bonus += 0.020


    # --------------------------------------------------------
    # If query asks about employee IDs and
    # document author matches the requested role,
    # document evidence is especially useful.
    # --------------------------------------------------------

    query_text = normalize_text(
        query
    )


    if (
        "employee id" in query_text
        and
        record.get("author")
        and
        role_matches_query(
            query,
            record
        )
    ):

        bonus += 0.010


    # --------------------------------------------------------
    # Penalize useless Slack administrative messages
    # --------------------------------------------------------

    if is_admin_noise(
        record
    ):

        bonus -= 0.025


    return bonus


# ============================================================
# DENSE RETRIEVAL
# ============================================================

def dense_retrieve(
    query,
    product
):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )


    faiss.normalize_L2(
        query_embedding
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # We search all vectors first because the
    # correct ForecastForce result might have
    # global rank 52 or 144.
    #
    # Then we keep only the detected product.
    # --------------------------------------------------------

    search_k = index.ntotal


    scores, indices = index.search(
        query_embedding,
        search_k
    )


    results = []


    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx == -1:

            continue


        record = metadata[
            idx
        ]


        # Hard metadata filtering
        if (
            product
            and
            record.get(
                "product"
            ) != product
        ):

            continue


        results.append({
            "index": int(idx),
            "score": float(score)
        })


        if len(results) >= DENSE_K:

            break


    # Add dense ranks
    for rank, item in enumerate(
        results,
        start=1
    ):

        item[
            "rank"
        ] = rank


    return results


# ============================================================
# BM25 RETRIEVAL
# ============================================================

def bm25_retrieve(
    query,
    product
):

    query_tokens = tokenize(
        query
    )


    scores = bm25.get_scores(
        query_tokens
    )


    # --------------------------------------------------------
    # Search only the detected product
    # when product metadata is available.
    # --------------------------------------------------------

    if product:

        candidate_indices = (
            product_to_indices[
                product
            ]
        )

    else:

        candidate_indices = range(
            len(metadata)
        )


    ranked_indices = sorted(

        candidate_indices,

        key=lambda idx: scores[
            idx
        ],

        reverse=True
    )


    results = []


    for rank, idx in enumerate(
        ranked_indices[:BM25_K],
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
# RRF FUSION
# ============================================================

def fuse_results(
    bm25_results,
    dense_results
):

    fused = {}


    # --------------------------------------------------------
    # BM25
    # --------------------------------------------------------

    for item in bm25_results:

        idx = item[
            "index"
        ]


        if idx not in fused:

            fused[idx] = {

                "rrf_score": 0.0,

                "bm25_rank": None,

                "bm25_score": None,

                "dense_rank": None,

                "dense_score": None
            }


        fused[idx][
            "bm25_rank"
        ] = item[
            "rank"
        ]


        fused[idx][
            "bm25_score"
        ] = item[
            "score"
        ]


        fused[idx][
            "rrf_score"
        ] += (

            1.0
            /
            (
                RRF_K
                +
                item["rank"]
            )
        )


    # --------------------------------------------------------
    # DENSE
    # --------------------------------------------------------

    for item in dense_results:

        idx = item[
            "index"
        ]


        if idx not in fused:

            fused[idx] = {

                "rrf_score": 0.0,

                "bm25_rank": None,

                "bm25_score": None,

                "dense_rank": None,

                "dense_score": None
            }


        fused[idx][
            "dense_rank"
        ] = item[
            "rank"
        ]


        fused[idx][
            "dense_score"
        ] = item[
            "score"
        ]


        fused[idx][
            "rrf_score"
        ] += (

            1.0
            /
            (
                RRF_K
                +
                item["rank"]
            )
        )


    return fused


# ============================================================
# FINAL RETRIEVAL
# ============================================================

def retrieve(
    query,
    top_k=TOP_K
):

    # --------------------------------------------------------
    # STEP 1:
    # Detect enterprise product
    # --------------------------------------------------------

    product = detect_product(
        query
    )


    # --------------------------------------------------------
    # STEP 2:
    # Retrieve candidates
    # --------------------------------------------------------

    bm25_results = bm25_retrieve(
        query,
        product
    )


    dense_results = dense_retrieve(
        query,
        product
    )


    # --------------------------------------------------------
    # STEP 3:
    # RRF fusion
    # --------------------------------------------------------

    fused = fuse_results(
        bm25_results,
        dense_results
    )


    candidates = []


    # --------------------------------------------------------
    # STEP 4:
    # Metadata-aware reranking
    # --------------------------------------------------------

    for idx, scores in fused.items():

        record = metadata[
            idx
        ].copy()


        bonus = metadata_bonus(
            query,
            record
        )


        final_score = (

            scores[
                "rrf_score"
            ]

            +

            bonus
        )


        record[
            "rrf_score"
        ] = scores[
            "rrf_score"
        ]


        record[
            "metadata_bonus"
        ] = bonus


        record[
            "final_score"
        ] = final_score


        record[
            "bm25_rank"
        ] = scores[
            "bm25_rank"
        ]


        record[
            "dense_rank"
        ] = scores[
            "dense_rank"
        ]


        candidates.append(
            record
        )


    # --------------------------------------------------------
    # STEP 5:
    # Sort final candidates
    # --------------------------------------------------------

    candidates.sort(

        key=lambda x: x[
            "final_score"
        ],

        reverse=True
    )


    # --------------------------------------------------------
    # STEP 6:
    # Remove duplicate parent documents
    # --------------------------------------------------------

    results = []

    seen_documents = set()


    for candidate in candidates:

        parent = (

            candidate.get(
                "parent_doc_id"
            )

            or

            candidate.get(
                "doc_id"
            )
        )


        if parent in seen_documents:

            continue


        seen_documents.add(
            parent
        )


        results.append(
            candidate
        )


        if len(results) >= top_k:

            break


    # Add final rank
    for rank, result in enumerate(
        results,
        start=1
    ):

        result[
            "rank"
        ] = rank


    return results, product


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results,
    product
):

    print("\n" + "=" * 80)
    print("QUERY")
    print("=" * 80)

    print(query)


    print(
        "\nDetected Product:",
        product
    )


    print("\n" + "=" * 80)
    print("METADATA-AWARE HYBRID RESULTS")
    print("=" * 80)


    for result in results:

        print("\n" + "-" * 80)


        print(
            "Rank           :",
            result.get(
                "rank"
            )
        )


        print(
            "Final Score    :",
            round(
                result.get(
                    "final_score",
                    0
                ),
                5
            )
        )


        print(
            "RRF Score      :",
            round(
                result.get(
                    "rrf_score",
                    0
                ),
                5
            )
        )


        print(
            "Metadata Bonus :",
            round(
                result.get(
                    "metadata_bonus",
                    0
                ),
                5
            )
        )


        print(
            "BM25 Rank      :",
            result.get(
                "bm25_rank"
            )
        )


        print(
            "Dense Rank     :",
            result.get(
                "dense_rank"
            )
        )


        print(
            "Doc ID         :",
            result.get(
                "doc_id"
            )
        )


        print(
            "Source         :",
            result.get(
                "source"
            )
        )


        print(
            "Product        :",
            result.get(
                "product"
            )
        )


        # ----------------------------------------------------
        # AUTHOR INFORMATION
        # ----------------------------------------------------

        author = result.get(
            "author"
        )


        if author:

            print(
                "Author         :",
                author
            )


            print(
                "Author Role    :",
                get_employee_role(
                    author
                )
            )


        # ----------------------------------------------------
        # SLACK USER INFORMATION
        # ----------------------------------------------------

        user_id = result.get(
            "user_id"
        )


        if (
            user_id
            and
            user_id
            != "slack_admin_bot"
        ):

            print(
                "User ID        :",
                user_id
            )


            print(
                "User Role      :",
                get_employee_role(
                    user_id
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
        "\nDense candidates:",
        DENSE_K
    )


    print(
        "BM25 candidates :",
        BM25_K
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
                "\nExiting Metadata Retriever."
            )

            break


        if not query:

            print(
                "Please enter a question."
            )

            continue


        results, product = retrieve(
            query,
            TOP_K
        )


        display_results(
            query,
            results,
            product
        )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()

# Query
#   ↓
# Product detection → ForecastForce
#   ↓
# Hard product filtering
#   ↓
# Dense retrieval + enriched BM25
#   ↓
# Employee metadata
#   ↓
# Role = Marketing Research Analyst
#   ↓
# RRF + metadata reranking
#   ↓
# Correct documents at Rank 1 and Rank 2 ✅