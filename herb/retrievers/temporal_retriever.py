import re
from datetime import datetime

# Reuse our working metadata-aware retriever
import metadata_retriever as base


# ============================================================
# CONFIGURATION
# ============================================================

TOP_K = 5

# Retrieve a large initial pool
TEMPORAL_BM25_K = 500
TEMPORAL_DENSE_K = 500

# Records around the release anchor are also considered
TEMPORAL_WINDOW_DAYS = 90


# ============================================================
# TEMPORAL PHRASES
# ============================================================

TEMPORAL_PHRASES = [
    "previous release",
    "last release",
    "previous version",
    "last version",
    "earlier release",
    "prior release"
]


# ============================================================
# TEMPORAL QUERY DETECTION
# ============================================================

def is_temporal_query(query):
    """
    Detect whether the query refers to historical
    or previous-release information.
    """

    query_text = base.normalize_text(
        query
    )

    return any(
        phrase in query_text
        for phrase in TEMPORAL_PHRASES
    )


# ============================================================
# DATE EXTRACTION
# ============================================================

def get_record_datetime(record):
    """
    HERB records may store time as:

    timestamp
    or
    date

    Convert the available value into datetime.
    """

    value = (
        record.get("timestamp")
        or
        record.get("date")
    )

    if not value:
        return None

    try:

        return datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00"
            )
        )

    except ValueError:

        return None


# ============================================================
# RELEASE ANCHOR DETECTION
# ============================================================

def find_release_anchor(product):
    """
    Find an enterprise record that explicitly
    talks about the previous / last release.

    Example:

    "Let's start by reviewing the documents
    from the last release."

    Its date becomes our historical anchor.
    """

    if not product:
        return None


    possible_anchors = []


    for idx in base.product_to_indices[
        product
    ]:

        record = base.metadata[
            idx
        ]


        text = base.normalize_text(
            record.get(
                "text",
                ""
            )
        )


        # Does this record explicitly mention
        # previous / last release?
        matched = any(
            phrase in text
            for phrase in TEMPORAL_PHRASES
        )


        if not matched:
            continue


        record_date = get_record_datetime(
            record
        )


        if record_date is None:
            continue


        possible_anchors.append(
            (
                record_date,
                idx
            )
        )


    if not possible_anchors:
        return None


    # Sort by date
    possible_anchors.sort(
        key=lambda item: item[0]
    )


    # For our HERB experiment,
    # choose the earliest strong release reference.
    anchor_date, anchor_idx = (
        possible_anchors[0]
    )


    return {

        "index": anchor_idx,

        "date": anchor_date,

        "record": base.metadata[
            anchor_idx
        ]
    }


# ============================================================
# TEMPORAL WINDOW CANDIDATES
# ============================================================

def get_temporal_window_candidates(
    product,
    anchor,
    window_days=TEMPORAL_WINDOW_DAYS
):
    """
    Collect all records from the detected product
    that occur near the historical release anchor.

    This prevents useful historical evidence from
    disappearing just because BM25 or Dense ranked
    it below their Top-K candidate lists.
    """

    if not product:
        return []


    if anchor is None:
        return []


    anchor_date = anchor[
        "date"
    ]


    candidates = []


    for idx in base.product_to_indices[
        product
    ]:

        record = base.metadata[
            idx
        ]


        record_date = get_record_datetime(
            record
        )


        if record_date is None:
            continue


        difference = abs(
            (
                record_date
                -
                anchor_date
            ).days
        )


        if difference <= window_days:

            candidates.append(
                idx
            )


    return candidates


# ============================================================
# TEMPORAL BONUS
# ============================================================

def calculate_temporal_bonus(
    record,
    anchor
):
    """
    Reward evidence close to the release anchor.

    Penalize records that are far away.
    """

    if anchor is None:
        return 0.0


    record_date = get_record_datetime(
        record
    )


    if record_date is None:
        return 0.0


    anchor_date = anchor[
        "date"
    ]


    difference = abs(
        (
            record_date
            -
            anchor_date
        ).days
    )


    # Very close to release
    if difference <= 30:
        return 0.035


    # Still strongly relevant
    if difference <= 60:
        return 0.030


    # Moderately close
    if difference <= 90:
        return 0.020


    # Somewhat related
    if difference <= 180:
        return 0.005


    # Very far from requested historical release
    return -0.030


# ============================================================
# RELEASE TEXT BONUS
# ============================================================

def calculate_release_text_bonus(
    record
):
    """
    Reward records that explicitly mention:

    previous release
    last release
    previous version
    etc.
    """

    text = base.normalize_text(
        record.get(
            "text",
            ""
        )
    )


    if any(
        phrase in text
        for phrase in TEMPORAL_PHRASES
    ):

        return 0.020


    return 0.0


# ============================================================
# STRENGTH QUERY DETECTION
# ============================================================

def asks_for_strengths(query):
    """
    Detect whether user wants positive characteristics,
    benefits or strengths.
    """

    query_text = base.normalize_text(
        query
    )


    terms = [
        "strength",
        "strengths",
        "positive feedback",
        "benefit",
        "benefits",
        "advantage",
        "advantages",
        "appreciated"
    ]


    return any(
        term in query_text
        for term in terms
    )


# ============================================================
# CUSTOMER-STRENGTH QUERY DETECTION
# ============================================================

def asks_for_customer_strengths(query):
    """
    Detect questions specifically asking about
    positive customer feedback.
    """

    query_text = base.normalize_text(
        query
    )


    return (
        "customer" in query_text
        and
        asks_for_strengths(query)
    )


# ============================================================
# CUSTOMER EVIDENCE DETECTION
# ============================================================

def contains_customer_evidence(
    record
):
    """
    Determine whether the record actually contains
    customer-related evidence.

    HERB customer IDs look like:

    CUST-0092
    CUST-0005
    """

    raw_text = str(
        record.get(
            "text",
            ""
        )
    )


    normalized_text = (
        base.normalize_text(
            raw_text
        )
    )


    # HERB customer ID
    if re.search(
        r"\bCUST-\d+\b",
        raw_text,
        re.IGNORECASE
    ):

        return True


    # General customer language
    if "customer" in normalized_text:

        return True


    return False


# ============================================================
# CUSTOMER FEEDBACK BONUS
# ============================================================

def calculate_feedback_bonus(
    query,
    record
):
    """
    For customer-strength questions:

    1. Require actual customer evidence.
    2. Reward positive-feedback language.
    3. Penalize issue / complaint language.

    This is a lightweight query-aware heuristic,
    not a sentiment-analysis model.
    """

    if not asks_for_strengths(
        query
    ):

        return 0.0


    text = base.normalize_text(
        record.get(
            "text",
            ""
        )
    )


    customer_evidence = (
        contains_customer_evidence(
            record
        )
    )


    # --------------------------------------------------------
    # IMPORTANT
    #
    # If user specifically asks for CUSTOMER strengths,
    # generic product documents containing words such as
    # "reliable" or "accuracy" should not rank highly.
    # --------------------------------------------------------

    if (
        asks_for_customer_strengths(
            query
        )
        and
        not customer_evidence
    ):

        return -0.050


    bonus = 0.0


    # --------------------------------------------------------
    # ACTUAL CUSTOMER EVIDENCE
    # --------------------------------------------------------

    if customer_evidence:

        bonus += 0.030


    # --------------------------------------------------------
    # POSITIVE TERMS
    # --------------------------------------------------------

    positive_terms = [

        "accurate",

        "accuracy",

        "reliable",

        "reliability",

        "seamless",

        "efficient",

        "efficiency",

        "good performance",

        "highly reliable",

        "comprehensive",

        "appreciated",

        "positive feedback",

        "minimal downtime",

        "easy to use"
    ]


    positive_matches = sum(

        1

        for term in positive_terms

        if term in text
    )


    bonus += min(

        positive_matches
        * 0.012,

        0.048
    )


    # --------------------------------------------------------
    # NEGATIVE TERMS
    # --------------------------------------------------------

    negative_terms = [

        "issue",

        "issues",

        "bug",

        "bugs",

        "problem",

        "problems",

        "complaint",

        "complaints",

        "need our attention",

        "reported issues",

        "timeout",

        "failure"
    ]


    negative_matches = sum(

        1

        for term in negative_terms

        if term in text
    )


    if negative_matches > 0:

        bonus -= min(

            negative_matches
            * 0.030,

            0.060
        )


    return bonus


# ============================================================
# MAIN RETRIEVAL FUNCTION
# ============================================================

def retrieve(
    query,
    top_k=TOP_K
):

    # ========================================================
    # STEP 1
    # Detect product
    # ========================================================

    product = base.detect_product(
        query
    )


    # ========================================================
    # STEP 2
    # Detect temporal query
    # ========================================================

    temporal = is_temporal_query(
        query
    )


    # ========================================================
    # STEP 3
    # Find release anchor
    # ========================================================

    anchor = None


    if temporal:

        anchor = find_release_anchor(
            product
        )


    # ========================================================
    # STEP 4
    # Temporarily increase candidate pool
    # ========================================================

    original_bm25_k = base.BM25_K

    original_dense_k = base.DENSE_K


    base.BM25_K = (
        TEMPORAL_BM25_K
    )

    base.DENSE_K = (
        TEMPORAL_DENSE_K
    )


    # ========================================================
    # STEP 5
    # BM25 retrieval
    # ========================================================

    bm25_results = (
        base.bm25_retrieve(
            query,
            product
        )
    )


    # ========================================================
    # STEP 6
    # Dense retrieval
    # ========================================================

    dense_results = (
        base.dense_retrieve(
            query,
            product
        )
    )


    # Restore original configuration
    base.BM25_K = (
        original_bm25_k
    )

    base.DENSE_K = (
        original_dense_k
    )


    # ========================================================
    # STEP 7
    # BM25 + Dense RRF Fusion
    # ========================================================

    fused = base.fuse_results(

        bm25_results,

        dense_results
    )


    # ========================================================
    # STEP 8
    # TEMPORAL CANDIDATE EXPANSION
    # ========================================================

    if (
        temporal
        and
        anchor is not None
    ):

        temporal_indices = (
            get_temporal_window_candidates(

                product,

                anchor
            )
        )


        for idx in temporal_indices:


            # Candidate may not appear in either
            # BM25 Top-500 or Dense Top-500.
            if idx not in fused:

                fused[idx] = {

                    "rrf_score": 0.0,

                    "bm25_rank": None,

                    "bm25_score": None,

                    "dense_rank": None,

                    "dense_score": None
                }


    # ========================================================
    # STEP 9
    # SCORE EVERY CANDIDATE
    # ========================================================

    candidates = []


    for idx, scores in fused.items():

        record = base.metadata[
            idx
        ].copy()


        # ----------------------------------------------------
        # Existing enterprise metadata score
        # ----------------------------------------------------

        metadata_score = (
            base.metadata_bonus(
                query,
                record
            )
        )


        # ----------------------------------------------------
        # Temporal proximity score
        # ----------------------------------------------------

        temporal_score = 0.0


        if temporal:

            temporal_score = (
                calculate_temporal_bonus(

                    record,

                    anchor
                )
            )


        # ----------------------------------------------------
        # Explicit release-language score
        # ----------------------------------------------------

        release_score = 0.0


        if temporal:

            release_score = (
                calculate_release_text_bonus(
                    record
                )
            )


        # ----------------------------------------------------
        # Customer / positive feedback score
        # ----------------------------------------------------

        feedback_score = (
            calculate_feedback_bonus(
                query,
                record
            )
        )


        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        final_score = (

            scores[
                "rrf_score"
            ]

            +

            metadata_score

            +

            temporal_score

            +

            release_score

            +

            feedback_score
        )


        # Save diagnostic information
        record.update({

            "rrf_score":
                float(
                    scores[
                        "rrf_score"
                    ]
                ),

            "metadata_bonus":
                float(
                    metadata_score
                ),

            "temporal_bonus":
                float(
                    temporal_score
                ),

            "release_bonus":
                float(
                    release_score
                ),

            "feedback_bonus":
                float(
                    feedback_score
                ),

            "final_score":
                float(
                    final_score
                ),

            "bm25_rank":
                scores[
                    "bm25_rank"
                ],

            "dense_rank":
                scores[
                    "dense_rank"
                ]
        })


        candidates.append(
            record
        )


    # ========================================================
    # STEP 10
    # SORT BY FINAL SCORE
    # ========================================================

    candidates.sort(

        key=lambda candidate:
            candidate[
                "final_score"
            ],

        reverse=True
    )


    # ========================================================
    # STEP 11
    # CUSTOMER-STRENGTH QUERY FOCUS
    # ========================================================

    if asks_for_customer_strengths(
        query
    ):

        focused_candidates = []

        other_candidates = []


        for candidate in candidates:


            if (
                contains_customer_evidence(
                    candidate
                )

                and

                candidate.get(
                    "feedback_bonus",
                    0
                ) > 0.030
            ):

                focused_candidates.append(
                    candidate
                )


            else:

                other_candidates.append(
                    candidate
                )


        # Positive customer evidence first.
        #
        # Everything else remains as fallback.
        candidates = (

            focused_candidates

            +

            other_candidates
        )


    # ========================================================
    # STEP 12
    # DOCUMENT-LEVEL DEDUPLICATION
    # ========================================================

    results = []

    seen_documents = set()


    for candidate in candidates:


        document_key = (

            candidate.get(
                "parent_doc_id"
            )

            or

            candidate.get(
                "doc_id"
            )

            or

            candidate.get(
                "chunk_id"
            )
        )


        if (
            document_key
            in seen_documents
        ):

            continue


        seen_documents.add(
            document_key
        )


        results.append(
            candidate
        )


        if len(results) >= top_k:

            break


    # ========================================================
    # STEP 13
    # ADD FINAL RANK
    # ========================================================

    for rank, result in enumerate(
        results,
        start=1
    ):

        result[
            "rank"
        ] = rank


    return (
        results,
        product,
        temporal,
        anchor
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    query,
    results,
    product,
    temporal,
    anchor
):

    print(
        "\n"
        +
        "=" * 80
    )

    print("QUERY")

    print(
        "=" * 80
    )


    print(query)


    print(
        "\nDetected Product:",
        product
    )


    print(
        "Temporal Query :",
        temporal
    )


    # ========================================================
    # DISPLAY RELEASE ANCHOR
    # ========================================================

    if anchor:

        print(
            "Release Anchor Date:",
            anchor[
                "date"
            ]
        )


        print(
            "Release Anchor Doc :",
            anchor[
                "record"
            ].get(
                "doc_id"
            )
        )


        print(
            "Anchor Text:"
        )


        print(
            anchor[
                "record"
            ].get(
                "text",
                ""
            )
        )


    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "TEMPORAL-AWARE HYBRID RESULTS"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # DISPLAY EACH RESULT
    # ========================================================

    for result in results:

        print(
            "\n"
            +
            "-" * 80
        )


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
            "Temporal Bonus :",
            round(
                result.get(
                    "temporal_bonus",
                    0
                ),
                5
            )
        )


        print(
            "Release Bonus  :",
            round(
                result.get(
                    "release_bonus",
                    0
                ),
                5
            )
        )


        print(
            "Feedback Bonus :",
            round(
                result.get(
                    "feedback_bonus",
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


        # ====================================================
        # AUTHOR
        # ====================================================

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
                base.get_employee_role(
                    author
                )
            )


        # ====================================================
        # SLACK USER
        # ====================================================

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
                base.get_employee_role(
                    user_id
                )
            )


        # ====================================================
        # CUSTOMER EVIDENCE
        # ====================================================

        print(
            "Customer Evidence:",
            contains_customer_evidence(
                result
            )
        )


        print(
            "\nTEXT:"
        )


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
        "\n"
        +
        "=" * 70
    )

    print(
        "TEMPORAL-AWARE HYBRID RETRIEVER"
    )

    print(
        "=" * 70
    )


    while True:


        query = input(

            "\nEnter your question "
            "(or type 'exit' to quit): "

        ).strip()


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if query.lower() == "exit":

            print(
                "\nExiting Temporal Retriever."
            )

            break


        # ----------------------------------------------------
        # EMPTY QUERY
        # ----------------------------------------------------

        if not query:

            print(
                "Please enter a question."
            )

            continue


        # ----------------------------------------------------
        # RETRIEVE
        # ----------------------------------------------------

        (
            results,
            product,
            temporal,
            anchor
        ) = retrieve(

            query,

            TOP_K
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        display_results(

            query,

            results,

            product,

            temporal,

            anchor
        )


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":

    main()