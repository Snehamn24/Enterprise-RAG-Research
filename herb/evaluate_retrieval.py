import csv
import gc
import importlib
import json
import math
import os
import random
import sys


# ============================================================
# PATHS
# ============================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.dirname(
    CURRENT_DIR
)

RETRIEVERS_DIR = os.path.join(
    CURRENT_DIR,
    "retrievers"
)

PRODUCTS_DIR = os.path.join(
    CURRENT_DIR,
    "data",
    "HERB",
    "products"
)

INDEX_METADATA_PATH = os.path.join(
    CURRENT_DIR,
    "data",
    "index_metadata.jsonl"
)

RESULTS_DIR = os.path.join(
    ROOT_DIR,
    "results"
)

SAMPLE_PATH = os.path.join(
    RESULTS_DIR,
    "evaluation_sample.json"
)

DETAILS_PATH = os.path.join(
    RESULTS_DIR,
    "retrieval_evaluation_details.csv"
)

SUMMARY_PATH = os.path.join(
    RESULTS_DIR,
    "retrieval_evaluation_summary.csv"
)


# Make retriever modules importable
if RETRIEVERS_DIR not in sys.path:
    sys.path.insert(
        0,
        RETRIEVERS_DIR
    )


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Start small.
# We will increase this only after the evaluator works.
SAMPLE_SIZE = 15

# Ask every retriever for more than 10 chunks.
# Then we collapse them to unique documents.
RETRIEVAL_POOL_K = 50

MAX_EVALUATION_K = 10


METHODS = [
    (
        "BM25",
        "bm25_retriever"
    ),
    (
        "Dense",
        "dense_retriever"
    ),
    (
        "Hybrid",
        "hybrid_retriever"
    ),
    (
        "Metadata",
        "metadata_retriever"
    ),
    (
        "Temporal",
        "temporal_retriever"
    )
]


# ============================================================
# OUR DEVELOPMENT QUESTIONS
# ============================================================

# IMPORTANT:
#
# We already used these questions to inspect failures and
# modify the retrieval algorithms.
#
# Therefore they should NOT be part of our new held-out
# evaluation sample.

DEV_QUESTIONS = {

    (
        "Find employee IDs of Marketing Research Analysts "
        "who worked on the previous release of ForecastForce?"
    ),

    (
        "Find employee IDs of Product Managers "
        "who worked on the previous release of ForecastForce?"
    ),

    (
        "What strengths were highlighted by customers "
        "for the previous release of ForecastForce?"
    )
}


# ============================================================
# TEMPORAL LANGUAGE
# ============================================================

TEMPORAL_MARKERS = [

    "previous release",
    "last release",

    "previous version",
    "last version",

    "earlier release",
    "prior release",

    "latest release",
    "latest version",

    "current release",
    "current version",

    "before the release",
    "after the release"
]


# ============================================================
# QUESTION CLASSIFICATION
# ============================================================

def classify_question(
    question,
    question_type
):
    """
    Divide questions into useful evaluation groups.

    Examples:

    temporal_person
    temporal_content
    person
    content
    """

    text = question.lower()

    is_temporal = any(
        marker in text
        for marker in TEMPORAL_MARKERS
    )

    is_person = (
        str(
            question_type
        ).lower()
        ==
        "person"
    )


    if is_temporal and is_person:

        return "temporal_person"


    if is_temporal:

        return "temporal_content"


    if is_person:

        return "person"


    return "content"


# ============================================================
# LOAD HERB QUESTIONS
# ============================================================

def load_all_questions():

    questions = []


    for filename in sorted(
        os.listdir(
            PRODUCTS_DIR
        )
    ):

        if not filename.endswith(
            ".json"
        ):
            continue


        path = os.path.join(
            PRODUCTS_DIR,
            filename
        )


        with open(
            path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        product = os.path.splitext(
            filename
        )[0]


        answerable = data.get(
            "answerable_questions",
            []
        )


        for item in answerable:

            question = str(
                item.get(
                    "question",
                    ""
                )
            ).strip()


            if not question:
                continue


            # -----------------------------------------------
            # Remove our three development questions
            # -----------------------------------------------

            if question in DEV_QUESTIONS:
                continue


            citations = item.get(
                "citations",
                []
            )


            if not citations:
                continue


            # Remove duplicate citations while preserving order
            citations = list(
                dict.fromkeys(
                    citations
                )
            )


            question_type = item.get(
                "type",
                "unknown"
            )


            category = classify_question(
                question,
                question_type
            )


            questions.append({

                "product":
                    product,

                "question":
                    question,

                "type":
                    question_type,

                "category":
                    category,

                "citations":
                    citations,

                "ground_truth":
                    item.get(
                        "ground_truth",
                        []
                    )
            })


    return questions


# ============================================================
# DIVERSE SAMPLING
# ============================================================

def pick_diverse(
    items,
    amount,
    rng
):
    """
    Select questions while trying to avoid taking
    everything from only one product.
    """

    groups = {}


    for item in items:

        product = item[
            "product"
        ]

        groups.setdefault(
            product,
            []
        ).append(
            item
        )


    product_names = list(
        groups.keys()
    )


    rng.shuffle(
        product_names
    )


    for product in product_names:

        rng.shuffle(
            groups[
                product
            ]
        )


    selected = []


    while len(selected) < amount:

        added_any = False


        for product in product_names:

            if len(selected) >= amount:
                break


            if groups[
                product
            ]:

                selected.append(
                    groups[
                        product
                    ].pop()
                )

                added_any = True


        if not added_any:
            break


    return selected


# ============================================================
# CREATE HELD-OUT SAMPLE
# ============================================================

def create_evaluation_sample(
    questions
):

    rng = random.Random(
        RANDOM_SEED
    )


    buckets = {

        "temporal_person": [],

        "temporal_content": [],

        "person": [],

        "content": []
    }


    for item in questions:

        category = item[
            "category"
        ]

        buckets[
            category
        ].append(
            item
        )


    # --------------------------------------------------------
    # 15-question initial evaluation
    #
    # 4 temporal person
    # 4 temporal content
    # 4 person
    # 3 normal content
    # --------------------------------------------------------

    quotas = {

        "temporal_person": 4,

        "temporal_content": 4,

        "person": 4,

        "content": 3
    }


    sample = []


    for category, amount in quotas.items():

        selected = pick_diverse(

            buckets[
                category
            ],

            amount,

            rng
        )


        sample.extend(
            selected
        )


    # --------------------------------------------------------
    # If one category was too small,
    # fill remaining slots using unused questions.
    # --------------------------------------------------------

    selected_keys = {

        (
            item["product"],
            item["question"]
        )

        for item in sample
    }


    remaining = [

        item

        for item in questions

        if (
            item["product"],
            item["question"]
        )
        not in selected_keys
    ]


    rng.shuffle(
        remaining
    )


    while (
        len(sample) < SAMPLE_SIZE
        and
        remaining
    ):

        sample.append(
            remaining.pop()
        )


    # Randomize final order,
    # but reproducibly because seed is fixed.
    rng.shuffle(
        sample
    )


    return sample[
        :SAMPLE_SIZE
    ]


# ============================================================
# LOAD OR CREATE SAMPLE
# ============================================================

def get_evaluation_sample():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True
    )


    # --------------------------------------------------------
    # Reuse existing sample if it exists.
    #
    # This is important.
    #
    # Every retrieval method must be tested against
    # exactly the same questions.
    # --------------------------------------------------------

    if os.path.exists(
        SAMPLE_PATH
    ):

        with open(
            SAMPLE_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            sample = json.load(
                file
            )


        print(
            "\nUsing existing evaluation sample:"
        )

        print(
            SAMPLE_PATH
        )

        print(
            "Questions:",
            len(sample)
        )


        return sample


    all_questions = (
        load_all_questions()
    )


    print(
        "\nAnswerable HERB questions available:",
        len(all_questions)
    )


    sample = (
        create_evaluation_sample(
            all_questions
        )
    )


    with open(
        SAMPLE_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            sample,
            file,
            indent=2,
            ensure_ascii=False
        )


    print(
        "\nNew held-out evaluation sample saved:"
    )

    print(
        SAMPLE_PATH
    )

    print(
        "Questions:",
        len(sample)
    )


    return sample


# ============================================================
# LOAD INDEX-REACHABLE IDS
# ============================================================

def load_index_ids():
    """
    Build a set containing every document ID that our
    current retrieval index can actually return.

    This is important because some benchmark citations
    may not exist in our processed retrieval corpus.
    """

    ids = set()


    with open(
        INDEX_METADATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            record = json.loads(
                line
            )


            doc_id = record.get(
                "doc_id"
            )


            parent_doc_id = record.get(
                "parent_doc_id"
            )


            if doc_id:

                ids.add(
                    str(
                        doc_id
                    )
                )


            if parent_doc_id:

                ids.add(
                    str(
                        parent_doc_id
                    )
                )


    return ids


# ============================================================
# HANDLE DIFFERENT RETRIEVER RETURN FORMATS
# ============================================================

def extract_results(
    output
):
    """
    BM25:
        results

    Dense:
        results

    Hybrid:
        results

    Metadata:
        results, product

    Temporal:
        results, product, temporal, anchor

    We only need the first part: results.
    """

    if isinstance(
        output,
        tuple
    ):

        return output[0]


    return output


# ============================================================
# COLLAPSE CHUNKS INTO UNIQUE DOCUMENTS
# ============================================================

def collapse_to_documents(
    results,
    max_k=MAX_EVALUATION_K
):
    """
    BM25/Dense may return several chunks from the same
    parent document.

    Retrieval evaluation is performed at document level,
    because HERB citations are document/message IDs.
    """

    documents = []

    seen = set()


    for result in results:

        parent_doc_id = result.get(
            "parent_doc_id"
        )

        doc_id = result.get(
            "doc_id"
        )

        chunk_id = result.get(
            "chunk_id"
        )


        primary_id = (

            parent_doc_id

            or

            doc_id

            or

            chunk_id
        )


        if primary_id is None:
            continue


        primary_id = str(
            primary_id
        )


        if primary_id in seen:
            continue


        seen.add(
            primary_id
        )


        aliases = set()


        if parent_doc_id:

            aliases.add(
                str(
                    parent_doc_id
                )
            )


        if doc_id:

            aliases.add(
                str(
                    doc_id
                )
            )


        if chunk_id:

            aliases.add(
                str(
                    chunk_id
                )
            )


        documents.append({

            "primary_id":
                primary_id,

            "aliases":
                aliases
        })


        if len(documents) >= max_k:
            break


    return documents


# ============================================================
# RETRIEVAL RELEVANCE
# ============================================================

def is_relevant(
    document,
    gold
):

    return bool(

        document[
            "aliases"
        ]

        &

        gold
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    documents,
    reachable_gold
):

    gold = set(
        reachable_gold
    )


    # Cannot fairly score if none of the official
    # citations exist in our current index.
    if not gold:

        return {

            "hit5": None,

            "precision5": None,

            "recall5": None,

            "mrr10": None,

            "ndcg5": None,

            "recall10": None,

            "ndcg10": None
        }


    # ========================================================
    # HIT@5
    # ========================================================

    top5 = documents[:5]


    hit5 = 1.0 if any(

        is_relevant(
            document,
            gold
        )

        for document in top5

    ) else 0.0


    # ========================================================
    # PRECISION@5
    # ========================================================

    relevant_top5 = sum(

        1

        for document in top5

        if is_relevant(
            document,
            gold
        )
    )


    precision_denominator = max(
        1,
        len(top5)
    )


    precision5 = (

        relevant_top5

        /

        precision_denominator
    )


    # ========================================================
    # RECALL@5
    # ========================================================

    found_gold_5 = set()


    for document in top5:

        found_gold_5.update(

            document[
                "aliases"
            ]

            &

            gold
        )


    recall5 = (

        len(
            found_gold_5
        )

        /

        len(
            gold
        )
    )


    # ========================================================
    # RECALL@10
    # ========================================================

    top10 = documents[:10]

    found_gold_10 = set()


    for document in top10:

        found_gold_10.update(

            document[
                "aliases"
            ]

            &

            gold
        )


    recall10 = (

        len(
            found_gold_10
        )

        /

        len(
            gold
        )
    )


    # ========================================================
    # MRR@10
    # ========================================================

    mrr10 = 0.0


    for rank, document in enumerate(
        top10,
        start=1
    ):

        if is_relevant(
            document,
            gold
        ):

            mrr10 = (
                1.0
                /
                rank
            )

            break


    # ========================================================
    # NDCG
    # ========================================================

    def ndcg_at_k(k):

        selected = documents[:k]


        dcg = 0.0


        for rank, document in enumerate(
            selected,
            start=1
        ):

            relevance = (

                1.0

                if is_relevant(
                    document,
                    gold
                )

                else 0.0
            )


            dcg += (

                relevance

                /

                math.log2(
                    rank + 1
                )
            )


        ideal_relevant = min(
            len(gold),
            k
        )


        idcg = sum(

            1.0

            /

            math.log2(
                rank + 1
            )

            for rank in range(
                1,
                ideal_relevant + 1
            )
        )


        if idcg == 0:

            return 0.0


        return (
            dcg
            /
            idcg
        )


    return {

        "hit5":
            hit5,

        "precision5":
            precision5,

        "recall5":
            recall5,

        "mrr10":
            mrr10,

        "ndcg5":
            ndcg_at_k(5),

        "recall10":
            recall10,

        "ndcg10":
            ndcg_at_k(10)
    }


# ============================================================
# CLEAN MODULE FROM MEMORY
# ============================================================

def cleanup_module(
    module_name
):

    if module_name in sys.modules:

        del sys.modules[
            module_name
        ]


    # Temporal imports metadata_retriever internally.
    if (
        module_name
        ==
        "temporal_retriever"
    ):

        if (
            "metadata_retriever"
            in sys.modules
        ):

            del sys.modules[
                "metadata_retriever"
            ]


    gc.collect()


# ============================================================
# EVALUATE ONE RETRIEVER
# ============================================================

def evaluate_method(
    method_name,
    module_name,
    sample,
    index_ids
):

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "EVALUATING:",
        method_name
    )

    print(
        "=" * 80
    )


    # Import only one retriever at a time.
    module = importlib.import_module(
        module_name
    )


    rows = []


    for question_number, item in enumerate(
        sample,
        start=1
    ):

        question = item[
            "question"
        ]


        print(
            f"\n[{question_number}/{len(sample)}]",
            item["product"],
            "|",
            item["category"]
        )

        print(
            question
        )


        citations = [

            str(
                citation
            )

            for citation in item[
                "citations"
            ]
        ]


        reachable = [

            citation

            for citation in citations

            if citation in index_ids
        ]


        citation_coverage = (

            len(
                reachable
            )

            /

            len(
                citations
            )

            if citations

            else 0.0
        )


        error = ""


        try:

            # Ask for a larger result pool.
            #
            # BM25 and Dense may return multiple chunks
            # belonging to the same document.
            output = module.retrieve(

                question,

                RETRIEVAL_POOL_K
            )


            raw_results = (
                extract_results(
                    output
                )
            )


            documents = (
                collapse_to_documents(
                    raw_results,
                    MAX_EVALUATION_K
                )
            )


            metrics = (
                calculate_metrics(
                    documents,
                    reachable
                )
            )


            retrieved_ids = [

                document[
                    "primary_id"
                ]

                for document in documents
            ]


        except Exception as exc:

            print(
                "ERROR:",
                exc
            )


            error = str(
                exc
            )


            documents = []

            retrieved_ids = []


            metrics = {

                "hit5": None,

                "precision5": None,

                "recall5": None,

                "mrr10": None,

                "ndcg5": None,

                "recall10": None,

                "ndcg10": None
            }


        scorable = (

            len(
                reachable
            ) > 0

            and

            not error
        )


        row = {

            "method":
                method_name,

            "question_number":
                question_number,

            "product":
                item[
                    "product"
                ],

            "category":
                item[
                    "category"
                ],

            "type":
                item[
                    "type"
                ],

            "question":
                question,

            "ground_truth":
                json.dumps(
                    item[
                        "ground_truth"
                    ],
                    ensure_ascii=False
                ),

            "gold_citations":
                json.dumps(
                    citations,
                    ensure_ascii=False
                ),

            "reachable_citations":
                json.dumps(
                    reachable,
                    ensure_ascii=False
                ),

            "citation_count":
                len(
                    citations
                ),

            "reachable_count":
                len(
                    reachable
                ),

            "citation_coverage":
                citation_coverage,

            "retrieved_top10":
                json.dumps(
                    retrieved_ids,
                    ensure_ascii=False
                ),

            "scorable":
                int(
                    scorable
                ),

            "hit5":
                metrics[
                    "hit5"
                ],

            "precision5":
                metrics[
                    "precision5"
                ],

            "recall5":
                metrics[
                    "recall5"
                ],

            "mrr10":
                metrics[
                    "mrr10"
                ],

            "ndcg5":
                metrics[
                    "ndcg5"
                ],

            "recall10":
                metrics[
                    "recall10"
                ],

            "ndcg10":
                metrics[
                    "ndcg10"
                ],

            "error":
                error
        }


        rows.append(
            row
        )


        if scorable:

            print(

                "Hit@5:",
                round(
                    metrics[
                        "hit5"
                    ],
                    3
                ),

                "| Recall@5:",
                round(
                    metrics[
                        "recall5"
                    ],
                    3
                ),

                "| MRR@10:",
                round(
                    metrics[
                        "mrr10"
                    ],
                    3
                )
            )


        else:

            print(
                "Question could not be scored."
            )


    # Remove retriever from memory before
    # loading the next method.
    del module


    cleanup_module(
        module_name
    )


    return rows


# ============================================================
# AVERAGE
# ============================================================

def average_metric(
    rows,
    field
):

    values = [

        row[
            field
        ]

        for row in rows

        if (
            row[
                "scorable"
            ]
            ==
            1
            and
            row[
                field
            ]
            is not None
        )
    ]


    if not values:

        return None


    return (

        sum(values)

        /

        len(values)
    )


# ============================================================
# BUILD SUMMARY
# ============================================================

def build_summary(
    all_rows
):

    summary = []


    categories = [

        "ALL",

        "temporal_person",

        "temporal_content",

        "person",

        "content"
    ]


    method_names = [

        method_name

        for (
            method_name,
            _
        )
        in METHODS
    ]


    for method in method_names:

        method_rows = [

            row

            for row in all_rows

            if row[
                "method"
            ]
            ==
            method
        ]


        for category in categories:


            if category == "ALL":

                selected = (
                    method_rows
                )


            else:

                selected = [

                    row

                    for row
                    in method_rows

                    if row[
                        "category"
                    ]
                    ==
                    category
                ]


            if not selected:
                continue


            scorable_count = sum(

                row[
                    "scorable"
                ]

                for row in selected
            )


            error_count = sum(

                1

                for row in selected

                if row[
                    "error"
                ]
            )


            summary.append({

                "method":
                    method,

                "group":
                    category,

                "questions":
                    len(
                        selected
                    ),

                "scorable":
                    scorable_count,

                "errors":
                    error_count,

                "citation_coverage":
                    average_metric(
                        selected,
                        "citation_coverage"
                    ),

                "hit5":
                    average_metric(
                        selected,
                        "hit5"
                    ),

                "precision5":
                    average_metric(
                        selected,
                        "precision5"
                    ),

                "recall5":
                    average_metric(
                        selected,
                        "recall5"
                    ),

                "mrr10":
                    average_metric(
                        selected,
                        "mrr10"
                    ),

                "ndcg5":
                    average_metric(
                        selected,
                        "ndcg5"
                    ),

                "recall10":
                    average_metric(
                        selected,
                        "recall10"
                    ),

                "ndcg10":
                    average_metric(
                        selected,
                        "ndcg10"
                    )
            })


    return summary


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(
    path,
    rows
):

    if not rows:
        return


    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(

            file,

            fieldnames=list(
                rows[0].keys()
            )
        )


        writer.writeheader()

        writer.writerows(
            rows
        )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_summary(
    summary
):

    print(
        "\n"
        +
        "=" * 100
    )

    print(
        "OVERALL RETRIEVAL EVALUATION"
    )

    print(
        "=" * 100
    )


    print(

        f"{'Method':<12}"
        f"{'Hit@5':>10}"
        f"{'Recall@5':>12}"
        f"{'MRR@10':>12}"
        f"{'nDCG@5':>12}"
        f"{'Recall@10':>12}"
        f"{'nDCG@10':>12}"
    )


    print(
        "-" * 100
    )


    for row in summary:

        if row[
            "group"
        ] != "ALL":

            continue


        def value(field):

            number = row[
                field
            ]


            if number is None:

                return "N/A"


            return f"{number:.4f}"


        print(

            f"{row['method']:<12}"
            f"{value('hit5'):>10}"
            f"{value('recall5'):>12}"
            f"{value('mrr10'):>12}"
            f"{value('ndcg5'):>12}"
            f"{value('recall10'):>12}"
            f"{value('ndcg10'):>12}"
        )


# ============================================================
# SHOW SAMPLE
# ============================================================

def display_sample(
    sample
):

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "HELD-OUT QUESTIONS"
    )

    print(
        "=" * 80
    )


    for number, item in enumerate(
        sample,
        start=1
    ):

        print(
            f"\n{number}. "
            f"[{item['category']}] "
            f"[{item['product']}]"
        )

        print(
            item[
                "question"
            ]
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        +
        "=" * 80
    )

    print(
        "HERB RETRIEVAL EVALUATION"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # STEP 1
    # Load fixed held-out questions
    # ========================================================

    sample = (
        get_evaluation_sample()
    )


    display_sample(
        sample
    )


    # ========================================================
    # STEP 2
    # Determine which benchmark citations
    # actually exist in our retrieval index.
    # ========================================================

    print(
        "\nLoading index document IDs..."
    )


    index_ids = (
        load_index_ids()
    )


    print(
        "Reachable document IDs:",
        len(index_ids)
    )


    # ========================================================
    # STEP 3
    # Evaluate every retrieval method
    # ========================================================

    all_rows = []


    for (
        method_name,
        module_name
    ) in METHODS:


        method_rows = (
            evaluate_method(

                method_name,

                module_name,

                sample,

                index_ids
            )
        )


        all_rows.extend(
            method_rows
        )


        # Save partial progress after every method.
        #
        # If something fails later,
        # we still keep earlier results.
        save_csv(
            DETAILS_PATH,
            all_rows
        )


    # ========================================================
    # STEP 4
    # Build research summary
    # ========================================================

    summary = (
        build_summary(
            all_rows
        )
    )


    save_csv(
        SUMMARY_PATH,
        summary
    )


    # ========================================================
    # STEP 5
    # Display final table
    # ========================================================

    print_summary(
        summary
    )


    print(
        "\nDetailed results saved to:"
    )

    print(
        DETAILS_PATH
    )


    print(
        "\nSummary saved to:"
    )

    print(
        SUMMARY_PATH
    )


    print(
        "\nHeld-out question set:"
    )

    print(
        SAMPLE_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()