import math

def recall_at_k(retrieved_docs, gold_docs):
    """
    Recall@K = Relevant Retrieved / Total Relevant Documents
    """
    retrieved_ids = {doc["doc_id"] for doc, score in retrieved_docs}
    gold_ids = set(gold_docs)

    if not gold_ids:
        return 0.0

    return len(retrieved_ids.intersection(gold_ids)) / len(gold_ids)


def precision_at_k(retrieved_docs, gold_docs):
    """
    Precision@K = Relevant Retrieved / Retrieved Documents
    """
    retrieved_ids = [doc["doc_id"] for doc, score in retrieved_docs]
    gold_ids = set(gold_docs)

    relevant = sum(1 for doc_id in retrieved_ids if doc_id in gold_ids)

    return relevant / len(retrieved_ids) if retrieved_ids else 0.0


def mrr(retrieved_docs, gold_docs):
    """
    Mean Reciprocal Rank
    Finds the position of the FIRST relevant document.
    """

    gold_ids = set(gold_docs)

    for rank, (doc, score) in enumerate(retrieved_docs, start=1):
        if doc["doc_id"] in gold_ids:
            return 1 / rank

    return 0.0


def ndcg_at_k(retrieved_docs, gold_docs):
    """
    nDCG@K
    """

    gold_ids = set(gold_docs)

    dcg = 0

    for i, (doc, score) in enumerate(retrieved_docs):
        rel = 1 if doc["doc_id"] in gold_ids else 0
        dcg += rel / math.log2(i + 2)

    ideal_rels = [1] * min(len(gold_ids), len(retrieved_docs))

    idcg = 0

    for i, rel in enumerate(ideal_rels):
        idcg += rel / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0