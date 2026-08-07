import json

from src.bm25_retriever import BM25Retriever
from src.dense_retriever import DenseRetriever
from src.hybrid_retriever import HybridRetriever
from src.evaluate import (
    recall_at_k,
    precision_at_k,
    mrr,
    ndcg_at_k
)

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            data.append(json.loads(line))
    return data


documents = load_jsonl("data/documents.jsonl")
questions = load_jsonl("data/questions.jsonl")

bm25 = BM25Retriever(documents)
dense = DenseRetriever(documents)
hybrid = HybridRetriever(bm25, dense)

retrievers = {
    "BM25": bm25,
    "Dense": dense,
    "Hybrid": hybrid
}

for name, retriever in retrievers.items():
    total_recall = 0
    total_precision = 0
    total_mrr = 0
    total_ndcg = 0

    print(f"\n===== {name} =====")

    for q in questions:
        print("\nQuestion:", q["question"])
        print("Gold Docs:", q["gold_docs"])

        results = retriever.retrieve(q["question"], top_k=5)

        recall = recall_at_k(results, q["gold_docs"])
        precision = precision_at_k(results, q["gold_docs"])
        reciprocal_rank = mrr(results, q["gold_docs"])
        ndcg = ndcg_at_k(results, q["gold_docs"])

        total_recall += recall
        total_precision += precision
        total_mrr += reciprocal_rank
        total_ndcg += ndcg

        print("\nQuestion:", q["question"])
        print("Gold Docs:", q["gold_docs"])

        print("Retrieved Docs:")
        for doc, score in results:
            print(doc["doc_id"], round(score, 4))

        print(f"Recall@5    : {recall:.4f}")
        print(f"Precision@5 : {precision:.4f}")
        print(f"MRR         : {reciprocal_rank:.4f}")
        print(f"nDCG@5      : {ndcg:.4f}")

    # IMPORTANT: This must stay INSIDE the outer loop
    n = len(questions)

    print(f"\n===== {name} Evaluation =====")
    print(f"Average Recall@5    : {total_recall/n:.4f}")
    print(f"Average Precision@5 : {total_precision/n:.4f}")
    print(f"Average MRR         : {total_mrr/n:.4f}")
    print(f"Average nDCG@5      : {total_ndcg/n:.4f}")