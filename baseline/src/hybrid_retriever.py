class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever

    def retrieve(self, query, top_k=5):
        bm25_results = self.bm25.retrieve(query, top_k)
        dense_results = self.dense.retrieve(query, top_k)

        combined = {}

        for doc, score in bm25_results:
            combined[doc["doc_id"]] = {
                "doc": doc,
                "score": score,
                "source": "bm25"
            }

        for doc, score in dense_results:
            if doc["doc_id"] in combined:
                combined[doc["doc_id"]]["score"] += score
                combined[doc["doc_id"]]["source"] = "hybrid"
            else:
                combined[doc["doc_id"]] = {
                    "doc": doc,
                    "score": score,
                    "source": "dense"
                }

        ranked = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return [(item["doc"], item["score"]) for item in ranked[:top_k]]