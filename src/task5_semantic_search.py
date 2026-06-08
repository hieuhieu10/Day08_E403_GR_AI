import chromadb
from sentence_transformers import SentenceTransformer


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Dense semantic search on ChromaDB.
    """

    model = SentenceTransformer("BAAI/bge-m3")
    query_embedding = model.encode(query, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path="./chromadb")

    collection = client.get_collection("DrugLawDocs")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    output = []

    for i in range(len(results["documents"][0])):

        distance = results["distances"][0][i]

        # cosine similarity approximation (Chroma uses distance)
        score = 1 - distance

        output.append({
            "content": results["documents"][0][i],
            "score": float(score),
            "metadata": results["metadatas"][0][i],
        })

    output.sort(key=lambda x: x["score"], reverse=True)

    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma túy", top_k=5)

    for r in results:
        print(f"[{r['score']:.4f}] {r['content'][:120]}...")