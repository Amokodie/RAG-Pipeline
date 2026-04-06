"""
Mini RAG starter for Session 8 classroom exercise.

Baseline retriever:
- paragraph chunking
- TF-IDF vectorization
- cosine similarity ranking

Optional extension:
- replace TF-IDF with sentence-transformers
"""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parents[1]
CORPUS_DIR = BASE_DIR / "dataset" / "corpus"
QUERY_FILE = BASE_DIR / "dataset" / "eval_queries.csv"


def load_documents(corpus_dir: Path):
    docs = []
    for path in sorted(corpus_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        doc_id = path.name.split("_")[0]
        docs.append({"doc_id": doc_id, "filename": path.name, "text": text})
    return docs


def chunk_by_paragraph(documents):
    chunks = []
    for doc in documents:
        paragraphs = [p.strip() for p in doc["text"].split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs, start=1):
            chunks.append(
                {
                    "chunk_id": f'{doc["doc_id"]}_P{i}',
                    "doc_id": doc["doc_id"],
                    "text": para,
                }
            )
    return pd.DataFrame(chunks)


def build_retriever(chunk_df):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(chunk_df["text"])
    return vectorizer, matrix


def retrieve(query, vectorizer, matrix, chunk_df, top_k=3):
    q = vectorizer.transform([query])
    scores = cosine_similarity(q, matrix).flatten()
    out = chunk_df.copy()
    out["score"] = scores
    out = out.sort_values("score", ascending=False).head(top_k)
    return out[["chunk_id", "doc_id", "score", "text"]]


def authority_filter(query_results, allowed_docs=None):
    if allowed_docs is None:
        return query_results
    return query_results[query_results["doc_id"].isin(allowed_docs)]


def main():
    docs = load_documents(CORPUS_DIR)
    chunk_df = chunk_by_paragraph(docs)
    vectorizer, matrix = build_retriever(chunk_df)

    print(f"Loaded {len(docs)} documents and {len(chunk_df)} chunks.\n")

    query_df = pd.read_csv(QUERY_FILE)
    sample_ids = ["Q04", "Q05", "Q10"]

    for qid in sample_ids:
        row = query_df[query_df["query_id"] == qid].iloc[0]
        print("=" * 80)
        print(f"{qid}: {row['query']}")
        print("-" * 80)
        result = retrieve(row["query"], vectorizer, matrix, chunk_df, top_k=3)
        print(result.to_string(index=False))
        print()

    # TODO 1:
    # Try filtering to high-authority docs only for Q01/Q02 and compare results.
    #
    # TODO 2:
    # Add a simple synonym rewrite before retrieval, for example:
    # "uneven left-right fan draw" -> "fan current imbalance"
    #
    # TODO 3:
    # Compare paragraph chunking with a smaller fixed-size chunker.


if __name__ == "__main__":
    main()
