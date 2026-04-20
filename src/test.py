# juste un test
import bm25s
import pickle

import chromadb
from sentence_transformers import SentenceTransformer


def verify_bm25(
    query: str,
    index_path: str = "src/data/processed/bm25_index",
    chunks_path: str = "src/data/processed/chunks.pkl",
    k: int = 5,
):
    # 1. Charger l'index et les métadatas
    retriever = bm25s.BM25.load(index_path, load_corpus=True)
    with open(chunks_path, "rb") as f:
        metadata = pickle.load(f)

    # 2. Tokeniser la requête
    query_tokens = bm25s.tokenize([query])

    # 3. Rechercher
    results, scores = retriever.retrieve(query_tokens, k=k * 5)

    # 4. Afficher les résultats (MinimalSource)
    print(f"Résultats pour : '{query}'")
    for i in range(k * 5):
        # Récupérer l'index global du chunk trouvé
        doc_index = results[0, i]
        source = metadata[doc_index]

        print(f"[{i+1}] Score: {scores[0, i]:.4f}")
        print(f"    Fichier: {source.file_path}")
        print(
            f"    Position: {source.first_character_index} -> {source.last_character_index}"
        )

    chroma_retriever = chromadb.PersistentClient(path="src/data/processed")
    collection = chroma_retriever.get_collection("Corpus")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(query, show_progress_bar=False).tolist()
    res = collection.query(query_embeddings=embeddings, n_results=k * 5)
    ids = res['ids'][0]
    documents = res['documents'][0]
    metadatas = res['metadatas'][0]
    distances = res['distances'][0]

    for i in range(len(ids)):
        meta = metadatas[i]
        print(f"Rang {i+1} | Distance: {distances[i]:.4f}")
        print(f"Fichier: {meta['file_path']}")

