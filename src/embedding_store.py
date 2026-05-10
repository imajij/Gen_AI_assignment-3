"""
Embedding Store Module
Per-collection ChromaDB persistent store with local SentenceTransformer embeddings.

API:
 - add_documents(collection_name, chunks)
 - similarity_search(collection_name, query, top_k)
 - delete_collection(collection_name)
 - load_collection(collection_name)
 - list_collections()
 - get_collection_info(collection_name)
"""

import os
import time
import re
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


def _safe_collection_name(name: str) -> str:
    safe = name.replace(" ", "_").replace("/", "_")
    safe = re.sub(r"[^A-Za-z0-9_\-]", "", safe)
    if not safe:
        safe = f"collection_{int(time.time())}"
    return safe


class EmbeddingStore:
    """Manage embeddings and vectors using local SentenceTransformers + ChromaDB.

    Each logical source (file, session) can map to its own Chroma collection.
    """

    def __init__(self, persist_dir: str = "./data/vector_store", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.persist_dir = persist_dir
        self.model_name = model_name

        os.makedirs(self.persist_dir, exist_ok=True)

        # Prefer persistent client; fall back to in-memory if unavailable
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
        except Exception:
            self.client = chromadb.Client(
                Settings(anonymized_telemetry=False, allow_reset=True)
            )

        try:
            self.embedder = SentenceTransformer(self.model_name)
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{self.model_name}': {e}")

        self.default_collection_name = "documents"
        self.active_collection_name = self.default_collection_name
        self.active_collection = self._get_or_create_collection(self.default_collection_name)

    # -- collection helpers --
    def _get_or_create_collection(self, name: str):
        name = _safe_collection_name(name)
        try:
            return self.client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        except Exception:
            try:
                return self.client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
            except Exception as e:
                raise RuntimeError(f"Unable to create or get collection '{name}': {e}")

    def load_collection(self, collection_name: str) -> bool:
        """Set the active collection to the named collection (creates if missing)."""
        try:
            col = self._get_or_create_collection(collection_name)
            self.active_collection = col
            self.active_collection_name = _safe_collection_name(collection_name)
            return True
        except Exception as e:
            raise RuntimeError(f"Failed to load collection '{collection_name}': {e}")

    # -- core operations --
    def add_documents(self, chunks: List[Dict[str, Any]], collection_name: Optional[str] = None, batch_size: int = 128) -> int:
        """Add chunks to a collection. Each chunk: {'id','content','metadata'}.

        Returns number of items successfully added.
        """
        if not chunks:
            return 0

        col = self._get_or_create_collection(collection_name) if collection_name else self.active_collection

        added = 0
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids = [c['id'] for c in batch]
            docs = [c['content'] for c in batch]

            metadatas = []
            for c in batch:
                meta = c.get('metadata', {}) or {}
                flat = {}
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        flat[k] = v
                    else:
                        flat[k] = str(v)
                metadatas.append(flat)

            try:
                embeddings = self.embedder.encode(docs)
                # Convert numpy arrays to lists for ChromaDB
                embeddings = [e.tolist() if hasattr(e, 'tolist') else e for e in embeddings]
            except Exception as e:
                raise RuntimeError(f"Embedding generation failed: {e}")

            try:
                col.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metadatas)
                added += len(batch)
            except Exception:
                # fallback: try one-by-one
                for idx, (iid, doc, emb, meta) in enumerate(zip(ids, docs, embeddings, metadatas)):
                    try:
                        col.add(ids=[iid], documents=[doc], embeddings=[emb], metadatas=[meta])
                        added += 1
                    except Exception as e:
                        print(f"Failed to add id={iid}: {e}")

        return added

    def similarity_search(self, query: str, collection_name: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return top-k similar chunks for the query from the specified collection."""
        col = self._get_or_create_collection(collection_name) if collection_name else self.active_collection

        try:
            q_emb = self.embedder.encode(query)
            # Convert numpy array to list for ChromaDB
            q_emb = q_emb.tolist() if hasattr(q_emb, 'tolist') else q_emb
        except Exception as e:
            raise RuntimeError(f"Failed to encode query: {e}")

        try:
            results = col.query(query_embeddings=[q_emb], n_results=top_k)
        except Exception as e:
            raise RuntimeError(f"ChromaDB query failed: {e}")

        out: List[Dict[str, Any]] = []
        docs = results.get('documents', [])
        ids = results.get('ids', [])
        metadatas = results.get('metadatas', [])
        distances = results.get('distances', [])

        if docs and len(docs) > 0:
            docs0 = docs[0]
            ids0 = ids[0] if ids else [None] * len(docs0)
            metas0 = metadatas[0] if metadatas else [{}] * len(docs0)
            dists0 = distances[0] if distances else [0.0] * len(docs0)

            for i, doc in enumerate(docs0):
                dist = dists0[i] if i < len(dists0) else 0.0
                relevance = 1.0 - dist if isinstance(dist, (int, float)) else 0.0
                out.append({
                    'id': ids0[i] if i < len(ids0) else f"doc_{i}",
                    'content': doc,
                    'metadata': metas0[i] if i < len(metas0) else {},
                    'distance': dist,
                    'relevance_score': relevance,
                })

        return out

    def delete_collection(self, collection_name: str) -> bool:
        """Delete a named collection from ChromaDB. Returns True on success."""
        name = _safe_collection_name(collection_name)
        try:
            # try client-level delete
            try:
                self.client.delete_collection(name=name)
            except Exception:
                # fallback: delete all ids in collection
                col = self._get_or_create_collection(name)
                try:
                    all_ids = col.get(include=["ids"]).get('ids', [])
                except Exception:
                    all_ids = None
                if all_ids:
                    # flatten
                    flat_ids = [i for sub in all_ids for i in sub] if isinstance(all_ids[0], list) else all_ids
                    if flat_ids:
                        col.delete(ids=flat_ids)

            # reset active if needed
            if getattr(self, 'active_collection_name', None) == name:
                self.active_collection_name = self.default_collection_name
                self.active_collection = self._get_or_create_collection(self.default_collection_name)
            return True
        except Exception as e:
            print(f"Error deleting collection '{collection_name}': {e}")
            return False

    def list_collections(self) -> List[str]:
        try:
            cols = self.client.list_collections()
            names = [c.get('name') if isinstance(c, dict) else getattr(c, 'name', str(c)) for c in cols]
            return names
        except Exception:
            return [self.default_collection_name]

    def get_collection_info(self, collection_name: Optional[str] = None) -> Dict[str, Any]:
        try:
            col = self._get_or_create_collection(collection_name) if collection_name else self.active_collection
            count = col.count()
            return {
                'name': getattr(col, 'name', collection_name or self.active_collection_name),
                'total_documents': count,
                'embedding_model': self.model_name,
                'persist_dir': self.persist_dir,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get collection info: {e}")
