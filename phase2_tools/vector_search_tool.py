import json
from pathlib import Path


class VectorSearchTool:
    """semantic search with faiss"""

    def __init__(self, index_path="../phase1_data_preparation/outputs/faiss_index.bin",
                 metadata_path="../phase1_data_preparation/outputs/chunk_metadata.json"):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)

        if not self.index_path.exists():
            raise FileNotFoundError(f"index not found: {index_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"metadata not found: {metadata_path}")

        self.model = None
        self.index = None
        self.chunks = None

    def _load(self):
        # lazy load to save memory
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import faiss

            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.index = faiss.read_index(str(self.index_path))

            with open(self.metadata_path, 'r') as f:
                self.chunks = json.load(f)['chunks']

    def search(self, query, top_k=5):
        try:
            self._load()
            emb = self.model.encode([query])
            dists, idxs = self.index.search(emb.astype('float32'), top_k)

            results = []
            for idx, dist in zip(idxs[0], dists[0]):
                c = self.chunks[idx]
                results.append({
                    "paper_id": c['paper_id'],
                    "text": c['text'],
                    "score": float(dist),
                    "chunk_id": c['chunk_id']
                })
            return {"success": True, "query": query, "count": len(results), "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_within_paper(self, query, paper_id, top_k=3):
        try:
            self._load()

            # get chunks for this paper
            paper_idxs = [i for i, c in enumerate(self.chunks) if c['paper_id'] == paper_id]
            if not paper_idxs:
                return {"success": False, "error": f"no chunks for {paper_id}"}

            emb = self.model.encode([query])

            import numpy as np
            chunk_embs = np.array([self.index.reconstruct(i) for i in paper_idxs])

            from numpy.linalg import norm
            dists = [norm(emb[0] - ce) for ce in chunk_embs]
            sorted_idxs = np.argsort(dists)[:top_k]

            results = []
            for idx in sorted_idxs:
                orig_idx = paper_idxs[idx]
                c = self.chunks[orig_idx]
                results.append({
                    "paper_id": c['paper_id'],
                    "text": c['text'],
                    "score": float(dists[idx]),
                    "chunk_id": c['chunk_id']
                })
            return {"success": True, "query": query, "paper_id": paper_id, "count": len(results), "results": results}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_paper_chunks(self, paper_id):
        try:
            self._load()
            chunks = [c for c in self.chunks if c['paper_id'] == paper_id]
            if not chunks:
                return {"success": False, "error": f"no chunks for {paper_id}"}
            return {"success": True, "paper_id": paper_id, "count": len(chunks), "chunks": chunks}
        except Exception as e:
            return {"success": False, "error": str(e)}
