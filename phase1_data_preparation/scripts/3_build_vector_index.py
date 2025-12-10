import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
import faiss


@dataclass
class Chunk:
    chunk_id: int
    paper_id: str
    text: str
    start_pos: int
    end_pos: int


class TextChunker:
    def __init__(self, chunk_size=512, overlap=50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        print(f"chunker: {chunk_size} words, {overlap} overlap")

    def chunk_text(self, text, paper_id, start_chunk_id=0):
        words = text.split()
        chunks = []
        chunk_id = start_chunk_id
        stride = self.chunk_size - self.overlap

        i = 0
        while i < len(words):
            end = min(i + self.chunk_size, len(words))
            chunk_words = words[i:end]
            chunk_text = ' '.join(chunk_words)

            chunk = Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                text=chunk_text,
                start_pos=i,
                end_pos=end
            )
            chunks.append(chunk)

            chunk_id += 1
            i += stride

            if end >= len(words):
                break

        return chunks


class VectorIndexBuilder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.chunks = []
        print(f"using model: {model_name}")

    def load_model(self):
        print(f"loading {self.model_name}...")
        self.model = SentenceTransformer(self.model_name)
        dim = self.model.get_sentence_embedding_dimension()
        print(f"  loaded, dim={dim}")

    def create_embeddings(self, texts):
        print(f"embedding {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings

    def build_index(self, chunks):
        if not self.model:
            print("error: load model first")
            return

        if len(chunks) == 0:
            print("error: no chunks")
            return

        print(f"\nbuilding index from {len(chunks)} chunks...")

        texts = [chunk.text for chunk in chunks]
        embeddings = self.create_embeddings(texts)

        dim = embeddings.shape[1]
        print(f"embedding dim: {dim}")

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype('float32'))
        self.chunks = chunks

        print(f"  index built, {self.index.ntotal} vectors")

    def save_index(self, index_path="faiss_index.bin", metadata_path="chunk_metadata.json"):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)

        print(f"saving index to {index_path}...")
        faiss.write_index(self.index, index_path)

        print(f"saving metadata to {metadata_path}...")
        metadata = {
            'model_name': self.model_name,
            'num_chunks': len(self.chunks),
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'paper_id': c.paper_id,
                    'text': c.text,
                    'start_pos': c.start_pos,
                    'end_pos': c.end_pos
                }
                for c in self.chunks
            ]
        }

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        print("  saved")

    def load_index(self, index_path="faiss_index.bin", metadata_path="chunk_metadata.json"):
        print(f"loading index from {index_path}...")
        self.index = faiss.read_index(index_path)
        print(f"  loaded {self.index.ntotal} vectors")

        print(f"loading metadata from {metadata_path}...")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        self.chunks = [
            Chunk(
                chunk_id=c['chunk_id'],
                paper_id=c['paper_id'],
                text=c['text'],
                start_pos=c['start_pos'],
                end_pos=c['end_pos']
            )
            for c in metadata['chunks']
        ]

        self.model_name = metadata['model_name']
        print(f"  loaded {len(self.chunks)} chunks")

    def search(self, query, top_k=5):
        if not self.model:
            print("error: load model first")
            return []

        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            chunk = self.chunks[idx]
            results.append((chunk, float(dist)))

        return results


def load_all_papers(papers_dir):
    papers = {}
    papers_path = Path(papers_dir)

    if not papers_path.exists():
        print(f"error: dir not found: {papers_dir}")
        return papers

    txt_files = list(papers_path.glob("*.txt"))
    print(f"found {len(txt_files)} paper files")

    for txt_file in txt_files:
        paper_id = txt_file.stem

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
                papers[paper_id] = text
                print(f"  loaded {paper_id} ({len(text)} chars)")
        except Exception as e:
            print(f"  error loading {paper_id}: {e}")

    return papers


def main():
    import sys

    print("\n" + "="*50)
    print("phase 1.4: vector index creation")
    print("="*50 + "\n")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../outputs"

    # load papers
    print("step 1: loading papers...")
    papers_dir = Path(data_dir) / "full_text"
    papers = load_all_papers(papers_dir)
    print(f"  loaded {len(papers)} papers\n")

    # chunk em
    print("step 2: chunking...")
    chunker = TextChunker(chunk_size=512, overlap=50)
    all_chunks = []
    chunk_id = 0

    for paper_id, text in papers.items():
        chunks = chunker.chunk_text(text, paper_id, chunk_id)
        all_chunks.extend(chunks)
        chunk_id += len(chunks)
        print(f"  {paper_id}: {len(chunks)} chunks")

    print(f"\n  total: {len(all_chunks)} chunks\n")

    # build index
    print("step 3: building faiss index...")
    builder = VectorIndexBuilder()
    builder.load_model()
    builder.build_index(all_chunks)

    # save
    print("\nstep 4: saving...")
    index_path = str(Path(output_dir) / "faiss_index.bin")
    metadata_path = str(Path(output_dir) / "chunk_metadata.json")
    builder.save_index(index_path, metadata_path)

    print("\n" + "="*50)
    print(f"index saved to: {index_path}")
    print(f"metadata saved to: {metadata_path}")
    print("="*50)

    # quick test
    print("\ntesting search: 'vision-language-action models'")
    results = builder.search("vision-language-action models for robotics", top_k=3)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"\n{i}. {chunk.paper_id} (dist: {score:.2f})")
        print(f"   {chunk.text[:150]}...")

    print()


if __name__ == "__main__":
    main()
