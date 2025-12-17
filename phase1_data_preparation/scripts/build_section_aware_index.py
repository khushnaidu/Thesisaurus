"""
section-aware vector index builder
- respects section boundaries when chunking
- adds section metadata to each chunk
- creates better semantic retrieval
"""

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
    section: str
    text: str
    start_pos: int
    end_pos: int


class SectionAwareChunker:
    def __init__(self, target_size=400, min_size=100, max_size=600):
        # target chunk size in words, with some flexibility
        self.target_size = target_size
        self.min_size = min_size
        self.max_size = max_size
        print(f"chunker: target {target_size} words, range [{min_size}-{max_size}]")

    def chunk_sections(self, sections, paper_id, start_chunk_id=0):
        """chunk sections while respecting boundaries"""
        chunks = []
        chunk_id = start_chunk_id

        for section in sections:
            section_name = section['name']
            text = section['text']
            words = text.split()

            # if section is small enough, keep it as one chunk
            if len(words) <= self.max_size:
                if len(words) >= self.min_size:
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        section=section_name,
                        text=text,
                        start_pos=section['start'],
                        end_pos=section['start'] + len(text)
                    ))
                    chunk_id += 1
                elif len(words) > 20:  # still include tiny sections if >20 words
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        section=section_name,
                        text=text,
                        start_pos=section['start'],
                        end_pos=section['start'] + len(text)
                    ))
                    chunk_id += 1
                continue

            # section is too big, split it
            section_chunks = self._split_section(
                words, section_name, paper_id,
                section['start'], chunk_id
            )
            chunks.extend(section_chunks)
            chunk_id += len(section_chunks)

        return chunks

    def _split_section(self, words, section_name, paper_id, start_pos, start_chunk_id):
        """split a long section into chunks at sentence boundaries"""
        chunks = []
        chunk_id = start_chunk_id

        # try to split at sentence boundaries (periods)
        text = ' '.join(words)
        sentences = text.split('. ')

        current_words = []
        current_start = start_pos

        for sentence in sentences:
            sentence_words = sentence.split()

            # would adding this sentence make the chunk too big?
            if len(current_words) + len(sentence_words) > self.max_size:
                # save current chunk if big enough
                if len(current_words) >= self.min_size:
                    chunk_text = ' '.join(current_words)
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        section=section_name,
                        text=chunk_text,
                        start_pos=current_start,
                        end_pos=current_start + len(chunk_text)
                    ))
                    chunk_id += 1
                    current_start += len(chunk_text) + 1
                    current_words = []

            current_words.extend(sentence_words)

        # save remaining words
        if len(current_words) >= self.min_size:
            chunk_text = ' '.join(current_words)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                paper_id=paper_id,
                section=section_name,
                text=chunk_text,
                start_pos=current_start,
                end_pos=current_start + len(chunk_text)
            ))

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

        # include section name in the text for better semantic matching
        texts = [f"[{c.section}] {c.text}" for c in chunks]
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
            'section_aware': True,
            'chunks': [
                {
                    'chunk_id': c.chunk_id,
                    'paper_id': c.paper_id,
                    'section': c.section,
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


def load_sections(sections_dir):
    """load all section files"""
    sections_path = Path(sections_dir)
    all_sections = {}

    if not sections_path.exists():
        print(f"error: dir not found: {sections_dir}")
        return all_sections

    json_files = list(sections_path.glob("*.json"))
    print(f"found {len(json_files)} section files")

    for json_file in json_files:
        paper_id = json_file.stem
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                sections = json.load(f)
                all_sections[paper_id] = sections
                print(f"  loaded {paper_id} ({len(sections)} sections)")
        except Exception as e:
            print(f"  error loading {paper_id}: {e}")

    return all_sections


def main():
    import sys

    print("\n" + "="*50)
    print("section-aware vector index builder")
    print("="*50 + "\n")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../outputs"

    # load sections
    print("step 1: loading sections...")
    sections_dir = Path(data_dir) / "sections"
    all_sections = load_sections(sections_dir)
    print(f"  loaded {len(all_sections)} papers\n")

    # chunk them section-aware
    print("step 2: chunking with section awareness...")
    chunker = SectionAwareChunker(target_size=400, min_size=100, max_size=600)
    all_chunks = []
    chunk_id = 0

    for paper_id, sections in all_sections.items():
        chunks = chunker.chunk_sections(sections, paper_id, chunk_id)
        all_chunks.extend(chunks)
        chunk_id += len(chunks)
        print(f"  {paper_id}: {len(chunks)} chunks")

    print(f"\n  total: {len(all_chunks)} chunks\n")

    # show section distribution
    section_counts = {}
    for c in all_chunks:
        section_counts[c.section] = section_counts.get(c.section, 0) + 1
    print("section distribution:")
    for section, count in sorted(section_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {section}: {count}")

    # build index
    print("\nstep 3: building faiss index...")
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
        print(f"\n{i}. {chunk.paper_id} [{chunk.section}] (dist: {score:.2f})")
        print(f"   {chunk.text[:150]}...")

    print()


if __name__ == "__main__":
    main()
