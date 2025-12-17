"""
pdf extraction using pymupdf - produces much cleaner text than pdfplumber
also detects sections for better chunking later
"""

import json
import re
from pathlib import Path
import fitz  # pymupdf


class PDFExtractor:
    def __init__(self):
        # known section names in academic papers
        self.known_sections = {
            'abstract', 'introduction', 'background', 'related work',
            'method', 'methods', 'methodology', 'approach',
            'model', 'architecture', 'system', 'framework',
            'experiments', 'experiment', 'experimental setup', 'experimental results',
            'results', 'evaluation', 'analysis',
            'discussion', 'limitations', 'future work',
            'conclusion', 'conclusions', 'summary',
            'acknowledgments', 'acknowledgements',
            'references', 'bibliography',
            'appendix', 'supplementary', 'supplementary material',
            'training', 'training details', 'implementation', 'implementation details',
            'dataset', 'datasets', 'data collection',
            'ablation', 'ablation study', 'ablations',
            'preliminaries', 'problem setup', 'problem formulation',
        }

    def extract_text(self, pdf_path):
        """extract clean text from pdf using pymupdf"""
        doc = fitz.open(pdf_path)
        pages = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            # basic cleanup
            text = self._clean_text(text)
            pages.append({
                'page_num': page_num + 1,
                'text': text
            })

        doc.close()
        return pages

    def _clean_text(self, text):
        """clean up extracted text"""
        # fix common pdf artifacts
        text = re.sub(r'\s+', ' ', text)  # normalize whitespace
        text = re.sub(r'- \n', '', text)  # fix hyphenation
        text = re.sub(r'\n', ' ', text)  # single line
        text = re.sub(r'\s{2,}', ' ', text)  # multiple spaces
        text = text.strip()
        return text

    def detect_sections(self, full_text):
        """find section boundaries using known section names"""
        sections = []
        current_section = {'name': 'preamble', 'start': 0, 'text': ''}

        # split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', full_text)
        current_pos = 0

        for sentence in sentences:
            sentence_clean = sentence.strip()
            section_match = self._find_section_header(sentence_clean)

            if section_match:
                # save previous section if it has content
                if current_section['text'].strip() and len(current_section['text']) > 50:
                    sections.append(current_section.copy())

                current_section = {
                    'name': section_match,
                    'start': current_pos,
                    'text': sentence_clean + ' '
                }
            else:
                current_section['text'] += sentence_clean + ' '

            current_pos += len(sentence) + 1

        # save last section
        if current_section['text'].strip() and len(current_section['text']) > 50:
            sections.append(current_section)

        return sections

    def _find_section_header(self, text):
        """check if text starts with a known section header"""
        text_lower = text.lower().strip()

        # strip leading numbers/roman numerals
        text_clean = re.sub(r'^[\d\.]+\s*', '', text_lower)
        text_clean = re.sub(r'^[ivx]+[\.\s]+', '', text_clean)
        text_clean = text_clean.strip()

        # check first few words against known sections
        words = text_clean.split()[:4]  # first 4 words

        for i in range(len(words), 0, -1):
            candidate = ' '.join(words[:i])
            if candidate in self.known_sections:
                return candidate

        return None

    def extract_metadata(self, pages, pdf_path):
        """extract title, authors, abstract from first pages"""
        if not pages:
            return {}

        first_page = pages[0]['text']
        second_page = pages[1]['text'] if len(pages) > 1 else ''

        # title is usually in the first 500 chars
        title = self._extract_title(first_page[:500])

        # authors usually follow title
        authors = self._extract_authors(first_page[:1500])

        # abstract
        abstract = self._extract_abstract(first_page + ' ' + second_page)

        # year from text or filename
        year = self._extract_year(first_page[:2000], pdf_path)

        return {
            'paper_id': Path(pdf_path).stem,
            'title': title,
            'authors': authors,
            'year': year,
            'abstract': abstract,
            'num_pages': len(pages),
            'pdf_path': str(pdf_path)
        }

    def _extract_title(self, text):
        """grab the title - usually first big text chunk"""
        # look for capitalized text at the start
        lines = text.split('.')
        for line in lines[:5]:
            line = line.strip()
            # title is usually >10 chars, <200 chars
            if 10 < len(line) < 200:
                # skip if looks like author line
                if '@' in line or 'university' in line.lower():
                    continue
                return line
        return text[:100].strip()

    def _extract_authors(self, text):
        """extract author names"""
        # look for common author patterns
        # names with commas, or names before institution
        author_section = text[:1000]

        # find text between title and abstract
        abstract_pos = author_section.lower().find('abstract')
        if abstract_pos > 0:
            author_section = author_section[:abstract_pos]

        # look for names (capitalized words with commas)
        names = re.findall(r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', author_section)

        # filter out common non-name patterns
        skip_words = ['Abstract', 'Introduction', 'University', 'Institute', 'Department']
        names = [n for n in names if not any(w in n for w in skip_words)]

        return ', '.join(names[:15]) if names else ''

    def _extract_abstract(self, text):
        """find the abstract section"""
        # look for "Abstract" header
        abstract_match = re.search(
            r'(?:Abstract|ABSTRACT)[:\s]*(.+?)(?:(?:\d+\.?\s+)?Introduction|INTRODUCTION|Keywords|1\s)',
            text,
            re.DOTALL | re.IGNORECASE
        )

        if abstract_match:
            abstract = abstract_match.group(1).strip()
            # clean it up
            abstract = re.sub(r'\s+', ' ', abstract)
            if len(abstract) > 50:
                return abstract[:2000]  # cap at 2000 chars

        return ''

    def _extract_year(self, text, pdf_path):
        """extract publication year"""
        # look for arxiv date pattern first (more reliable)
        arxiv_match = re.search(r'arXiv:(\d{4})\.\d+', text)
        if arxiv_match:
            year_digits = arxiv_match.group(1)[:2]
            return 2000 + int(year_digits)

        # look for explicit year mentions
        year_patterns = [
            r'(?:published|accepted|submitted).*?(20[12]\d)',
            r'(?:conference|workshop|journal).*?(20[12]\d)',
            r'\b(20[12]\d)\b',
        ]

        for pattern in year_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 2015 <= year <= 2025:
                    return year

        # fallback to filename
        filename = Path(pdf_path).stem
        year_match = re.search(r'20[12]\d', filename)
        if year_match:
            return int(year_match.group())

        return None

    def process_pdf(self, pdf_path):
        """full extraction pipeline for one pdf"""
        print(f"  extracting: {Path(pdf_path).name}")

        pages = self.extract_text(pdf_path)
        if not pages:
            print(f"    warning: no pages extracted")
            return None, None

        # combine all text
        full_text = ' '.join([p['text'] for p in pages])

        # detect sections
        sections = self.detect_sections(full_text)
        print(f"    found {len(sections)} sections")

        # extract metadata
        metadata = self.extract_metadata(pages, pdf_path)
        print(f"    title: {metadata['title'][:50]}...")

        return {
            'metadata': metadata,
            'sections': sections,
            'full_text': full_text,
            'pages': pages
        }


def batch_extract(papers_dir, output_dir):
    """process all pdfs in a directory"""
    papers_path = Path(papers_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # output dirs
    fulltext_dir = output_path / "full_text"
    sections_dir = output_path / "sections"
    fulltext_dir.mkdir(exist_ok=True)
    sections_dir.mkdir(exist_ok=True)

    # find all pdfs
    pdfs = list(papers_path.glob("*.pdf"))
    print(f"found {len(pdfs)} pdfs\n")

    extractor = PDFExtractor()
    all_metadata = []

    for i, pdf_path in enumerate(pdfs, 1):
        print(f"[{i}/{len(pdfs)}] {pdf_path.name}")

        try:
            result = extractor.process_pdf(pdf_path)
            if not result:
                continue

            paper_id = result['metadata']['paper_id']

            # save full text
            with open(fulltext_dir / f"{paper_id}.txt", 'w', encoding='utf-8') as f:
                f.write(result['full_text'])

            # save sections as json
            with open(sections_dir / f"{paper_id}.json", 'w', encoding='utf-8') as f:
                json.dump(result['sections'], f, indent=2)

            # add to metadata
            meta = result['metadata']
            meta['full_text'] = f"[Saved to {paper_id}.txt]"
            meta['sections_file'] = f"[Saved to {paper_id}.json]"
            all_metadata.append(meta)

            print(f"    saved")

        except Exception as e:
            print(f"    error: {e}")

    # save metadata
    with open(output_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(all_metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"done! extracted {len(all_metadata)} papers")
    print(f"output: {output_path}")
    print('='*50)

    return all_metadata


if __name__ == "__main__":
    import sys

    print("pdf extraction with pymupdf")
    print("="*50 + "\n")

    papers_dir = sys.argv[1] if len(sys.argv) > 1 else "../../papers"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "../data"

    batch_extract(papers_dir, output_dir)
