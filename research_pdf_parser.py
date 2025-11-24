"""
Phase 1.1: PDF Processing Infrastructure
Extracts metadata and full text from research papers
"""

import os
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path

try:
    import pdfplumber
    from PyPDF2 import PdfReader
except ImportError:
    print("Missing dependencies. Install with: pip install pdfplumber PyPDF2")
    raise


@dataclass
class PDFMetadata:
    """Structured metadata extracted from a PDF"""
    paper_id: str
    title: str
    authors: str
    year: Optional[int]
    venue: Optional[str]
    arxiv_id: Optional[str]
    pdf_path: str
    abstract: str
    num_pages: int
    full_text: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class PageContent:
    """Content from a single page"""
    page_num: int
    text: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PDFProcessor:
    """Comprehensive PDF processor for research papers."""
    
    def __init__(self, papers_directory: str):
        self.papers_dir = Path(papers_directory)
        if not self.papers_dir.exists():
            raise ValueError(f"Directory not found: {papers_directory}")
    
    def extract_full_text(self, pdf_path: str) -> str:
        """Extract all text from PDF with proper spacing."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                text_parts = []
                for page in pdf.pages:
                    # Use layout-aware extraction with custom settings
                    text = page.extract_text(
                        x_tolerance=3,        # Horizontal tolerance for grouping chars into words
                        y_tolerance=3,        # Vertical tolerance for grouping chars into lines
                        layout=True,          # Preserve layout
                        x_density=7.25,       # Character density for word detection
                        y_density=13          # Line density
                    )
                    if text:
                        # Post-process to fix common issues
                        text = self._fix_spacing(text)
                        text_parts.append(text)
                return "\n\n".join(text_parts)
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def _fix_spacing(self, text: str) -> str:
        """Fix common spacing issues in extracted text."""
        # Add space before capital letters in obvious concatenations
        # e.g., "trainingData" -> "training Data"
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Add space between word and number
        # e.g., "learning3" -> "learning 3"
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Fix common concatenated words (corpus-based)
        common_fixes = [
            (r'\blearningrate\b', 'learning rate'),
            (r'\bbatchsize\b', 'batch size'),
            (r'\btrainingdata\b', 'training data'),
            (r'\btrainingset\b', 'training set'),
            (r'\btestset\b', 'test set'),
            (r'\bvalidationset\b', 'validation set'),
            (r'\bneuralnetwork\b', 'neural network'),
            (r'\bdeeplearning\b', 'deep learning'),
            (r'\breinforcementlearning\b', 'reinforcement learning'),
        ]
        
        for pattern, replacement in common_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text
    
    def extract_page_by_page(self, pdf_path: str) -> List[PageContent]:
        """Extract text page by page for citation purposes."""
        pages = []
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13
                    )
                    if text:
                        text = self._fix_spacing(text)
                        pages.append(PageContent(page_num=i, text=text))
        except Exception as e:
            print(f"Error extracting pages from {pdf_path}: {e}")
        return pages
    
    def extract_metadata(self, pdf_path: str, paper_id: Optional[str] = None, 
                        include_full_text: bool = True) -> PDFMetadata:
        """Extract metadata from PDF."""
        path = Path(pdf_path)
        
        if paper_id is None:
            paper_id = path.stem.replace(" ", "_").replace("-", "_")
        
        full_text = self.extract_full_text(pdf_path)
        
        return PDFMetadata(
            paper_id=paper_id,
            title=self._extract_title(full_text, pdf_path),
            authors=self._extract_authors(full_text),
            year=self._extract_year(full_text, pdf_path),
            venue=self._extract_venue(full_text),
            arxiv_id=self._extract_arxiv_id(full_text),
            pdf_path=str(path.absolute()),
            abstract=self._extract_abstract(full_text),
            num_pages=self._get_page_count(pdf_path),
            full_text=full_text if include_full_text else None
        )
    
    def _extract_title(self, text: str, pdf_path: str) -> str:
        """Extract paper title."""
        try:
            reader = PdfReader(pdf_path)
            if reader.metadata and reader.metadata.title:
                title = reader.metadata.title.strip()
                if title and len(title) > 10 and len(title) < 300:
                    return title
        except:
            pass
        
        lines = text.split('\n')
        for line in lines[:15]:
            line = line.strip()
            if (len(line) > 15 and len(line) < 250 and 
                not line.startswith('http') and
                not re.search(r'^\d{4}$', line) and
                not re.search(r'arXiv:', line, re.IGNORECASE)):
                return line
        
        return Path(pdf_path).stem.replace("_", " ").replace("-", " ")
    
    def _extract_authors(self, text: str) -> str:
        """Extract authors."""
        lines = text.split('\n')
        
        title_idx = -1
        for i, line in enumerate(lines[:15]):
            line = line.strip()
            if len(line) > 15 and len(line) < 250:
                title_idx = i
                break
        
        if title_idx == -1:
            return "Unknown"
        
        author_lines = []
        for i in range(title_idx + 1, min(title_idx + 15, len(lines))):
            line = lines[i].strip()
            
            if not line:
                continue
            
            if re.match(r'^(Abstract|Introduction|1\s+Introduction)', line, re.IGNORECASE):
                break
            
            if re.search(r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+', line):
                if not re.search(r'\b(University|Institute|Laboratory|Lab|Department|College|School)\b', line):
                    author_lines.append(line)
                    if len(author_lines) >= 3:
                        break
            else:
                if author_lines:
                    break
        
        if author_lines:
            authors = ' '.join(author_lines)
            authors = re.sub(r'[∗†‡§¶#\d]', '', authors)
            authors = re.sub(r'\s+', ' ', authors).strip()
            if len(authors) > 500:
                authors = authors[:500] + "..."
            return authors
        
        return "Unknown"
    
    def _extract_year(self, text: str, pdf_path: str) -> Optional[int]:
        """Extract publication year."""
        try:
            reader = PdfReader(pdf_path)
            if reader.metadata and reader.metadata.creation_date:
                return reader.metadata.creation_date.year
        except:
            pass
        
        first_page = text[:1000]
        years = re.findall(r'\b(20[0-2][0-9])\b', first_page)
        if years:
            return int(years[0])
        
        return None
    
    def _extract_venue(self, text: str) -> Optional[str]:
        """Extract conference/journal venue."""
        search_text = text[:3000].lower()
        
        venues_patterns = [
            (r'neurips', 'NeurIPS'),
            (r'\bicml\b', 'ICML'),
            (r'\biclr\b', 'ICLR'),
            (r'\bcvpr\b', 'CVPR'),
            (r'\biccv\b', 'ICCV'),
            (r'\beccv\b', 'ECCV'),
            (r'\bicra\b', 'ICRA'),
            (r'\biros\b', 'IROS'),
            (r'\brss\b', 'RSS'),
            (r'\bcorl\b', 'CoRL'),
            (r'arxiv', 'arXiv'),
        ]
        
        for pattern, venue_name in venues_patterns:
            match = re.search(pattern + r'\s*(?:20\d{2})?', search_text)
            if match:
                year_match = re.search(r'(20\d{2})', match.group())
                if year_match:
                    return f"{venue_name} {year_match.group(1)}"
                return venue_name
        
        return None
    
    def _extract_arxiv_id(self, text: str) -> Optional[str]:
        """Extract arXiv ID if present."""
        search_text = text[:5000]
        
        patterns = [
            r'arXiv:(\d{4}\.\d{4,5})',
            r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
            r'ar[xX]iv\s*:\s*(\d{4}\.\d{4,5})',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, search_text, re.IGNORECASE)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_abstract(self, text: str) -> str:
        """Extract abstract section."""
        patterns = [
            r'Abstract\s*[:\-—]?\s*\n(.*?)(?:\n\s*\n\s*(?:1\s+)?Introduction|\n\s*\n\s*Keywords|\Z)',
            r'ABSTRACT\s*[:\-—]?\s*\n(.*?)(?:\n\s*\n\s*(?:1\s+)?INTRODUCTION|\n\s*\n\s*KEYWORDS|\Z)',
        ]
        
        for pattern in patterns:
            abstract_match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if abstract_match:
                abstract = abstract_match.group(1).strip()
                abstract = re.sub(r'\s+', ' ', abstract)
                abstract = re.sub(r'[∗†‡§¶•]', '', abstract)
                if len(abstract) > 50:
                    return abstract[:1500]
        
        return ""
    
    def _get_page_count(self, pdf_path: str) -> int:
        """Get number of pages in PDF."""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except:
            return 0
    
    def batch_process_pdfs(self, output_dir: str = "./processed_papers",
                           include_full_text: bool = True,
                           save_pages: bool = True) -> List[PDFMetadata]:
        """Process all PDFs in the directory."""
        pdf_files = list(self.papers_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {self.papers_dir}")
            return []
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if include_full_text:
            fulltext_dir = output_path / "full_text"
            fulltext_dir.mkdir(exist_ok=True)
        
        if save_pages:
            pages_dir = output_path / "pages"
            pages_dir.mkdir(exist_ok=True)
        
        print(f"Found {len(pdf_files)} PDF files. Processing...")
        
        all_metadata = []
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.name}")
            try:
                metadata = self.extract_metadata(str(pdf_path), include_full_text=include_full_text)
                
                print(f"  ✓ Title: {metadata.title[:60]}...")
                print(f"  ✓ Authors: {metadata.authors[:60]}...")
                print(f"  ✓ Year: {metadata.year}, Pages: {metadata.num_pages}")
                
                if include_full_text and metadata.full_text:
                    text_file = fulltext_dir / f"{metadata.paper_id}.txt"
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(metadata.full_text)
                    print(f"  ✓ Full text saved ({len(metadata.full_text):,} chars)")
                    metadata.full_text = f"[Saved to {text_file.name}]"
                
                if save_pages:
                    pages = self.extract_page_by_page(str(pdf_path))
                    pages_file = pages_dir / f"{metadata.paper_id}_pages.json"
                    with open(pages_file, 'w', encoding='utf-8') as f:
                        json.dump([p.to_dict() for p in pages], f, indent=2, ensure_ascii=False)
                    print(f"  ✓ Pages saved ({len(pages)} pages)")
                
                all_metadata.append(metadata)
                
            except Exception as e:
                print(f"  ✗ Error processing {pdf_path.name}: {e}")
        
        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump([m.to_dict() for m in all_metadata], f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"✓ Successfully processed {len(all_metadata)}/{len(pdf_files)} papers")
        print(f"✓ Metadata: {metadata_file}")
        if include_full_text:
            print(f"✓ Full text: {fulltext_dir}/")
        if save_pages:
            print(f"✓ Pages: {pages_dir}/")
        print(f"\n Next: Run extraction with")
        print(f"   python extract_structured_info.py")
        
        return all_metadata


if __name__ == "__main__":
    import sys
    
    print(f"Phase 1.1: PDF Processing")
    print(f"{'='*60}\n")
    
    papers_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/khushnaidu/NLP_Project/papers"
    
    if not os.path.exists(papers_dir):
        print(f"Error: Directory '{papers_dir}' not found.")
        print(f"Usage: python research_pdf_parser.py <papers_directory>")
        sys.exit(1)
    
    processor = PDFProcessor(papers_dir)
    processor.batch_process_pdfs()

