import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


class WebSearchTool:
    """Search arXiv for papers"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
    
    def _parse_entry(self, entry):
        """Parse arXiv XML entry"""
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        paper = {}
        paper['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        paper['arxiv_id'] = entry.find('atom:id', ns).text.split('/abs/')[-1]
        paper['url'] = entry.find('atom:id', ns).text
        paper['published'] = entry.find('atom:published', ns).text[:10]
        
        authors = entry.findall('atom:author', ns)
        paper['authors'] = [a.find('atom:name', ns).text for a in authors]
        
        summary = entry.find('atom:summary', ns)
        paper['abstract'] = summary.text.strip().replace('\n', ' ') if summary is not None else ''
        
        pdf_link = entry.find("atom:link[@title='pdf']", ns)
        paper['pdf_url'] = pdf_link.get('href') if pdf_link is not None else None
        
        return paper
    
    def search_arxiv(self, query, max_results=5):
        """Search arXiv"""
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            papers = [self._parse_entry(entry) for entry in entries]
            
            return {
                "success": True,
                "query": query,
                "count": len(papers),
                "papers": papers
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_paper_by_arxiv_id(self, arxiv_id):
        """Get paper details by arXiv ID"""
        try:
            params = {'id_list': arxiv_id, 'max_results': 1}
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entry = root.find('atom:entry', ns)
            
            if entry is None:
                return {"success": False, "error": f"Paper {arxiv_id} not found"}
            
            paper = self._parse_entry(entry)
            
            return {"success": True, "paper": paper}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_by_author(self, author_name, max_results=5):
        """Search by author"""
        try:
            params = {
                'search_query': f'au:{author_name}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            papers = [self._parse_entry(entry) for entry in entries]
            
            return {
                "success": True,
                "author": author_name,
                "count": len(papers),
                "papers": papers
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_recent_papers(self, topic, max_results=10):
        """Find recent papers on a topic"""
        try:
            params = {
                'search_query': f'all:{topic}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            
            url = f"{self.base_url}?{urllib.parse.urlencode(params)}"
            
            with urllib.request.urlopen(url) as response:
                xml_data = response.read()
            
            root = ET.fromstring(xml_data)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            entries = root.findall('atom:entry', ns)
            papers = [self._parse_entry(entry) for entry in entries]
            
            return {
                "success": True,
                "topic": topic,
                "count": len(papers),
                "papers": papers
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
