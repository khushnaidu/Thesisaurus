import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET


class WebSearchTool:
    """arxiv api wrapper"""

    def __init__(self):
        self.base = "http://export.arxiv.org/api/query"

    def _parse(self, entry):
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        p = {}
        p['title'] = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
        p['arxiv_id'] = entry.find('atom:id', ns).text.split('/abs/')[-1]
        p['url'] = entry.find('atom:id', ns).text
        p['published'] = entry.find('atom:published', ns).text[:10]

        authors = entry.findall('atom:author', ns)
        p['authors'] = [a.find('atom:name', ns).text for a in authors]

        summ = entry.find('atom:summary', ns)
        p['abstract'] = summ.text.strip().replace('\n', ' ') if summ is not None else ''

        pdf = entry.find("atom:link[@title='pdf']", ns)
        p['pdf_url'] = pdf.get('href') if pdf is not None else None
        return p

    def search_arxiv(self, query, max_results=5):
        try:
            params = {
                'search_query': f'all:{query}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            url = f"{self.base}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url) as resp:
                xml = resp.read()

            root = ET.fromstring(xml)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            papers = [self._parse(e) for e in entries]

            return {"success": True, "query": query, "count": len(papers), "papers": papers}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_paper_by_arxiv_id(self, arxiv_id):
        try:
            params = {'id_list': arxiv_id, 'max_results': 1}
            url = f"{self.base}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url) as resp:
                xml = resp.read()

            root = ET.fromstring(xml)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entry = root.find('atom:entry', ns)

            if entry is None:
                return {"success": False, "error": f"{arxiv_id} not found"}

            return {"success": True, "paper": self._parse(entry)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_by_author(self, author, max_results=5):
        try:
            params = {
                'search_query': f'au:{author}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            url = f"{self.base}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url) as resp:
                xml = resp.read()

            root = ET.fromstring(xml)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            papers = [self._parse(e) for e in entries]

            return {"success": True, "author": author, "count": len(papers), "papers": papers}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def search_recent_papers(self, topic, max_results=10):
        try:
            params = {
                'search_query': f'all:{topic}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'submittedDate',
                'sortOrder': 'descending'
            }
            url = f"{self.base}?{urllib.parse.urlencode(params)}"

            with urllib.request.urlopen(url) as resp:
                xml = resp.read()

            root = ET.fromstring(xml)
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', ns)
            papers = [self._parse(e) for e in entries]

            return {"success": True, "topic": topic, "count": len(papers), "papers": papers}
        except Exception as e:
            return {"success": False, "error": str(e)}
