import sqlite3
from pathlib import Path


class DatabaseTool:
    """queries the sqlite db for paper info"""

    def __init__(self, db_path="../phase1_data_preparation/outputs/papers.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"db not found: {db_path}")

    def _query(self, sql, params=()):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = [dict(r) for r in cur.fetchall()]
        conn.close()
        return rows

    def get_all_datasets(self, **kwargs):
        sql = """
            SELECT d.name, COUNT(DISTINCT pd.paper_id) as paper_count
            FROM datasets d
            LEFT JOIN paper_datasets pd ON d.id = pd.dataset_id
            GROUP BY d.name ORDER BY paper_count DESC
        """
        res = self._query(sql)
        return {"success": True, "count": len(res), "datasets": res}

    def get_all_vision_models(self, **kwargs):
        rows = self._query("SELECT paper_id, vision_encoder FROM papers WHERE vision_encoder != ''")

        counts = {}
        for row in rows:
            for v in row['vision_encoder'].split(', '):
                v = v.strip()
                if v:
                    counts[v] = counts.get(v, 0) + 1

        res = [{"vision_encoder": k, "paper_count": v} for k, v in sorted(counts.items(), key=lambda x: -x[1])]
        return {"success": True, "count": len(res), "vision_models": res}

    def get_all_robots(self, **kwargs):
        sql = """
            SELECT r.name, COUNT(DISTINCT pr.paper_id) as paper_count
            FROM robots r
            LEFT JOIN paper_robots pr ON r.id = pr.robot_id
            GROUP BY r.name ORDER BY paper_count DESC
        """
        res = self._query(sql)
        return {"success": True, "count": len(res), "robots": res}

    def get_all_hardware(self, **kwargs):
        sql = """
            SELECT h.name, h.type, COUNT(DISTINCT ph.paper_id) as paper_count
            FROM hardware h
            LEFT JOIN paper_hardware ph ON h.id = ph.hardware_id
            GROUP BY h.name, h.type ORDER BY paper_count DESC
        """
        res = self._query(sql)
        return {"success": True, "count": len(res), "hardware": res}

    def get_papers_by_year(self, **kwargs):
        sql = "SELECT paper_id, title, year, venue FROM papers ORDER BY year DESC, title"
        res = self._query(sql)
        return {"success": True, "count": len(res), "papers": res}

    def get_paper_metadata(self, paper_id):
        res = self._query("SELECT * FROM papers WHERE paper_id = ?", (paper_id,))
        if not res:
            return {"success": False, "error": f"paper {paper_id} not found"}
        return {"success": True, "paper": res[0]}

    def search_papers_by_dataset(self, dataset_name):
        sql = """
            SELECT DISTINCT p.paper_id, p.title, p.year, p.model_name
            FROM papers p
            JOIN paper_datasets pd ON p.paper_id = pd.paper_id
            JOIN datasets d ON pd.dataset_id = d.id
            WHERE d.name LIKE ?
        """
        res = self._query(sql, (f"%{dataset_name}%",))
        return {"success": True, "count": len(res), "papers": res}

    def get_database_overview(self, **kwargs):
        stats = {}
        stats['total_papers'] = self._query("SELECT COUNT(*) as c FROM papers")[0]['c']
        stats['total_datasets'] = self._query("SELECT COUNT(*) as c FROM datasets")[0]['c']
        stats['total_robots'] = self._query("SELECT COUNT(*) as c FROM robots")[0]['c']
        stats['total_hardware'] = self._query("SELECT COUNT(*) as c FROM hardware")[0]['c']
        return {"success": True, "stats": stats}
