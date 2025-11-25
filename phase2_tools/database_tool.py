import sqlite3
from pathlib import Path


class DatabaseTool:
    """Query SQLite database for paper metadata"""
    
    def __init__(self, db_path="../phase1_data_preparation/outputs/papers.db"):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def _execute_query(self, query, params=()):
        """Run SQL query, return results as list of dicts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(query, params)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    
    def get_all_datasets(self):
        """Get all datasets with usage counts"""
        query = """
            SELECT d.name, COUNT(DISTINCT pd.paper_id) as paper_count
            FROM datasets d
            LEFT JOIN paper_datasets pd ON d.id = pd.dataset_id
            GROUP BY d.name
            ORDER BY paper_count DESC
        """
        results = self._execute_query(query)
        return {"success": True, "count": len(results), "datasets": results}
    
    def get_all_vision_models(self):
        """Get all vision encoders with usage counts"""
        query = """
            SELECT vision_encoder, COUNT(*) as paper_count
            FROM papers
            WHERE vision_encoder IS NOT NULL
            GROUP BY vision_encoder
            ORDER BY paper_count DESC
        """
        results = self._execute_query(query)
        return {"success": True, "count": len(results), "vision_models": results}
    
    def get_training_setups(self):
        """Get training hyperparameters for all papers"""
        query = """
            SELECT 
                paper_id, title, optimizer, learning_rate, 
                batch_size, epochs, augmentations, pretrained_weights
            FROM papers
            WHERE optimizer IS NOT NULL 
               OR learning_rate IS NOT NULL 
               OR batch_size IS NOT NULL
        """
        results = self._execute_query(query)
        return {"success": True, "count": len(results), "training_setups": results}
    
    def get_all_hardware(self):
        """Get all hardware (robots, sensors, GPUs) with usage counts"""
        query = """
            SELECT h.name, h.type, COUNT(DISTINCT ph.paper_id) as paper_count
            FROM hardware h
            LEFT JOIN paper_hardware ph ON h.id = ph.hardware_id
            GROUP BY h.name, h.type
            ORDER BY paper_count DESC
        """
        results = self._execute_query(query)
        return {"success": True, "count": len(results), "hardware": results}
    
    def get_papers_by_year(self):
        """Get all papers sorted by year and venue"""
        query = """
            SELECT paper_id, title, year, venue
            FROM papers
            ORDER BY year DESC, title
        """
        results = self._execute_query(query)
        return {"success": True, "count": len(results), "papers": results}
    
    def get_paper_metadata(self, paper_id):
        """Get full metadata for a specific paper"""
        query = "SELECT * FROM papers WHERE paper_id = ?"
        results = self._execute_query(query, (paper_id,))
        
        if not results:
            return {"success": False, "error": f"Paper '{paper_id}' not found"}
        
        return {"success": True, "paper": results[0]}
    
    def search_papers_by_dataset(self, dataset_name):
        """Find papers using a specific dataset"""
        query = """
            SELECT DISTINCT p.paper_id, p.title, p.year, p.model_name
            FROM papers p
            JOIN paper_datasets pd ON p.paper_id = pd.paper_id
            JOIN datasets d ON pd.dataset_id = d.id
            WHERE d.name LIKE ?
        """
        results = self._execute_query(query, (f"%{dataset_name}%",))
        return {"success": True, "count": len(results), "papers": results}
    
    def get_database_overview(self):
        """Get database statistics"""
        stats = {}
        result = self._execute_query("SELECT COUNT(*) as count FROM papers")
        stats['total_papers'] = result[0]['count']
        
        result = self._execute_query("SELECT COUNT(*) as count FROM datasets")
        stats['total_datasets'] = result[0]['count']
        
        result = self._execute_query("SELECT COUNT(*) as count FROM robots")
        stats['total_robots'] = result[0]['count']
        
        result = self._execute_query("SELECT COUNT(*) as count FROM papers WHERE model_size IS NOT NULL")
        stats['papers_with_model_size'] = result[0]['count']
        
        return {"success": True, "stats": stats}
