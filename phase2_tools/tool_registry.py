from .database_tool import DatabaseTool
from .vector_search_tool import VectorSearchTool
from .web_search_tool import WebSearchTool


class ToolRegistry:
    """holds all the tools and calls them"""

    def __init__(self, db_path=None, index_path=None, metadata_path=None):
        self.db = DatabaseTool(db_path) if db_path else DatabaseTool()
        self.vs = VectorSearchTool(index_path, metadata_path) if index_path else VectorSearchTool()
        self.web = WebSearchTool()

        # map names to functions
        self.tools = {
            # db
            "get_all_datasets": self.db.get_all_datasets,
            "get_all_vision_models": self.db.get_all_vision_models,
            "get_training_setups": self.db.get_training_setups,
            "get_all_hardware": self.db.get_all_hardware,
            "get_papers_by_year": self.db.get_papers_by_year,
            "get_paper_metadata": self.db.get_paper_metadata,
            "search_papers_by_dataset": self.db.search_papers_by_dataset,
            "get_database_overview": self.db.get_database_overview,
            # vector
            "semantic_search": self.vs.search,
            "search_within_paper": self.vs.search_within_paper,
            "get_paper_chunks": self.vs.get_paper_chunks,
            # web
            "search_arxiv": self.web.search_arxiv,
            "get_arxiv_paper": self.web.get_paper_by_arxiv_id,
            "search_by_author": self.web.search_by_author,
            "search_recent_papers": self.web.search_recent_papers,
        }

    def call_tool(self, name, **kwargs):
        if name not in self.tools:
            return {"success": False, "error": f"unknown tool: {name}"}
        try:
            return self.tools[name](**kwargs)
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_tools(self):
        return list(self.tools.keys())
