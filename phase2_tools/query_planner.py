class QueryPlanner:
    """Analyzes queries and decides which tools to call"""
    
    def __init__(self):
        # Keyword patterns for routing
        self.patterns = {
            'datasets': ['dataset', 'data', 'training data', 'evaluation'],
            'vision_models': ['vision', 'encoder', 'visual', 'clip', 'dino'],
            'training': ['training', 'hyperparameter', 'optimizer', 'learning rate', 'batch size'],
            'hardware': ['hardware', 'robot', 'gpu', 'sensor', 'gripper', 'compute'],
            'year': ['year', 'when', 'published', 'recent'],
            'paper_details': ['paper', 'tell me about', 'details', 'what is'],
            'semantic': ['how', 'explain', 'compare', 'strategy', 'method', 'approach', 'similar'],
            'arxiv': ['find', 'search for', 'arxiv', 'recent papers', 'new papers', 'web', 'look up', 'search web', 'online'],
        }
    
    def analyze_query(self, query):
        """
        Analyze query and return what type it is
        Returns: list of matched categories
        """
        query_lower = query.lower()
        matches = []
        
        for category, keywords in self.patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    matches.append(category)
                    break
        
        return matches
    
    def plan(self, query):
        """
        Create execution plan for a query
        Returns dict with tools to call and their params
        """
        categories = self.analyze_query(query)
        plan = {
            'query': query,
            'categories': categories,
            'tools': []
        }
        
        # Route based on categories
        if 'datasets' in categories:
            if 'common' in query.lower() or 'all' in query.lower() or 'list' in query.lower():
                plan['tools'].append({
                    'name': 'get_all_datasets',
                    'params': {}
                })
            else:
                # Check if specific dataset mentioned
                plan['tools'].append({
                    'name': 'semantic_search',
                    'params': {'query': query, 'top_k': 5}
                })
        
        if 'vision_models' in categories:
            plan['tools'].append({
                'name': 'get_all_vision_models',
                'params': {}
            })
        
        if 'training' in categories:
            plan['tools'].append({
                'name': 'get_training_setups',
                'params': {}
            })
        
        if 'hardware' in categories:
            plan['tools'].append({
                'name': 'get_all_hardware',
                'params': {}
            })
        
        if 'year' in categories:
            plan['tools'].append({
                'name': 'get_papers_by_year',
                'params': {}
            })
        
        # Check for specific paper references
        paper_keywords = ['rt-1', 'rt-2', 'openvla', 'octo', 'rebot']
        for keyword in paper_keywords:
            if keyword in query.lower():
                paper_id = keyword.replace('-', '')
                plan['tools'].append({
                    'name': 'get_paper_metadata',
                    'params': {'paper_id': paper_id}
                })
                break
        
        # If semantic/conceptual question or no specific category matched
        if 'semantic' in categories or len(plan['tools']) == 0:
            plan['tools'].append({
                'name': 'semantic_search',
                'params': {'query': query, 'top_k': 5}
            })
        
        # Check if they want to search arXiv
        if 'arxiv' in categories or 'find' in query.lower() or 'search for' in query.lower():
            plan['tools'].append({
                'name': 'search_arxiv',
                'params': {'query': query, 'max_results': 5}
            })
        
        return plan
    
    def execute_plan(self, plan, tool_registry):
        """Execute the plan using tool registry"""
        results = {}
        
        for tool_spec in plan['tools']:
            tool_name = tool_spec['name']
            params = tool_spec['params']
            
            result = tool_registry.call_tool(tool_name, **params)
            results[tool_name] = result
        
        return results

