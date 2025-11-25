class PromptBuilder:
    def __init__(self):
        self.system_instructions = """You are a research assistant that answers questions about robotics papers.
Use the tool results below to answer the user's question accurately.
If you don't find relevant information, say so."""
    
    def format_tool_results(self, results):
        """Convert tool results dict into readable text"""
        formatted = ""
        
        for tool_name, result in results.items():
            if not result.get('success'):
                continue
            
            formatted += f"\n--- {tool_name} ---\n"
            
            if 'datasets' in result:
                formatted += self._format_datasets(result['datasets'])
            elif 'vision_models' in result:
                formatted += self._format_vision_models(result['vision_models'])
            elif 'training_setups' in result:
                formatted += self._format_training(result['training_setups'])
            elif 'hardware' in result:
                formatted += self._format_hardware(result['hardware'])
            elif 'results' in result:
                formatted += self._format_search_results(result['results'])
            elif 'paper' in result:
                formatted += self._format_paper(result['paper'])
            else:
                formatted += str(result)
        
        return formatted
    
    def _format_datasets(self, datasets):
        text = "Datasets found:\n"
        for ds in datasets[:10]:
            text += f"- {ds['name']}: used in {ds['paper_count']} papers\n"
        return text
    
    def _format_vision_models(self, models):
        text = "Vision models found:\n"
        for m in models[:10]:
            text += f"- {m['model_name']}: used in {m['paper_count']} papers\n"
        return text
    
    def _format_training(self, setups):
        text = "Training setups:\n"
        for s in setups[:5]:
            text += f"\nPaper: {s.get('paper_id')}\n"
            if s.get('optimizer'):
                text += f"  Optimizer: {s['optimizer']}\n"
            if s.get('learning_rate'):
                text += f"  Learning rate: {s['learning_rate']}\n"
            if s.get('batch_size'):
                text += f"  Batch size: {s['batch_size']}\n"
        return text
    
    def _format_hardware(self, hardware):
        text = "Hardware found:\n"
        for h in hardware[:10]:
            text += f"- {h['hardware_name']}: used in {h['paper_count']} papers\n"
        return text
    
    def _format_search_results(self, results):
        text = "Relevant content:\n"
        for i, r in enumerate(results[:5], 1):
            text += f"\n{i}. From {r.get('paper_id')}:\n"
            text += f"{r.get('text', '')[:300]}...\n"
        return text
    
    def _format_paper(self, paper):
        text = f"Paper: {paper.get('title', paper.get('paper_id'))}\n"
        if 'year' in paper:
            text += f"Year: {paper['year']}\n"
        if 'abstract' in paper:
            text += f"Abstract: {paper['abstract'][:300]}...\n"
        return text
    
    def build_prompt(self, query, tool_results):
        """Build complete prompt for LLM"""
        formatted_results = self.format_tool_results(tool_results)
        
        prompt = f"""{self.system_instructions}

USER QUESTION: {query}

TOOL RESULTS:
{formatted_results}

Based on the above information, provide a clear and concise answer:"""
        
        return prompt

