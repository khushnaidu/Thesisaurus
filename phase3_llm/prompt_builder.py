class PromptBuilder:
    def __init__(self):
        self.system_instructions = """You are a research assistant that answers questions about robotics papers.
Answer the question directly using the tool results provided.
Keep your answer concise and factual. 
Do not ask follow-up questions or generate additional Q&A examples."""
    
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
            name = ds.get('name') or ds.get('dataset_name', 'Unknown')
            paper_count = ds.get('paper_count', 0)
            text += f"- {name}: used in {paper_count} papers\n"
        return text
    
    def _format_vision_models(self, models):
        text = "Vision models found:\n"
        for m in models[:10]:
            model_name = m.get('model_name') or m.get('vision_encoder', 'Unknown')
            paper_count = m.get('paper_count', 0)
            text += f"- {model_name}: used in {paper_count} papers\n"
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
            hw_name = h.get('hardware_name') or h.get('robot_platform', 'Unknown')
            paper_count = h.get('paper_count', 0)
            text += f"- {hw_name}: used in {paper_count} papers\n"
        return text
    
    def _format_search_results(self, results):
        text = "Relevant content:\n"
        for i, r in enumerate(results[:5], 1):
            paper_id = r.get('paper_id', 'unknown paper')
            content = r.get('text', r.get('content', ''))[:300]
            text += f"\n{i}. From {paper_id}:\n{content}...\n"
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

Answer the user's question using only the information above. Be concise and direct. Do not ask follow-up questions or generate additional examples.

ANSWER:"""
        
        return prompt

