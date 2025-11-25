"""
Example: How to use Phase 2 tools
Shows the complete flow: Query → Planner → Tools → Results
"""

from query_planner import QueryPlanner
from tool_registry import ToolRegistry

# Initialize
planner = QueryPlanner()
registry = ToolRegistry()

# Example query
query = "What are the most common datasets?"

print("User Query:", query)
print("\n" + "="*60)

# Step 1: Planner analyzes query
plan = planner.plan(query)
print(f"Planner says: Use {[t['name'] for t in plan['tools']]}")

# Step 2: Execute tools
print("\nExecuting tools...")
results = planner.execute_plan(plan, registry)

# Step 3: Display results
for tool_name, result in results.items():
    if result['success']:
        print(f"\n{tool_name} returned:")
        if 'datasets' in result:
            for ds in result['datasets'][:5]:
                print(f"  - {ds['name']}: {ds['paper_count']} papers")

print("\n" + "="*60)
print("This is what gets sent to the LLM for final answer generation!")

