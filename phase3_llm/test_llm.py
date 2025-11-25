from llm_wrapper import LLMWrapper

print("="*60)
print("Testing LLM Wrapper")
print("="*60)

# Test with 8B model (smaller, faster)
print("\n1. Loading Llama-3.1-8B-Instruct...")
llm = LLMWrapper(model_size="8b")
llm.load()

# Test basic generation
print("\n2. Testing basic generation...")
print("-"*60)

test_prompts = [
    "What is machine learning?",
    "Explain robotics in one sentence.",
    "What does a vision encoder do?"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\nTest {i}: {prompt}")
    print("Answer:", llm.generate(prompt, max_tokens=100))
    print("-"*60)

print("\n✓ All tests passed!")

