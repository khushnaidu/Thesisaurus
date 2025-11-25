import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class LLMWrapper:
    def __init__(self, model_size="8b"):
        """
        Load Llama model with 4-bit quantization
        model_size: "8b" for Llama-3.1-8B or "70b" for Llama-3.3-70B
        """
        if model_size == "8b":
            self.model_name = "meta-llama/Llama-3.1-8B-Instruct"
        elif model_size == "70b":
            self.model_name = "meta-llama/Llama-3.3-70B-Instruct"
        else:
            raise ValueError("model_size must be '8b' or '70b'")
        
        self.model_size = model_size
        self.model = None
        self.tokenizer = None
    
    def load(self):
        """Load model and tokenizer with 4-bit quantization"""
        print(f"Loading {self.model_name}...")
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            device_map="auto",
            torch_dtype=torch.float16
        )
        
        print(f"✓ Model loaded successfully")
    
    def generate(self, prompt, max_tokens=512, temperature=0.7):
        """Generate text from prompt"""
        if self.model is None:
            self.load()
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_length = len(self.tokenizer.decode(inputs['input_ids'][0]))
        answer = generated_text[prompt_length:].strip()
        
        return answer

