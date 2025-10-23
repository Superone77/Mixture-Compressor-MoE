import os
import torch
import logging
from mixtral_model.modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer, AutoConfig
from datautils import get_loaders
from eval_ppl_utils import llama_eval

def test_full_precision_ppl(model_path, dataset="wikitext2", batch_size=1):
    """Test full precision Mixtral-8x7B model perplexity on wikitext2"""
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = MixtralForCausalLM.from_pretrained(
        model_path, 
        config=config, 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Add seqlen attribute for compatibility with llama_eval
    model.seqlen = 2048
    
    # Get data loader
    print(f"Loading {dataset} dataset...")
    dataloader, testloader = get_loaders(dataset, seed=0, seqlen=2048)
    
    # Evaluate perplexity
    print("Evaluating perplexity...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Create a wrapper to handle Mixtral-specific parameters
    class MixtralWrapper:
        def __init__(self, model):
            self.model = model
            self.seqlen = 2048
            self.config = model.config
            
        def __call__(self, input_ids, attention_mask=None, **kwargs):
            # Generate position_ids for Mixtral
            batch_size, seq_len = input_ids.shape
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            
            return self.model(input_ids, attention_mask=attention_mask, position_ids=position_ids)
    
    # Wrap the model
    wrapped_model = MixtralWrapper(model)
    llama_eval(wrapped_model, testloader, device, dataset)
    
    print(f"Full precision Mixtral-8x7B evaluation on {dataset} completed")
    
    return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to Mixtral-8x7B model")
    parser.add_argument("--dataset", type=str, default="wikitext2", help="Dataset to test on")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    
    args = parser.parse_args()
    
    test_full_precision_ppl(args.model_path, args.dataset, args.batch_size)
