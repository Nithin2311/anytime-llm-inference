from datasets import load_dataset
from early_exit_model import EarlyExitTinyLlama
from dynamic_scheduler import generate_stateless_anytime

def run_pubmed_benchmark():
    print("Loading PubMedQA Dataset...")
    # Load a small subset of the training data
    dataset = load_dataset("pubmed_qa", "pqa_labeled", split="train[:3]")
    
    model = EarlyExitTinyLlama()
    
    # We will test a 45.0 ms deadline. 
    # This is tight enough to stress the model, but generous enough 
    # that it shouldn't force a Layer 16 exit on every single token.
    test_deadline = 45.0 
    
    print("\n" + "#"*60)
    print("STARTING CLINICAL DOMAIN BENCHMARK")
    print("#"*60)
    
    for i, item in enumerate(dataset):
        print(f"\n[ Clinical Query {i+1} / 3 ]")
        
        # Extract the medical context and the question
        context = item['context']['contexts'][0]
        question = item['question']
        
        # Format the prompt
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        print(f"Prompt length: {len(prompt.split())} words")
        
        # Run the generation
        generate_stateless_anytime(model, prompt, max_new_tokens=15, deadline_ms=test_deadline)
        print("-" * 60)

if __name__ == "__main__":
    run_pubmed_benchmark()