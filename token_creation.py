import tiktoken
# Load dataset in streaming mode
from datasets import load_dataset

dataset = load_dataset("openwebtext", split="train", streaming=True, trust_remote_code=True)

# Collect text data properly
text = '\n'.join(example["text"] for example in dataset)  # Correct way to extract text


#Tokenizer 
tokenzier = tiktoken.get_encoding("gpt2") 

#Convert text into tokens (Numerical representation)
tokens = tokenzier.encode( text, allowed_special = {"<|endoftext|>"})

print(f"Total tokens: {len(tokens)}")
print(f"Sample tokens: {tokens[ : 20]}") # Will display the first 20 tokens 