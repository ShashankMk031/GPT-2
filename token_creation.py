import tiktoken
from datasets import load_dataset

#Loading OpenWebText data
dataset = load_dataset("openwebtext", split="train",streaming = True, trust_remote_code=True) #
text = '\n'.join(dataset["text"] for example in dataset)

#Tokenizer 
tokenzier = tiktoken.get_encoding("gpt2") 

#Convert text into tokens (Numerical representation)
tokens = tokenzier.encode( text, allowed_special = {"<|endoftext|>"})

print(f"Total tokens: {len(tokens)}")
print(f"Sample tokens: {tokens[ : 20]}") # Will display the first 20 tokens 