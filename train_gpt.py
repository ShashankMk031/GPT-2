import torch 
import torch.nn as nn
import torch.optim as optim 
from  torch.utils.data import DataLoader, Dataset 
from datasets import load_dataset 
import tiktoken 
from model import GPT 

# Hyperparameters 
BATCH_SIZE = 8 
BLOCK_SIZE = 128 
EMBED_SIZE = 256 
NUM_HEADS = 4 
NUM_LAYERS = 6 
EPOCHS = 3 
LEARNING_RATE = 3e-4 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#Load dataset 
dataset =  load_dataset('openwebtext', split= 'train', streaming = True , trust_remote_code = True)

#Tokenizer 
tokenizer = tiktoken.get_encoding('gpt2')

#Dataset class 
class TextDataset(Dataset):
    def __init__(self, dataset):
        self.examples = [] 
        for example in dataset:
            tokens = tokenizer.encode(example["text"], allowed_special={"<|endoftext|>"})
            for i in range(0, len(tokens) - BLOCK_SIZE, BLOCK_SIZE):
                self.examples.append(tokens[i : i+BLOCK_SIZE])
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        x = torch.tensor(self.examples[idx][:-1], dtype = torch.long)
        y = torch.tensor(self.examples[idx][1:], dtype=torch.long)
        return x , y
    
#Prepare DataLoader 
train_dataset = TextDataset(dataset)
train_loader = DataLoader(train_dataset, batch_size= BATCH_SIZE, shuffle= True)

#Initialize model 
VOCAB_SIZE = tokenizer.n_vocab 
model = GPT(VOCAB_SIZE, EMBED_SIZE, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(DEVICE)

#Loss and Optimizer 
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr = LEARNING_RATE)

#Trainig Loop 
for epoch in range(EPOCHS):
    for batch_idx, (x, y) in enumerate(train_loader):
        x , y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(x) 
        loss = criterion(logits.view(-1, VOCAB_SIZE), y.view(-1))
        
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Epoch [ {epoch +1}/{EPOCHS}], Step [{batch_idx}], Loss: {loss.item():.4f}")
            
            