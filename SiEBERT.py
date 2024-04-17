import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import json

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def load_data(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_dialog(dialog, start_percentage, end_percentage):
    if isinstance(dialog, list):
        seeker_contents = [item['content'] for item in dialog if item['speaker'] == 'seeker']
        start_index = int(start_percentage * len(seeker_contents))
        end_index = int(end_percentage * len(seeker_contents))
        return ' '.join(seeker_contents[start_index:end_index])
    elif isinstance(dialog, str):
        sentences = dialog.split('.')
        seeker_contents = [sentence for sentence in sentences] #[str, str, ..., str]
        start_index = int(start_percentage * len(seeker_contents))
        end_index = int(end_percentage * len(seeker_contents))
        return seeker_contents[start_index:end_index]
    else:
        return None

dataset = load_data("ESConv.json")
dataframe = pd.DataFrame(dataset)

df = pd.DataFrame()
df['dialog'] = dataframe['dialog'].apply(lambda x: extract_dialog(x, 0, 1)) #take whole dialog from seeker
df['dialog'] = df['dialog'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

df['initial_emotion_intensity'] = dataframe['survey_score'].apply(
    lambda x: x['seeker']['initial_emotion_intensity'])
df['initial_emotion_intensity'].dropna(inplace=True)
df['initial_emotion_intensity'] = df['initial_emotion_intensity'].astype(int)
df['dialog'].dropna(inplace=True)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters
batch_size = 32
max_length = 128
learning_rate = 2e-5
epochs = 3
num_labels = 2  # Define the number of labels according to your task

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('siebert/siembert-tiny-dutch-cased')
model = AutoModelForSequenceClassification.from_pretrained('siebert/siembert-tiny-dutch-cased', num_labels=num_labels).to(device)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# Prepare data
train_texts = df['dialog'].tolist()
train_labels = df['initial_emotion_intensity'].tolist()
train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define the training function
def train(train_loader, model, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss / len(train_loader)

# Training loop
for epoch in range(epochs):
    train_loss = train(train_loader, model, optimizer, criterion, device)
    print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

# Save the trained model
model.save_pretrained("path_to_save_model")

#%%
