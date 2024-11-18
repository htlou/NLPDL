# train_model.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gensim
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import fugashi

from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

import logging

log_file = 'train_model.log'
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

smoothie = SmoothingFunction().method4

import math
import nltk
nltk.download('punkt')

from tqdm import tqdm

# For reproducibility
torch.manual_seed(42)

# Load data
train_df = pd.read_csv('data/train.csv')
val_df = pd.read_csv('data/val.csv')
test_df = pd.read_csv('data/test.csv')
tokenizer = fugashi.Tagger()

# Tokenization functions
def tokenize_english(text):
    return text.lower().split()


def tokenize_japanese(text):
    return [word.surface for word in tokenizer(text)]

# Build vocabularies
from collections import Counter

# Prepare the data: tokenize, build vocabularies, convert to indices

def build_vocab(sentences, min_freq=1):
    counter = Counter()
    for sentence in sentences:
        counter.update(sentence)
    vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
    index = 4
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = index
            index +=1
    return vocab

# Tokenize and build vocabularies
def prepare_data(df):
    src_sentences = []
    trg_sentences = []
    for idx, row in df.iterrows():
        src_sentences.append(tokenize_japanese(row['jp']))
        trg_sentences.append(tokenize_english(row['en']))
    return src_sentences, trg_sentences

train_src_sentences, train_trg_sentences = prepare_data(train_df)
val_src_sentences, val_trg_sentences = prepare_data(val_df)
test_src_sentences, test_trg_sentences = prepare_data(test_df)

# Build vocabularies
src_vocab = build_vocab(train_src_sentences, min_freq=1)
trg_vocab = build_vocab(train_trg_sentences, min_freq=1)

# Build reverse vocabularies
src_idx2word = {idx:word for word, idx in src_vocab.items()}
trg_idx2word = {idx:word for word, idx in trg_vocab.items()}

# Function to convert sentence to indices
def sentence_to_indices(sentence, vocab):
    indices = [vocab.get(word, vocab['<unk>']) for word in sentence]
    return indices

# Prepare datasets
class TranslationDataset(Dataset):
    def __init__(self, src_sentences, trg_sentences, src_vocab, trg_vocab, max_len=50):
        self.src_sentences = src_sentences
        self.trg_sentences = trg_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_indices = sentence_to_indices(self.src_sentences[idx], self.src_vocab)
        trg_indices = sentence_to_indices(self.trg_sentences[idx], self.trg_vocab)
        # Add <sos> and <eos> tokens
        trg_input = [self.trg_vocab['<sos>']] + trg_indices
        trg_output = trg_indices + [self.trg_vocab['<eos>']]
        return torch.tensor(src_indices), torch.tensor(trg_input), torch.tensor(trg_output)

# Collate function to pad batches
def collate_fn(batch):
    src_batch, trg_input_batch, trg_output_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    trg_lens = [len(s) for s in trg_input_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    trg_input_padded = nn.utils.rnn.pad_sequence(trg_input_batch, padding_value=trg_vocab['<pad>'], batch_first=True)
    trg_output_padded = nn.utils.rnn.pad_sequence(trg_output_batch, padding_value=trg_vocab['<pad>'], batch_first=True)
    return src_padded, trg_input_padded, trg_output_padded, src_lens, trg_lens

# Create datasets and dataloaders
batch_size = 64

train_dataset = TranslationDataset(train_src_sentences, train_trg_sentences, src_vocab, trg_vocab)
val_dataset = TranslationDataset(val_src_sentences, val_trg_sentences, src_vocab, trg_vocab)
test_dataset = TranslationDataset(test_src_sentences, test_trg_sentences, src_vocab, trg_vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Load pre-trained embeddings
embedding_model = gensim.models.Word2Vec.load('models/word2vec_embedding_256.model')

# Create embedding matrices
def create_embedding_matrix(vocab, embedding_model, embedding_dim):
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in embedding_model.wv:
            embedding_vector = embedding_model.wv[word]
            embedding_matrix[idx] = embedding_vector
    return torch.tensor(embedding_matrix, dtype=torch.float32)

embedding_dim = 256  # same as vector_size in word2vec

src_embedding_matrix = create_embedding_matrix(src_vocab, embedding_model, embedding_dim)
trg_embedding_matrix = create_embedding_matrix(trg_vocab, embedding_model, embedding_dim)

# Define the model: Encoder, Decoder with attention
class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, embedding_matrix, dropout=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        # src: [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        # outputs is packed, we don't need it for attention
        # hidden, cell: [n_layers, batch_size, hid_dim]
        return hidden, cell

class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))
        
    def forward(self, hidden, encoder_outputs, src_mask):
        # hidden: [batch_size, hid_dim]
        # encoder_outputs: [batch_size, src_len, hid_dim * n_directions]
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # Repeat hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hid_dim]
        energy = energy.permute(0, 2, 1)  # [batch_size, hid_dim, src_len]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [batch_size, 1, hid_dim]
        energy = torch.bmm(v, energy).squeeze(1)  # [batch_size, src_len]
        energy.masked_fill_(src_mask == 0, -1e10)
        return F.softmax(energy, dim=1)  # [batch_size, src_len]

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, embedding_matrix, attention, dropout=0.5):
        super().__init__()
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.attention = attention
        self.rnn = nn.LSTM((hid_dim * 1) + emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.fc_out = nn.Linear(hid_dim * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell, encoder_outputs, src_mask):
        # input: [batch_size]
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))  # [batch_size, 1, emb_dim]
        a = self.attention(hidden[-1], encoder_outputs, src_mask).unsqueeze(1)  # [batch_size, 1, src_len]
        # encoder_outputs: [batch_size, src_len, hid_dim * n_directions]
        weighted = torch.bmm(a, encoder_outputs)  # [batch_size, 1, hid_dim * n_directions]
        rnn_input = torch.cat((embedded, weighted), dim=2)  # [batch_size, 1, emb_dim + hid_dim * n_directions]
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # output: [batch_size, 1, hid_dim]
        output = torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1)  # [batch_size, hid_dim * 2 + emb_dim]
        prediction = self.fc_out(output)  # [batch_size, output_dim]
        return prediction, hidden, cell, a.squeeze(1)

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_src_mask(self, src):
        # src: [batch_size, src_len]
        mask = (src != self.src_pad_idx).to(self.device)
        return mask  # [batch_size, src_len]
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
        # src: [batch_size, src_len]
        # trg: [batch_size, trg_len]
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        hidden, cell = self.encoder(src, src_len)
        encoder_outputs, _ = self.encoder.rnn(nn.utils.rnn.pack_padded_sequence(self.encoder.embedding(src), src_len, batch_first=True, enforce_sorted=False))
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
        
        src_mask = self.create_src_mask(src)
        
        input = trg[:,0]
        
        for t in range(1, trg_len):
            output, hidden, cell, _ = self.decoder(input, hidden, cell, encoder_outputs, src_mask)
            outputs[:, t] = output
            top1 = output.argmax(1)
            input = trg[:, t] if np.random.random() < teacher_forcing_ratio else top1
        return outputs

# Initialize model
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)
ENC_EMB_DIM = embedding_dim
DEC_EMB_DIM = embedding_dim
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

# print(src_embedding_matrix.shape)
# print(trg_embedding_matrix.shape)
# print(INPUT_DIM)
# print(OUTPUT_DIM)
# print(ENC_EMB_DIM)
# print(DEC_EMB_DIM)

attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, src_embedding_matrix, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, trg_embedding_matrix, attn, DEC_DROPOUT)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'

model = Seq2Seq(enc, dec, src_vocab['<pad>'], device).to(device)

# Define optimizer and criterion
optimizer = optim.Adam(model.parameters())

criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab['<pad>'])

# Training loop
def train(model, iterator, optimizer, criterion, clip):
    model.train()
    epoch_loss = 0
    for i, (src, trg_input, trg_output, src_len, trg_len) in tqdm(enumerate(iterator), desc="Training", total=len(iterator)):
        src = src.to(device)
        trg_input = trg_input.to(device)
        trg_output = trg_output.to(device)
        src_len = torch.tensor(src_len).to('cpu')
        optimizer.zero_grad()
        output = model(src, src_len, trg_input)
        # output: [batch_size, trg_len, output_dim]
        output_dim = output.shape[-1]
        output = output[:,1:,:].reshape(-1, output_dim)
        trg_output = trg_output[:,1:].reshape(-1)
        loss = criterion(output, trg_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation loop
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_trg_outputs = []
    all_predicted_outputs = []
    with torch.no_grad():
        for i, (src, trg_input, trg_output, src_len, trg_len) in tqdm(enumerate(iterator), desc="Evaluating", total=len(iterator)):
            src = src.to(device)
            trg_input = trg_input.to(device)
            trg_output = trg_output.to(device)
            src_len = torch.tensor(src_len).to('cpu')
            output = model(src, src_len, trg_input, teacher_forcing_ratio=0)
            output_dim = output.shape[-1]
            output = output[:,1:,:]
            # For BLEU calculation
            for idx in range(output.shape[0]):
                pred_tokens = output[idx].argmax(1).cpu().numpy()
                trg_tokens = trg_output[idx,1:].cpu().numpy()
                trg_sentence = [trg_idx2word[idx] for idx in trg_tokens if idx != trg_vocab['<pad>'] and idx != trg_vocab['<eos>']]
                pred_sentence = [trg_idx2word[idx] for idx in pred_tokens if idx != trg_vocab['<pad>'] and idx != trg_vocab['<eos>']]
                all_trg_outputs.append([trg_sentence])
                all_predicted_outputs.append(pred_sentence)
            trg_output = trg_output[:,1:].reshape(-1)
            output = output.reshape(-1, output_dim)
            loss = criterion(output, trg_output)
            epoch_loss += loss.item()
    bleu_score = corpus_bleu(all_trg_outputs, all_predicted_outputs, smoothing_function=smoothie)
    perplexity = math.exp(epoch_loss / len(iterator))
    return epoch_loss / len(iterator), bleu_score, perplexity

N_EPOCHS = 50
CLIP = 1

best_valid_loss = float('inf')

for epoch in tqdm(range(N_EPOCHS), desc="Epochs"):
    train_loss = train(model, train_loader, optimizer, criterion, CLIP)
    valid_loss, valid_bleu, valid_perplexity = evaluate(model, val_loader, criterion)
    logging.info(f'Epoch: {epoch+1}')
    logging.info(f'\tTrain Loss: {train_loss:.3f}')
    logging.info(f'\t Val. Loss: {valid_loss:.3f} |  Val. BLEU: {valid_bleu:.2f} | Val. Perplexity: {valid_perplexity:.2f}')
    # Save the model if validation loss decreases
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        
    torch.save(model.state_dict(), f'models/50ep_default/model_{epoch+1}.pt')

# Evaluate on test set
test_loss, test_bleu, test_perplexity = evaluate(model, test_loader, criterion)
logging.info(f'Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.2f} | Test Perplexity: {test_perplexity:.2f}')

# Show predictions on the following test case
test_sentence = 'こんにちは'  # "Hello" in Japanese

test_sentence_tokens = tokenize_japanese(test_sentence)
test_sentence_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in test_sentence_tokens]
test_sentence_tensor = torch.tensor(test_sentence_indices, dtype=torch.long).unsqueeze(0).to(device)
test_sentence_length = torch.tensor([len(test_sentence_indices)]).to('cpu')

model.eval()
with torch.no_grad():
    hidden, cell = model.encoder(test_sentence_tensor, test_sentence_length)
    encoder_outputs, _ = model.encoder.rnn(nn.utils.rnn.pack_padded_sequence(model.encoder.embedding(test_sentence_tensor), test_sentence_length, batch_first=True, enforce_sorted=False))
    encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
    src_mask = model.create_src_mask(test_sentence_tensor)
    input_token = torch.tensor([trg_vocab['<sos>']], device=device)
    generated_tokens = []
    for t in range(50):
        output, hidden, cell, _ = model.decoder(input_token, hidden, cell, encoder_outputs, src_mask)
        top1 = output.argmax(1)
        if top1.item() == trg_vocab['<eos>']:
            break
        else:
            generated_tokens.append(trg_idx2word[top1.item()])
        input_token = top1
    print('Japanese input:', test_sentence)
    print('English translation:', ' '.join(generated_tokens))
    logging.info(f'Japanese input: {test_sentence}')
    logging.info(f'English translation: {" ".join(generated_tokens)}')