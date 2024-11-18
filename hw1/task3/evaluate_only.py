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

# Create datasets and dataloaders
batch_size = 64

def sentence_to_indices(sentence, vocab):
    indices = [vocab.get(word, vocab['<unk>']) for word in sentence]
    return indices

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, embedding_matrix, dropout=0.5):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        embedded = self.dropout(self.embedding(src))
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_len, batch_first=True, enforce_sorted=False)
        outputs, (hidden, cell) = self.rnn(packed)
        return hidden, cell
    
class Attention(nn.Module):
    def __init__(self, hid_dim):
        super().__init__()
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Parameter(torch.rand(hid_dim))
        
    def forward(self, hidden, encoder_outputs, src_mask):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        energy = energy.permute(0, 2, 1)
        v = self.v.repeat(batch_size, 1).unsqueeze(1)
        energy = torch.bmm(v, energy).squeeze(1)
        energy.masked_fill_(src_mask == 0, -1e10)
        return F.softmax(energy, dim=1)

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
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        a = self.attention(hidden[-1], encoder_outputs, src_mask).unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        rnn_input = torch.cat((embedded, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        output = torch.cat((output.squeeze(1), weighted.squeeze(1), embedded.squeeze(1)), dim=1)
        prediction = self.fc_out(output)
        return prediction, hidden, cell, a.squeeze(1)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.device = device
        
    def create_src_mask(self, src):
        mask = (src != self.src_pad_idx).to(self.device)
        return mask
    
    def forward(self, src, src_len, trg, teacher_forcing_ratio=0.5):
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
        trg_input = [self.trg_vocab['<sos>']] + trg_indices
        trg_output = trg_indices + [self.trg_vocab['<eos>']]
        return torch.tensor(src_indices), torch.tensor(trg_input), torch.tensor(trg_output)

def collate_fn(batch):
    src_batch, trg_input_batch, trg_output_batch = zip(*batch)
    src_lens = [len(s) for s in src_batch]
    trg_lens = [len(s) for s in trg_input_batch]
    src_padded = nn.utils.rnn.pad_sequence(src_batch, padding_value=src_vocab['<pad>'], batch_first=True)
    trg_input_padded = nn.utils.rnn.pad_sequence(trg_input_batch, padding_value=trg_vocab['<pad>'], batch_first=True)
    trg_output_padded = nn.utils.rnn.pad_sequence(trg_output_batch, padding_value=trg_vocab['<pad>'], batch_first=True)
    return src_padded, trg_input_padded, trg_output_padded, src_lens, trg_lens

# Create embedding matrices
def create_embedding_matrix(vocab, embedding_model, embedding_dim):
    embedding_matrix = np.random.uniform(-0.1, 0.1, (len(vocab), embedding_dim))
    for word, idx in vocab.items():
        if word in embedding_model.wv:
            embedding_vector = embedding_model.wv[word]
            embedding_matrix[idx] = embedding_vector
    return torch.tensor(embedding_matrix, dtype=torch.float32)

val_dataset = TranslationDataset(val_src_sentences, val_trg_sentences, src_vocab, trg_vocab)
test_dataset = TranslationDataset(test_src_sentences, test_trg_sentences, src_vocab, trg_vocab)

val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Load pre-trained embeddings
embedding_model = gensim.models.Word2Vec.load('models/word2vec_embedding_256.model')

# Model parameters
INPUT_DIM = len(src_vocab)
OUTPUT_DIM = len(trg_vocab)
ENC_EMB_DIM = DEC_EMB_DIM = 256
HID_DIM = 256
N_LAYERS = 2
ENC_DROPOUT = DEC_DROPOUT = 0.5

# Create embedding matrices
src_embedding_matrix = create_embedding_matrix(src_vocab, embedding_model, ENC_EMB_DIM)
trg_embedding_matrix = create_embedding_matrix(trg_vocab, embedding_model, DEC_EMB_DIM)

# Initialize model
attn = Attention(HID_DIM)
enc = Encoder(INPUT_DIM, ENC_EMB_DIM, HID_DIM, N_LAYERS, src_embedding_matrix, ENC_DROPOUT)
dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, trg_embedding_matrix, attn, DEC_DROPOUT)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Seq2Seq(enc, dec, src_vocab['<pad>'], device).to(device)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_trg_outputs = []
    all_predicted_outputs = []
    smoothie = SmoothingFunction().method4
    
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

# Function to translate a single sentence
def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_length=50):
    model.eval()
    
    tokenizer = fugashi.Tagger()
    tokens = [word.surface for word in tokenizer(sentence)]
    
    src_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]
    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)
    src_len = torch.LongTensor([len(src_indices)]).to('cpu')

    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor, src_len)
        encoder_outputs, _ = model.encoder.rnn(nn.utils.rnn.pack_padded_sequence(model.encoder.embedding(src_tensor), src_len, batch_first=True, enforce_sorted=False))
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(encoder_outputs, batch_first=True)
        
        src_mask = model.create_src_mask(src_tensor)
        
        trg_idx = [trg_vocab['<sos>']]
        
        for _ in range(max_length):
            trg_tensor = torch.LongTensor([trg_idx[-1]]).to(device)
            
            with torch.no_grad():
                output, hidden, cell, _ = model.decoder(trg_tensor, hidden, cell, encoder_outputs, src_mask)
            
            pred_token = output.argmax(1).item()
            trg_idx.append(pred_token)

            if pred_token == trg_vocab['<eos>']:
                break
    
    trg_tokens = [trg_idx2word[i] for i in trg_idx]
    return trg_tokens[1:-1]  # Remove <sos> and <eos>


for i in [9, 10, 49, 50]:
# Load the trained model
    model.load_state_dict(torch.load(f'/data/align-anything/hantao/NLPDL/hw1/task3/models/50ep_default/model_{i}.pt'))

    # Criterion
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab['<pad>'])

    # Evaluate on validation set
    print(f"Evaluating on Epoch {i}...")
    val_loss, val_bleu, val_perplexity = evaluate(model, val_loader, criterion)
    print(f'Validation Loss: {val_loss:.3f} | Validation BLEU: {val_bleu:.4f} | Validation Perplexity: {val_perplexity:.2f}')

    test_loss, test_bleu, test_perplexity = evaluate(model, test_loader, criterion)
    print(f'Test Loss: {test_loss:.3f} | Test BLEU: {test_bleu:.4f} | Test Perplexity: {test_perplexity:.2f}')

    # Test the model on specific sentences
    test_sentences = [
        "私の名前は愛です",
        "昨日はお肉を食べません",
        "いただきますよう",
        "秋は好きです",
        "おはようございます"
    ]

    print("\nTesting specific sentences:")
    for sentence in test_sentences:
        translation = translate_sentence(sentence, src_vocab, trg_vocab, model, device)
        print(f"Japanese: {sentence}")
        print(f"English: {' '.join(translation)}\n")