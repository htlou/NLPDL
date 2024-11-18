import gensim
import pandas as pd
from sklearn.model_selection import train_test_split
import json

# Load and process dataset
df = pd.read_csv('data/eng_jpn.txt', delimiter='\t', header=None, names=['jp', 'en'])
train_df, val_test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=42)
# save three datasets
train_df.to_csv('data/train.csv', index=False)
val_df.to_csv('data/val.csv', index=False)
test_df.to_csv('data/test.csv', index=False)

# Tokenization
def tokenize_english(text):
    return text.lower().split()

# Use fugashi as the Japanese tokenizer
import fugashi
tokenizer = fugashi.Tagger()

def tokenize_japanese(text):
    output = []
    for word in tokenizer(text):
        output.append(word.surface)
    # print(output)
    return output

# Create corpus for embedding training
train_corpus = []
for en, jp in zip(train_df['en'], train_df['jp']):
    train_corpus.append(tokenize_english(en))
    train_corpus.append(tokenize_japanese(jp))

# Train word embeddings using Gensim Word2Vec
model = gensim.models.Word2Vec(train_corpus, vector_size=256, window=10, min_count=1, sg=1)
model.save('models/word2vec_embedding_256.model')

# Evaluate embeddings
print("Most similar words to 'こんにちは' in Japanese:")
print(model.wv.most_similar(tokenize_japanese('こんにちは'), topn=5))
print("\nMost similar words to 'hello' in English:")
print(model.wv.most_similar(tokenize_english('hello'), topn=5))