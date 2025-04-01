from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import re
from flask_cors import CORS
import os
import csv
from datetime import datetime

# Load word2idx mapping
with open("word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

# Load trained model
checkpoint = torch.load("final_model.pth", map_location=torch.device('cpu'))

# Constants
PAD_WORD = "<PAD>"
MAX_SENT_LEN = 16
MAX_SENT_NUM = 4
EMBED_DIM = 200
NUM_CLASSES = 4

class WordCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_filters=10, kernel_sizes=[2, 3, 4], padding_idx=0):
        super(WordCNN, self).__init__()
        self.trainable_embedding = nn.Embedding(vocab_size, 100, padding_idx=padding_idx)
        self.static_embedding = nn.Embedding(vocab_size, 100, padding_idx=padding_idx)
        self.static_embedding.weight.requires_grad = False
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, 200)) for k in kernel_sizes
        ])
        self.output_dim = num_filters * len(kernel_sizes)

    def forward(self, x):
        batch_size, num_sentences, sentence_len = x.shape
        embedded_trainable = self.trainable_embedding(x)
        embedded_static = self.static_embedding(x)
        embedded = torch.cat((embedded_trainable, embedded_static), dim=-1)
        embedded = embedded.view(batch_size * num_sentences, 1, sentence_len, -1)
        conv_outs = [F.max_pool1d(F.relu(conv(embedded)).squeeze(3), conv(embedded).size(2)).squeeze(2) for conv in self.convs]
        out = torch.cat(conv_outs, dim=1).view(batch_size, num_sentences, -1)
        return out

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_out):
        attn_weights = torch.softmax(self.score(torch.tanh(self.attn(lstm_out))).squeeze(2), dim=1)
        context_vector = torch.sum(lstm_out * attn_weights.unsqueeze(2), dim=1)
        return context_vector

class SentenceLSTMWithAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(SentenceLSTMWithAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context_vector = self.attention(lstm_out)
        return self.fc(context_vector)

# Initialize models
vocab_size = len(word2idx)
cnn_model = WordCNN(vocab_size, embed_dim=EMBED_DIM)
lstm_model = SentenceLSTMWithAttention(cnn_model.output_dim, 128, 2, NUM_CLASSES)

# Load trained weights
cnn_model.load_state_dict(checkpoint['cnn_model_state_dict'])
lstm_model.load_state_dict(checkpoint['lstm_model_state_dict'])
cnn_model.eval()
lstm_model.eval()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

def preprocess_text(text):
    text_clean = re.sub(r'[^\w\s]', ' ', text.lower()).split()
    chunks = []
    for i in range(0, len(text_clean), MAX_SENT_LEN):
        chunk = [word2idx.get(word, word2idx.get("<UNK>", 1)) for word in text_clean[i:i+MAX_SENT_LEN]]
        while len(chunk) < MAX_SENT_LEN:
            chunk.append(word2idx.get(PAD_WORD, 0))
        chunks.append(chunk)

    while len(chunks) < MAX_SENT_NUM:
        chunks.append([word2idx.get(PAD_WORD, 0)] * MAX_SENT_LEN)

    return torch.tensor(chunks[:MAX_SENT_NUM], dtype=torch.long).unsqueeze(0)

CATEGORIES = ['World News', 'Sports', 'Business', 'Science/Tech']

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    input_tensor = preprocess_text(text)
    
    with torch.no_grad():
        cnn_output = cnn_model(input_tensor)
        prediction = lstm_model(cnn_output)
        predicted_class = torch.argmax(prediction, dim=1).item()
    
    return jsonify({
        "category": CATEGORIES[predicted_class],
        "text": text
    })

@app.route('/predict-and-log', methods=['POST'])
def predict_and_log():
    data = request.get_json()
    text = data.get("text", "")
    
    if not text:
        return jsonify({"error": "No text provided"}), 400

    input_tensor = preprocess_text(text)
    
    with torch.no_grad():
        cnn_output = cnn_model(input_tensor)
        prediction = lstm_model(cnn_output)
        predicted_class = torch.argmax(prediction, dim=1).item()

    # Log prediction
    os.makedirs('logs', exist_ok=True)
    with open('logs/predictions.csv', 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if f.tell() == 0:  # If file is empty, write header
            writer.writerow(['timestamp', 'text', 'predicted_category'])
        writer.writerow([
            datetime.now().isoformat(),
            text,
            CATEGORIES[predicted_class]
        ])
    
    return jsonify({
        "category": CATEGORIES[predicted_class],
        "text": text
    })
import os
port = int(os.environ.get("PORT", 5000))

# if __name__ == '__main__':
#     app.run(host='0.0.0.0',port=port, debug=False)
