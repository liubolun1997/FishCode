import torch
import torch.nn as nn


class BiLSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(BiLSTMFeatureExtractor, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # 取最后一个时间步的双向输出 concat
        return torch.cat((lstm_out[:, -1, :self.lstm.hidden_size], lstm_out[:, 0, self.lstm.hidden_size:]), dim=1)
