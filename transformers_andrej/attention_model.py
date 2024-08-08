import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(split):
    df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_run.gzip')
    feature_array = []
    for x_pos,y_pos in zip(df['x_pos'], df['y_pos']):
            feature_array.append(torch.tensor([x_pos, y_pos]))

    feature_array = np.array(feature_array)
    feature_array = torch.tensor(feature_array)
    n = int(len(df)*0.8)
    
    return feature_array[:n] if split == 'train' else feature_array[n:]

def get_batch(context_len, batch_size, split, ):
    data = load_data(split=split)
    ix = torch.randint(len(data) - context_len, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+context_len+1] for i in ix])
    return x, y


def RMSELoss(pred, true):
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(pred, true))



class Head(nn.Module):
    def __init__(self, context_len, batch_size, dropout) -> None:
        super().__init__()
        # parameters
        self.batch_size = batch_size
        self.context_len = context_len
        self.dropout_value = dropout
        # layers
        #   # static layers
        self.pos_to_enc_layer = nn.Linear(2, 2,)
        self.enc_layer = nn.Linear(2, 1)
        self.output_layer = nn.Linear(self.context_len, 2)
        #   # dynamic layers
        self.key = nn.Linear(self.context_len, self.context_len)
        self.query = nn.Linear(self.context_len, self.context_len)
        self.values = nn.Linear(self.context_len, self.context_len)

        # tril
        self.register_buffer('tril', torch.tril(torch.ones(self.batch_size, self.batch_size)))

        # dropout
        self.dropout = nn.Dropout(self.dropout_value)


    def forward(self, x):
        x = self.pos_to_enc_layer(x) # [x, y] -> [i, j]
        x = self.enc_layer(x).squeeze(-1)        # [i, j] -> [k]

        B, C = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)  # [B, C] @ [C, B] -> [B, B]
        #wei = wei.masked_fill(self.tril[:C, :C] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)

        v = self.values(x)
        out = wei @ v # [B, B] @ [B, C] -> [B, C]
        
        out = self.output_layer(out) # [B, C] -> [B, 2]

        return out



if __name__ == '__main__':

    CONTEXT_LEN = 10
    BATCH_SIZE = 4
    DROPOUT = 0.1
    LEARNING_RATE = 1e-3
    NUM_EPOCHS = 10

    model = Head(context_len=CONTEXT_LEN, batch_size=BATCH_SIZE, dropout=DROPOUT)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    


    loss_list = []

    for epoch in range(NUM_EPOCHS):
        xb, yb = get_batch(context_len=CONTEXT_LEN, batch_size=BATCH_SIZE, split='train')
        optimizer.zero_grad(set_to_none=True)
        predictions = model(xb)
        loss = RMSELoss(predictions, yb)
        loss_list.append(loss)
        loss.backward()
        optimizer.step()
        print(f"iter. {epoch} - loss = {loss.item():4f}")

    plt.plot(loss_list)
    plt.show()



