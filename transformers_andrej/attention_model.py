import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

class RunsDataset(Dataset):
    def __init__(self, split='train', context_len=10):
        # Carregar as colunas diretamente
        df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15.gzip', columns=['x_pos', 'y_pos'])
        
        # Vectorização de arredondamento e conversão para tensor
        self.feature_array = np.column_stack([
            np.round(df['x_pos'].values, 5), 
            np.round(df['y_pos'].values, 5)
        ])
        
        # Convertendo para tensor de float32
        self.feature_array = torch.tensor(self.feature_array, dtype=torch.float32)
        
        # Tamanho de treinamento e teste
        n = int(len(df) * 0.8)
        
        # Dados para treino ou teste
        self.data = self.feature_array[:n] if split == 'train' else self.feature_array[n:]
        self.context_len = context_len

    def __len__(self):
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        # Preparando o índice de entrada e saída
        x = self.data[idx: idx + self.context_len]
        y = self.data[idx + 1: idx + self.context_len + 1]
        
        return x, y

def get_dataloader(split, batch_size, context_len, device, num_workers=4, pin_memory=True):
    # Criando o dataset
    dataset = RunsDataset(split, context_len)
    
    # Criando o DataLoader com multiprocessamento e pin_memory para GPU
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=pin_memory
                            , pin_memory_device=device
                            , persistent_workers=True )
    
    # Enviar os dados para o dispositivo correto (GPU ou CPU)
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        yield x_batch, y_batch







def load_data(split):
    df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15.gzip')
    feature_array = []
    for x_pos,y_pos in zip(df['x_pos'], df['y_pos']):
            feature_array.append(np.array([np.round(x_pos, 5), np.round(y_pos, 5)], dtype='double'))
    feature_array = np.array(feature_array)
    feature_array = torch.from_numpy(feature_array)
    feature_array = feature_array.float()
    n = int(len(df)*0.8)
    return feature_array[:n] if split == 'train' else feature_array[n:]

def get_batch1d(context_len, batch_size, split, device):
    data = load_data(split=split)
    ix = torch.randint(len(data) - context_len -1, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+context_len+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def get_batch2d(context_len, batch_size, split, device):
    data = load_data(split=split)
    ix = torch.randint(len(data) - context_len -1, (batch_size,))
    x = torch.stack([data[i:i+context_len] for i in ix])
    y = torch.stack([data[i+1:i+context_len+1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y

def RMSELoss(pred, true):
    criterion = nn.MSELoss(reduction='sum')
    return torch.sqrt(criterion(pred, true))

class PositionEncoding(nn.Module):
    
    def __init__(self, d_model=2, max_len=6):
        
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        
        position = torch.arange(start=0, end=max_len, step=1).float().unsqueeze(1)
        embedding_index = torch.arange(start=0, end=d_model, step=2).float()
        
        div_term = 1/torch.tensor(10000.0)**(embedding_index / d_model)
        

        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        
        self.register_buffer('pe', pe) 

        
    def forward(self, word_embeddings):
        return word_embeddings + self.pe[:word_embeddings.size(0), :] 
    
    def return_positions(self, embeded_positions):
        return embeded_positions - self.pe[:embeded_positions.size(0), :]


class Head(nn.Module):
    def __init__(self, context_len, batch_size, dropout, output_dim) -> None:
        super().__init__()
        # parameters
        self.batch_size = batch_size
        self.context_len = context_len
        self.dropout_value = dropout
        self.output_dim = output_dim

        # layers
        #   # static layers
        self.pos_to_enc_layer = nn.Linear(2, 2,)
        self.enc_layer = nn.Linear(2, 1)
        self.output_layer = nn.Linear(self.context_len, self.output_dim)
        #   # dynamic layers
        self.key = nn.Linear(2, self.context_len)
        self.query = nn.Linear(2, self.context_len)
        self.values = nn.Linear(2, self.context_len)

        # tril
        self.register_buffer('tril', torch.tril(torch.ones(self.context_len, self.context_len)))

        # dropout
        self.dropout = nn.Dropout(self.dropout_value)


    def forward(self, x):
        # x = self.pos_to_enc_layer(x) # [x, y] -> [i, j]
        # x = self.enc_layer(x).squeeze(-1)        # [i, j] -> [k]

        B, C, _ = x.shape

        k = self.key(x)
        q = self.query(x)

        wei = q @ k.transpose(-2, -1)  # [B, C] @ [C, B] -> [B, B]
        wei = wei.masked_fill(self.tril[:C, :C] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)

        v = self.values(x)
        out = wei @ v # [B, B] @ [B, C] -> [B, C]
        
        out = self.output_layer(out) # [B, C] -> [B, 2]

        return out


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, context_len, batch_size, dropout, head_output_dim):
        super().__init__()
        self.context_len = context_len
        self.batch_size = batch_size
        self.dropout = dropout
        self.num_heads = num_heads
        self.head_output_dim = head_output_dim
        # positional embedding
        self.pose = PositionEncoding(d_model=2, max_len=context_len)
        
        # self.linear_x = nn.Linear(3, 1, bias=False)
        # self.linear_y = nn.Linear(3, 1, bias=False)
        self.linear_both = nn.Linear(self.num_heads*self.head_output_dim, 2, bias=False)

        # creating the multi heads
        self.heads = nn.ModuleList(
            [ Head(context_len=self.context_len
                    , batch_size=self.batch_size
                    , dropout=self.dropout
                    , output_dim=self.head_output_dim)
                for _ in range(num_heads)
                    ]
                    )
    def forward(self, x):
        embed_positions = self.pose(x)
        concat_table = torch.cat(
                [h(embed_positions) for h in self.heads]
                , dim=-1
                )
        output = self.linear_both(concat_table)
        # output = torch.empty(concat_table.shape[0], concat_table.shape[1], 2)

        # # Aplicar as transformações na última dimensão
        # for i in range(concat_table.shape[0]):  # Iterar sobre a primeira dimensão
        #     for j in range(concat_table.shape[1]):  # Iterar sobre a segunda dimensão
        #         even_input = concat_table[i, j, 0:3]  # Pega 3 elementos para índices pares
        #         odd_input = concat_table[i, j, 3:6]   # Pega 3 elementos para índices ímpares
                
        #         output[i, j, 0] = self.linear_x(even_input).item()  # Resultado para pares
        #         output[i, j, 1] = self.linera_y(odd_input).item()    # Resultado para ímpares

        return output



if __name__ == '__main__':
    CONTEXT_LEN = 4
    BATCH_SIZE = 8
    DROPOUT = 0.1
    LEARNING_RATE = 1e-4
    NUM_HEADS = 2
    NUM_EPOCHS = 1e2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    # model = Head(context_len=CONTEXT_LEN, batch_size=BATCH_SIZE, dropout=DROPOUT)
    model = MultiHeadAttention(
                  num_heads = NUM_HEADS
                , context_len = CONTEXT_LEN
                , batch_size = BATCH_SIZE
                , dropout = DROPOUT
                , head_output_dim=2)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Criar o DataLoader
    dataloader = get_dataloader(split='train', batch_size=BATCH_SIZE, context_len=CONTEXT_LEN, device=DEVICE)

 
    loss_list = []
    epoch = 0
    for xb, yb in dataloader:
        while epoch <= NUM_EPOCHS:
            xb, yb = get_batch2d(context_len=CONTEXT_LEN, batch_size=BATCH_SIZE, split='train', device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(xb)
            predictions = predictions.to(DEVICE)
            # print(predictions.shape)
            loss = RMSELoss(predictions, yb)
            # print(loss)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().numpy())
            if epoch%10==0:
                print(f"iter. {epoch} - loss = {loss.item():4f}", datetime.now())
            epoch+=1

    plt.plot(loss_list)
    plt.show()



