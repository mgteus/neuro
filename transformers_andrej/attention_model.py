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



GRID_SCALER = 1
GRID_DEFINITION = 8

class CustomRunsDataset():
    def __init__(self, split='train', device='cpu',  context_len=10, file=3):
        # Carregar as colunas diretamente
        # df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15_100.gzip', columns=['x_pos', 'y_pos'])
        # df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15.gzip', columns=['x_pos', 'y_pos'])
        path = {0:r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15_100.gzip',
                 1:r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15.gzip',
                 2:r'/home/mgteus/workspace/neuro/transformers_andrej/train_run.gzip'}
        
        df = pd.read_parquet(path[file], columns=['x_pos', 'y_pos'])
        
        # Vectorização de arredondamento e conversão para tensor
        self.feature_array = np.column_stack([
            GRID_SCALER*np.round(df['x_pos'].values, GRID_DEFINITION), 
            GRID_SCALER*np.round(df['y_pos'].values, GRID_DEFINITION)
        ])
        self.device = device
        # Convertendo para tensor de float32
        self.feature_array = torch.tensor(self.feature_array, dtype=torch.float32)
        
        # Tamanho de treinamento e teste
        n = np.min([int(len(df) * 0.8), int(100_000)])
        
        # Dados para treino ou teste
        self.data = self.feature_array[:n] if split == 'train' else self.feature_array[n:]
        print('dataset sent to', device, 'with size ', len(self.data))
        self.data = self.data.to(device,)
        self.context_len = context_len

    # def __len__(self):
    #     # print(len(self.data))
    #     return len(self.data) - self.context_len

    # def __getitem__(self, idx):
    #     # Preparando o índice de entrada e saída
    #     x = self.data[idx: idx + self.context_len]
    #     y = self.data[idx + 1: idx + self.context_len + 1]

    #     # y = y.to(self.device)
    #     # x = x.to(self.device)
        
    #     return x, y


class RunsDataset(Dataset):
    def __init__(self, split='train',device='cpu',  context_len=10, ):
        # Carregar as colunas diretamente
        # df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15_100.gzip', columns=['x_pos', 'y_pos'])
        df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15.gzip', columns=['x_pos', 'y_pos'])
        
        # Vectorização de arredondamento e conversão para tensor
        self.feature_array = np.column_stack([
            GRID_SCALER*np.round(df['x_pos'].values, GRID_DEFINITION), 
            GRID_SCALER*np.round(df['y_pos'].values, GRID_DEFINITION)
        ])
        self.device = device
        # Convertendo para tensor de float32
        self.feature_array = torch.tensor(self.feature_array, dtype=torch.float32)
        
        # Tamanho de treinamento e teste
        n = int(len(df) * 0.8)
        
        # Dados para treino ou teste
        self.data = self.feature_array[:n] if split == 'train' else self.feature_array[n:]
        # print('dataset sent to', device)
        # self.data = self.data.to(device,)
        self.context_len = context_len

    def __len__(self):
        # print(len(self.data))
        return len(self.data) - self.context_len

    def __getitem__(self, idx):
        # Preparando o índice de entrada e saída
        x = self.data[idx: idx + self.context_len]
        y = self.data[idx + 1: idx + self.context_len + 1]

        # y = y.to(self.device)
        # x = x.to(self.device)
        
        return x, y

def get_dataloader(split, batch_size, context_len, device, num_workers=4, pin_memory=True):
    # Criando o dataset
    dataset = RunsDataset(split, device, context_len)
    # Criando o DataLoader com multiprocessamento e pin_memory para GPU
    dataloader = DataLoader(dataset
                            , batch_size=batch_size
                            , shuffle=True if split=='train' else False
                            , num_workers=num_workers
                            , pin_memory=True if device == 'cuda' else False
                            # , pin_memory_device=device if device == 'cuda' else None
                            , persistent_workers=True
                            , drop_last=True)
    

        # Retorna um gerador que produz xb, yb
    def generator():
        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)  # Desempacota os dados do batch
            yield xb, yb    # Gera os dados como um par

    return generator()
    # # Enviar os dados para o dispositivo correto (GPU ou CPU)
    # for x_batch, y_batch in dataloader:
    #     x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    #     yield x_batch, y_batch

def custom_dataloader(dataset, batch_size, device):
        # Verifique o tamanho total do dataset
    dataset_size = len(dataset.data)
    # print('dataset is on ', dataset.data.get_device())
    context_len = dataset.context_len
    max_iter = dataset_size // batch_size
    print(max_iter, 'expected iters with size', batch_size)
    iter_counter = 0
    # Garante que índices aleatórios não ultrapassem os limites do dataset
    max_start_idx = dataset_size - context_len - 1
    def generator():
        for _ in range(max_iter): 
            # Gere n índices aleatórios no intervalo permitido
            ix = torch.randint(len(dataset.data) - context_len -1, (batch_size,))
            x = torch.stack([dataset.data[i:i+context_len] for i in ix])
            y = torch.stack([dataset.data[i+1:i+context_len+1] for i in ix])

            # x, y = x.to(device), y.to(device)

            yield x, y
        
    return generator()

def load_data(split):
    df = pd.read_parquet(r'/home/mgteus/workspace/neuro/transformers_andrej/train_runs_15_100.gzip')
    feature_array = []
    for x_pos,y_pos in zip(df['x_pos'], df['y_pos']):
            feature_array.append(np.array([GRID_SCALER*np.round(x_pos, GRID_DEFINITION), GRID_SCALER*np.round(y_pos, GRID_DEFINITION)], dtype='double'))
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
    criterion = nn.MSELoss()
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
        # print(word_embeddings.get_device())
        # print(self.pe[:word_embeddings.size(0), :].get_device())
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
        # self.pos_to_enc_layer = nn.Linear(2, 2,)
        self.enc_layer = nn.Linear(2, 1)
        self.output_layer = nn.Linear(self.context_len, self.output_dim, bias=False)
        #   # dynamic layers
        self.key = nn.Linear(2, self.output_dim, bias=False)
        self.query = nn.Linear(2, self.output_dim, bias=False)
        self.values = nn.Linear(2, self.output_dim, bias=False)

        # tril
        self.register_buffer('tril', torch.tril(torch.ones(self.context_len, self.context_len)))

        # dropout
        self.dropout = nn.Dropout(self.dropout_value)


    def forward(self, x):
        # x = self.pos_to_enc_layer(x) # [x, y] -> [i, j]
        # x = self.enc_layer(x).squeeze(-1)        # [i, j] -> [k]

        B, C, _ = x.shape

        k = self.key(x)
        q =  self.query(x) #self.key(x)

        wei = q @ k.transpose(-2, -1) * self.output_dim**-0.5 # [B, C] @ [C, B] -> [B, B]
        wei = wei.masked_fill(self.tril[:C, :C] == 0, float('-inf'))
        wei = nn.functional.softmax(wei, dim=-1)
        # wei = self.dropout(wei)
        # print(wei.var(),)
        v = self.values(x)
        out = wei @ v # [B, B] @ [B, C] -> [B, C]
        
        # out = self.output_layer(out) # [B, C] -> [B, 2]

        return out

class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_heads, context_len, batch_size, dropout, head_output_dim):
        super().__init__()
        self.context_len = context_len
        self.batch_size = batch_size
        self.dropout_value = dropout
        self.num_heads = num_heads
        self.head_output_dim = head_output_dim
        # positional embedding
        self.pose = PositionEncoding(d_model=self.num_heads*self.head_output_dim, max_len=context_len)
        
        # self.linear_x = nn.Linear(3, 1, bias=False)
        # self.linear_y = nn.Linear(3, 1, bias=False)
        self.linear_both = nn.Linear(self.num_heads*self.head_output_dim, self.num_heads*self.head_output_dim, bias=False)
        self.dropout = nn.Dropout(self.dropout_value)

        # creating the multi heads
        self.heads = nn.ModuleList(
            [ Head(context_len=self.context_len
                    , batch_size=self.batch_size
                    , dropout=self.dropout_value
                    , output_dim=self.head_output_dim)
                for _ in range(num_heads)
                    ]
                    )
    def forward(self, x):
        output = torch.cat(
                [h(self.pose(x)) for h in self.heads]
                , dim=-1
                )
        output = self.linear_both(output)
        # output = self.dropout(output)
        return output

class FeedForwardLayer(nn.Module):
    def __init__(self, embed_size, dropout, inner_dim=0, ):
        self.embed_size = embed_size
        self.inner_dim = inner_dim if inner_dim > 0 else embed_size
        self.dropout_value = dropout
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(self.embed_size, self.inner_dim),
            nn.ReLU(),
            nn.Linear(self.inner_dim, self.embed_size),
            # nn.Dropout(self.dropout_value),
        )
    def forward(self, x):
        return self.layer(x)

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, context_len, batch_size, dropout, head_output_dim):
        super().__init__()
        self.num_heads = num_heads
        self.context_len = context_len
        self.batch_size = batch_size
        self.dropout_value = dropout
        self.head_output_dim = head_output_dim
        self.sa = MultiHeadAttention(num_heads=self.num_heads,
                                     context_len=self.context_len,
                                     batch_size=self.batch_size,
                                     dropout=self.dropout_value,
                                     head_output_dim=self.head_output_dim)
        
        self.ffwd = FeedForwardLayer(
                    embed_size=self.num_heads*self.head_output_dim
                    , dropout=self.dropout_value
                    , inner_dim=4*self.num_heads*self.head_output_dim)

        # self.lay_norm1 = nn.LayerNorm(self.num_heads*self.head_output_dim)
        # self.lay_norm2 = nn.LayerNorm(self.num_heads*self.head_output_dim)
    def forward(self, x):
        # x = x + self.sa(self.lay_norm1(x))
        # x = x + self.ffwd(self.lay_norm2(x))

        x = x + self.sa(x)
        x = x + self.ffwd(x)


        return x
    
class Transformers(nn.Module):
    def __init__(self, num_blocks, num_heads, context_len, batch_size, dropout, head_output_dim):
        super().__init__()
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.context_len = context_len
        self.batch_size = batch_size
        self.dropout_value = dropout
        self.head_output_dim = head_output_dim
        

        self.blocks = nn.ModuleList(
            [ TransformerBlock( num_heads = self.num_heads
                , context_len = self.context_len
                , batch_size = self.batch_size
                , dropout = self.dropout_value
                , head_output_dim=self.head_output_dim)
                for _ in range(self.num_blocks)
                    ]
                    )
        # self.blocks.append(nn.LayerNorm(self.num_heads*self.head_output_dim))

        self.net = nn.Sequential(*self.blocks)
        
    def forward(self, x):
        x = x + self.net(x)
        return x


if __name__ == '__main__':
    PATH = r'/home/mgteus/workspace/neuro/transformers_andrej/models/test_model.pt'
    print(datetime.now())
    CONTEXT_LEN = 128
    BATCH_SIZE = 1024
    DROPOUT = 0.2
    LEARNING_RATE = 1e-5
    NUM_HEADS = 2
    HEAD_SIZE = 1
    NUM_EPOCHS = 1e2
    NUM_BLOCKS = 2
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


    # model = Head(context_len=CONTEXT_LEN, batch_size=BATCH_SIZE, dropout=DROPOUT)
    # model = TransformerBlock( num_heads = NUM_HEADS
    #             , context_len = CONTEXT_LEN
    #             , batch_size = BATCH_SIZE
    #             , dropout = DROPOUT
    #             , head_output_dim=HEAD_SIZE)
    model = Transformers(num_blocks=NUM_BLOCKS,
                         num_heads = NUM_HEADS
                , context_len = CONTEXT_LEN
                , batch_size = BATCH_SIZE
                , dropout = DROPOUT
                , head_output_dim=HEAD_SIZE)
    
        # Model class must be defined somewhere
    model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Criar o DataLoader
    # dataloader = get_dataloader(split='train', batch_size=BATCH_SIZE, context_len=CONTEXT_LEN, device=DEVICE,)
    # val_dataloader = get_dataloader(split='test', batch_size=BATCH_SIZE, context_len=CONTEXT_LEN, device=DEVICE)
    # dataloader = get_dataloader(split='train', batch_size=BATCH_SIZE, context_len=CONTEXT_LEN, device=DEVICE,)

    dataset_train = CustomRunsDataset(split='train', device=DEVICE, context_len=CONTEXT_LEN, file=1)
    dataset_test = CustomRunsDataset(split='test', device=DEVICE, context_len=CONTEXT_LEN, file=2)

    train_loss = []
    batch_loss_list = []
    test_loss = []
    val_batch_loss_list = []
    epoch = 0
    for epoch in range(int(NUM_EPOCHS)):
        model.train()
        epoch_loss = 0
        counter = 1
        dataloader = custom_dataloader(dataset=dataset_train, batch_size=BATCH_SIZE, device=DEVICE)
        # dataloader = get_dataloader(split='train', batch_size=BATCH_SIZE, context_len=CONTEXT_LEN, device=DEVICE,)
        for xb, yb in dataloader:
            # xb = xb.to(DEVICE)
            # yb = yb.to(DEVICE)
            # print(xb.shape, yb.shape, counter)
            # xb, yb = get_batch2d(context_len=CONTEXT_LEN, batch_size=BATCH_SIZE, split='train', device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(xb)
            predictions = predictions.to(DEVICE)
            # print(predictions.shape)
            # print(yb.shape)
            loss = RMSELoss(predictions.view(BATCH_SIZE*CONTEXT_LEN, NUM_HEADS), yb.view(BATCH_SIZE*CONTEXT_LEN, NUM_HEADS))
            # print(loss)
            loss.backward()
            optimizer.step()
            loss_cpu = loss.cpu().detach().numpy()
            epoch_loss += loss_cpu
            # batch_loss_list.append(loss_cpu)
            counter+=1
        # print(counter, train_loss)
        train_loss.append(epoch_loss / counter)




        model.eval()
        val_epoch_loss = 0
        val_counter = 1
        with torch.no_grad():
            val_dataloader = custom_dataloader(dataset=dataset_test, batch_size=BATCH_SIZE, device=DEVICE)
            for xbt, ybt in val_dataloader:
                # xbt = xbt.to(DEVICE)
                # ybt = ybt.to(DEVICE)
                val_predictions = model(xbt)
                val_loss = RMSELoss(predictions.view(BATCH_SIZE*CONTEXT_LEN, NUM_HEADS), ybt.view(BATCH_SIZE*CONTEXT_LEN, NUM_HEADS))
                val_loss_cpu = val_loss.cpu().detach().numpy()
                val_epoch_loss += val_loss_cpu
                # val_batch_loss_list.append(val_loss_cpu)
                val_counter+=1
                
            test_loss.append(val_epoch_loss / val_counter)
        if epoch%(NUM_EPOCHS/10)==0:
            print(f"iter. {epoch:02d} - loss [train] = {np.mean(train_loss):4f} - loss [test]  = {np.mean(test_loss):4f}", datetime.now())
            # print(f"iter. {epoch:02d} - loss  = {np.mean(test_loss):4f} [test]", datetime.now())



    print(f"{len(train_loss)=}")
    print(f"{len(test_loss)=}")

    
    plt.plot(train_loss, label='train')
    plt.plot(test_loss, label='val')
    plt.ylabel(r'Loss ($\Delta$)')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()



    # plt.plot(batch_loss_list, label='train')
    # plt.plot(val_batch_loss_list, label='val')
    # plt.ylabel(r'Loss ($\Delta$)')
    # plt.xlabel('Batch')
    # plt.legend()
    # plt.show()

    save_model = input('salvar modelo?')
    if save_model == '1':
        torch.save(model.state_dict(), PATH)

        print('modelo salvo')


    
    # model.load_state_dict(torch.load(PATH, weights_only=True))
    # model.eval()
    
    # model = model.to(DEVICE)



        # model = MultiHeadAttention(
        #           num_heads = NUM_HEADS
        #         , context_len = CONTEXT_LEN
        #         , batch_size = BATCH_SIZE
        #         , dropout = DROPOUT
        #         , head_output_dim=HEAD_SIZE)