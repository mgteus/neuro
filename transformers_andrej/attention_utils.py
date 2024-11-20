import numpy
import torch
import torch.nn as nn
import pandas as pd

import numpy as np
from glob import glob 




def concat_runs(paths:list = [], final_path:str = ''):
    file_count = 0
    files_to_concat = []
    for file in glob(r'runs/15min/**.gzip'):
        if file_count < 30:
            files_to_concat.append(file)
            file_count+=1
    

    dfs_list = [pd.read_parquet(file) for file in files_to_concat]
    df = pd.concat(dfs_list, ignore_index=True)
    print(df.shape)
    df.to_parquet(r'transformers_andrej/train_runs_15.gzip')
    return




if __name__ == '__main__':

    # print(pd.read_parquet(r'transformers_andrej/train_run.gzip').shape)
    concat_runs()



