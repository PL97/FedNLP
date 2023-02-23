import pandas as pd
import numpy as np
import os


def split_df_by_ratio(df, ratio=0.8):
    df_len = df.shape[0]
    df_1_len = int(ratio*df_len)
    idx = list(range(df_len))
    np.random.shuffle(idx)
    df_1 = df.iloc[idx[:df_1_len]]
    df_2 = df.iloc[idx[df_1_len:]]
    df_1.reset_index(drop=True, inplace=True)
    df_2.reset_index(drop=True, inplace=True)
    return df_1, df_2

def split_df_by_num(df, num=1):
    df_len = df.shape[0]
    df_1_len = num
    idx = list(range(df_len))
    np.random.shuffle(idx)
    df_1 = df.iloc[idx[:df_1_len]]
    df_2 = df.iloc[idx[df_1_len:]]
    df_1.reset_index(drop=True, inplace=True)
    df_2.reset_index(drop=True, inplace=True)
    return df_1, df_2

# TODO: make split the daasets in batch
if __name__ == "__main__":
    ds = "NCBI-disease"
    mode = "devel" # ! train | devel
    saved_name = "val" if mode == "devel" else mode
    df = pd.read_csv(os.path.join(ds, mode+".tsv"))
    # client_dfs, test_df = split_df(df, ratio=0.8)
    num_clients = 10
    client_size = int(df.shape[0]/num_clients)
    os.makedirs(f"{ds}/{num_clients}_split", exist_ok=True)
    for i in range(num_clients):
        if i != num_clients-1:
            client_df, df = split_df_by_num(df, client_size)
        else:
            client_df = df
        print(df.shape, client_df.shape)
        ## split into train, test, val
        client_df.to_csv(f"{ds}/{num_clients}_split/site-{i+1}_{saved_name}.csv")
    
    
    
    
    df = pd.read_csv(os.path.join(ds, mode+".tsv"))
    df.to_csv(f"{ds}/{num_clients}_split/site-0_{saved_name}.csv")
