import pandas as pd
import numpy as np


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

if __name__ == "__main__":
    df = pd.read_csv("re-training-data.corp.csv")
    # client_dfs, test_df = split_df(df, ratio=0.8)
    num_clients = 10
    client_size = int(df.shape[0]/num_clients)
    import os
    os.makedirs(f"{num_clients}_split", exist_ok=True)
    for i in range(num_clients):
        if i != num_clients-1:
            client_df, df = split_df_by_num(df, client_size)
        else:
            client_df = df
        print(df.shape, client_df.shape)
        ## split into train, test, val
        client_df_train, client_df_val = split_df_by_ratio(client_df, ratio=0.9)
        client_df_train.to_csv(f"{num_clients}_split/site-{i+1}_train.csv")
        client_df_val.to_csv(f"{num_clients}_split/site-{i+1}_val.csv")
    
    
    
    
    df = pd.read_csv("re-training-data.corp.csv")
    # client_dfs, test_df = split_df(df, ratio=0.8)

    ## split into train, test, val
    client_df_0_train, client_df_0_val = split_df_by_ratio(df, ratio=0.9)



    client_df_0_train.to_csv(f"{num_clients}_split/site-0_train.csv")
    client_df_0_val.to_csv(f"{num_clients}_split/site-0_val.csv")
