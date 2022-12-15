import pandas as pd
import numpy as np


def split_df(df, ratio=0.8):
    df_len = df.shape[0]
    df_1_len = int(ratio*df_len)
    idx = list(range(df_len))
    np.random.shuffle(idx)
    df_1 = df.iloc[idx[:df_1_len]]
    df_2 = df.iloc[idx[df_1_len:]]
    df_1.reset_index(drop=True, inplace=True)
    df_2.reset_index(drop=True, inplace=True)
    return df_1, df_2

if __name__ == "__main__":
    # df = pd.read_csv("ner.csv")
    # # client_dfs, test_df = split_df(df, ratio=0.8)
    # client_dfs = df
    # client_df_1, client_df_2 = split_df(client_dfs, ratio=0.5)

    # ## split into train, test, val
    # client_df_1_train, client_df_1_val = split_df(client_df_1, ratio=0.9)

    # client_df_2_train, client_df_2_val = split_df(client_df_2, ratio=0.9)



    # client_df_1_train.to_csv("site-1_train.csv")
    # client_df_1_val.to_csv("site-1_val.csv")

    # client_df_2_train.to_csv("site-2_train.csv")
    # client_df_2_val.to_csv("site-2_val.csv")
    # # test_df.to_csv("test.csv")
    
    
    
    
    df = pd.read_csv("ner.csv")
    # client_dfs, test_df = split_df(df, ratio=0.8)

    ## split into train, test, val
    client_df_1_train, client_df_1_val = split_df(df, ratio=0.9)



    client_df_1_train.to_csv("site-0_train.csv")
    client_df_1_val.to_csv("site-0_val.csv")
