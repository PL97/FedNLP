import pandas as pd
import numpy as np

seed = 1997
df = pd.read_csv("ner.csv")

df = df.sample(frac=1, random_state=seed)
df.reset_index(drop=True, inplace=True)
df_site1, df_site2, df_test = np.split(df,
                            [int(.4 * len(df)), int(.8 * len(df))])

print(df_site1.shape, df_site2.shape, df_test.shape)
df_site1.reset_index(drop=True, inplace=True)
df_site2.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

## split train test val
df_site1 = df_site1.sample(frac=1, random_state=seed)
df_site1.reset_index(drop=True, inplace=True)
df_site1_train, df_site1_val, df_site1_test = np.split(df_site1,
                            [int(.8 * len(df_site1)), int(.9 * len(df_site1))])

df_site2 = df_site2.sample(frac=1, random_state=seed)
df_site2.reset_index(drop=True, inplace=True)
df_site2_train, df_site2_val, df_site2_test = np.split(df_site2,
                            [int(.8 * len(df_site2)), int(.9 * len(df_site2))])

df_site1_train.reset_index(drop=True, inplace=True)
df_site1_val.reset_index(drop=True, inplace=True)
df_site1_test.reset_index(drop=True, inplace=True)

df_site2_train.reset_index(drop=True, inplace=True)
df_site2_val.reset_index(drop=True, inplace=True)
df_site2_test.reset_index(drop=True, inplace=True)


df_site1_train.to_csv("site-1_train.csv")
df_site1_val.to_csv("site-1_val.csv")
df_site1_test.to_csv("site-1_test.csv")

df_site2_train.to_csv("site-2_train.csv")
df_site2_val.to_csv("site-2_val.csv")
df_site2_test.to_csv("site-2_test.csv")
df_test.to_csv("test.csv")
