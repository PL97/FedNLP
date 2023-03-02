import pandas as pd

df = pd.read_csv("test.corp.csv")
# df = df.sample(500)
print(df.head)
relations = df.relation.to_list()


# label_map = {r: i for i, r in enumerate(set(relations))}
# ## sort the map based on alphabet order (increasing)
# label_map = dict(sorted(label_map.items(), key=lambda item: item[0]))
# print(label_map)
# import json
# with open("label_map.json", "w") as outfile:
#     json.dump(label_map, outfile)

# exit("finished")


## what below is to generate some plot to analysis the statistics of the data distribution

import seaborn as sns
import matplotlib.pyplot as plt
# plt.figure(figsize=(15,8))
ax = sns.displot(df, x='relation')
ax.fig.set_figwidth(10)
ax.fig.set_figheight(5)
ax.set_xticklabels(rotation=30)
plt.tight_layout()
plt.savefig("relation_distribution_test")
plt.close()


import re
import math
df['text_length'] = [math.log2(len(re.split(" +", re.sub("[^0-9a-zA-Z]", " ", x).strip()))) for x in df.text]
ax = sns.displot(df, x='text_length', hue='relation', kind='kde')
ax.fig.suptitle(f"min: {int(math.pow(2, df['text_length'].min()))}   max:{int(math.pow(2, df['text_length'].max()))} (x-axis is displayed in log scale)")
# plt.tight_layout()
plt.savefig("sentence_length_test")
