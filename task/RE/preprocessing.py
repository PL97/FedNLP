import os
import re
from copy import deepcopy

### format sentence num, entity type, words position, POS tag, word
## remove special characters
## annotation bugs

'''
 1. one entity with multiple span
 2. multiple relations in one sentence
'''



file = "re-training-data.corp"
sentences_orig = []
sentences = []
labels = []
words = []
entity_map = {}
in_sentence = True
with open(file, 'r') as doc:
    lines = doc.readlines()
    for line in lines:
        line = re.sub("\n", "", line)       # remove \n 
        line_parse = re.split("[ \t]+", line)       # split by \t
            
                
        if len(line_parse) == 5:        # if this is a tag row
            words.append(line_parse[4])
            if line_parse[1] != 0:
                entity_map[int(line_parse[2])] = f"@{line_parse[1].upper()}$"
                
        else:
            if re.search("[0-9a-zA-Z]", "".join(words)):        # if is a valid sentence
                sentences.append(deepcopy(words))
                sentences_orig.append(deepcopy(words))
                words = []

            if len(line_parse) == 3:      # if this is a relation row
                if len(labels) == len(sentences):
                    sentences.append(deepcopy(sentences_orig[-1]))
                    sentences_orig.append(deepcopy(sentences_orig[-1]))
                
                while len(labels) < len(sentences)-1:
                    labels.append("No relation")
                
                
                labels.append(line_parse[2].upper())
                ## need to modify the sentence at the same time
                sentences[-1][int(line_parse[0])] = entity_map[int(line_parse[0])]
                sentences[-1][int(line_parse[1])] = entity_map[int(line_parse[1])]
                
                
                ## check the correctness of alignment
                tmp_entities = labels[-1].split("-")
                tmp_sentence = " ".join(sentences[-1])
                
                # try:
                #     assert re.search(f"[@]{tmp_entities[0]}[$]", tmp_sentence) and re.search(f"[@]{tmp_entities[1]}[$]", tmp_sentence)
                # except:
                #     print(tmp_entities, tmp_sentence)
                #     exit("errors")
                
        # if len(sentences) > 20:
        #     break

## post fix
while len(labels) < len(sentences):
    labels.append("No relation")
  

## convert list of words to sentence
sentences = [" ".join(x) for x in sentences]
sentences_orig = [" ".join(x) for x in sentences_orig]

print(len(sentences), len(sentences_orig), len(labels))

import pandas as pd
df = pd.DataFrame({"text": sentences, "relation": labels, "orig_text": sentences_orig})
df.to_csv(f"{file}.csv")
  

# for x, y in zip(sentences, labels):
#     print(" ".join(x), y)
            
