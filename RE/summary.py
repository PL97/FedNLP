import json
import numpy as np
import os
from collections import defaultdict
import re

def parse_json(my_json, verbose=False):
    ret_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for metric in ['precision', 'recall', 'f1-score']:
        for mode in ['macro avg', 'weighted avg']:
            # try:
            #     ret_dict[mode][metric] = my_json['test']['1'][metric]
            # except:
            ret_dict[mode][metric] = my_json['test'][mode][metric]
    return ret_dict

def parse_json_multi(my_json, verbose=False):
    ret_dict = defaultdict(lambda: defaultdict(lambda: 0))
    print(my_json)
    for metric in ['precision', 'recall', 'f1-score']:
        for mode in ['macro avg']:
            ret_dict[mode][metric] = my_json['test'][mode][metric]
    return ret_dict
            
        
workspaces = ['workspace1', 'workspace2', 'workspace3']  
for pd, d, f in os.walk(workspaces[0]):
    if "evaluation.json" in f:
        print(pd)
        # try:
        tmp_dicts = []
        for ws in workspaces:
            json_path = os.path.join(re.sub(workspaces[0], ws, pd), "evaluation.json")
            my_json = json.load(open(json_path))
            tmp_dict = parse_json(my_json)
            tmp_dicts.append(tmp_dict)
        
        ## calculate the mean of selected metrics
        for metric in ['precision', 'recall', 'f1-score']:
            vals = defaultdict(lambda: [])
            for mode in ['macro avg']:
                for i in range(len(workspaces)):
                    tmp = tmp_dicts[i][mode][metric]
                    vals[mode].append(tmp)
            
            print(f"{metric}-{mode}:", "{:.3f}±{:.3f}".format(np.mean(vals['macro avg']), np.std(vals['macro avg'])))
        print("\n\n")
                        
                
        # except:
        #     print("***************workspace missing:", pd)
            
        