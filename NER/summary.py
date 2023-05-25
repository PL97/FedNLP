import json
import numpy as np
import os
from collections import defaultdict
import re

def parse_json(my_json, verbose=False):
    ret_dict = defaultdict(lambda: defaultdict(lambda: 0))
    for metric in ['precision', 'recall', 'f1-score']:
        for mode in ['lenient', 'strict']:
            val = []
            for k, v in my_json['test'][mode].items():
                val.append(v[metric])
            if verbose:
                print(f"{mode}: {np.mean(val)}")
            ret_dict[mode][metric] = np.mean(val)
    return ret_dict

def parse_json_full(my_json, verbose=False):
    ret_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
    for metric in ['precision', 'recall', 'f1-score']:
        for mode in ['lenient', 'strict']:
            for entity, v in my_json['test'][mode].items():
            # val = []
            # for k, v in my_json['test'][mode].items():
            #     val.append(v[metric])
            # if verbose:
            #     print(f"{mode}: {np.mean(val)}")
                ret_dict[mode][metric][entity] = v[metric]
    return ret_dict

    
def summarize_ner():
    workspaces = ['workspace1', 'workspace2', 'workspace3'] 
    for pd, d, f in os.walk(workspaces[0]):
        if "evaluation.json" in f:
            print(pd)
            try:
                tmp_dicts = []
                for ws in workspaces:
                    json_path = os.path.join(re.sub(workspaces[0], ws, pd), "evaluation.json")
                    my_json = json.load(open(json_path))
                    tmp_dict = parse_json(my_json)
                    tmp_dicts.append(tmp_dict)
                
                ## calculate the mean of selected metrics
                for metric in ['precision', 'recall', 'f1-score']:
                    vals = defaultdict(lambda: [])
                    for mode in ['lenient', 'strict']:
                        for i in range(len(workspaces)):
                            tmp = tmp_dicts[i][mode][metric]
                            vals[mode].append(tmp)
                    
                    print(f"{metric}-{mode}:", "{:.3f}±{:.3f} ({:.3f}±{:.3f})".format(np.mean(vals['lenient']), np.std(vals['lenient']), \
                                                                                        np.mean(vals['strict']), np.std(vals['strict'])))
                print("\n\n")
                            
                    
            except:
                print("***************workspace missing:", pd)
                
                
def summarize_size_effect():
    ret_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: [])))
    workspaces = ["workspace_size2", "workspace_size3", "workspace_size4", "workspace_size5", "workspace_size6", "workspace_size7", "workspace_size8", "workspace_size9", "workspace_size10"] 
    for ws in workspaces:
        for pd, d, f in os.walk(ws):
            
                
            if "evaluation.json" in f:
                # print(pd)
                # print(d)
                model = pd.split("/")[-2]
                
                
                tmp_dicts = []

                json_path = os.path.join(pd, "evaluation.json")
                my_json = json.load(open(json_path))
                tmp_dict = parse_json(my_json)
                tmp_dicts.append(tmp_dict)
                
                ## calculate the mean of selected metrics
                for metric in ['precision', 'recall', 'f1-score']:
                    vals = defaultdict(lambda: [])
                    for mode in ['lenient', 'strict']:
                        tmp = tmp_dict[mode][metric]
                        vals[mode].append(tmp)
                    
                    # print(f"{metric}-{mode}:", "{:.3f}±{:.3f} ({:.3f}±{:.3f})".format(np.mean(vals['lenient']), np.std(vals['lenient']), \
                    #                                                                     np.mean(vals['strict']), np.std(vals['strict'])))
                    
                    ret_dict[model][metric]['lenient'].append(np.mean(vals['lenient']))
                    ret_dict[model][metric]['strict'].append(np.mean(vals['strict']))
                # print("\n\n")
    
    
    ## format output
    for model in ret_dict.keys():
        for metric in ret_dict[model].keys():
            if metric != "f1-score":
                continue
            print("###################{}-{}#######################".format(model, metric))
            print(ret_dict[model][metric]['lenient'])
            # print(ret_dict[model][metric]['strict'])
            print("\n\n\n")
            

def summarize_feature_shift_effect():
    ret_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: []))))
    workspaces = ['workspace1', 'workspace2', 'workspace3']
    
    for ws in workspaces:
        for pd, d, f in os.walk(ws):
            if "feature_shift" not in pd:
                    continue
            if "evaluation.json" in f:
                
                # print(pd)
                # print(d)
                model = pd.split("/")[-2] if pd.split("/")[-2] != "baseline" else pd.split("/")[-3]
                method = pd.split("/")[-1]
                
                tmp_dicts = []

                json_path = os.path.join(pd, "evaluation.json")
                my_json = json.load(open(json_path))
                tmp_dict = parse_json(my_json)
                tmp_dicts.append(tmp_dict)
                
                ## calculate the mean of selected metrics
                for metric in ['precision', 'recall', 'f1-score']:
                    vals = defaultdict(lambda: [])
                    for mode in ['lenient', 'strict']:
                        tmp = tmp_dict[mode][metric]
                        vals[mode].append(tmp)
                    
                    ret_dict[model][metric][method]['lenient'].append(np.mean(vals['lenient']))
                    ret_dict[model][metric][method]['strict'].append(np.mean(vals['strict']))
                # print("\n\n")
    ## format output
    for model in ret_dict.keys():
        for metric in ret_dict[model].keys():
            if metric != "f1-score":
                continue
            for method in ret_dict[model][metric].keys():
            
                print("###################{}-{}-{}#######################".format(model, metric, method))
                print(ret_dict[model][metric][method]['lenient'])
                # print(ret_dict[model][metric]['strict'])
                print("\n\n\n")
                                
                        
def summarize_fedalg_effect():
    ret_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda:defaultdict(lambda: [])))))
    workspaces = ['workspace1', 'workspace2', 'workspace3']
    
    for ws in workspaces:
        for pd, d, f in os.walk(ws):
            if "fedprox" not in pd:
                    continue
            if "evaluation.json" in f:
                
                # print(d)
                model = pd.split("/")[-2] if pd.split("/")[-2] != "baseline" else pd.split("/")[-3]
                dataset = pd.split("/")[1]
                method = pd.split("/")[-1]
                
                tmp_dicts = []

                json_path = os.path.join(pd, "evaluation.json")
                my_json = json.load(open(json_path))
                tmp_dict = parse_json(my_json)
                tmp_dicts.append(tmp_dict)
                
                ## calculate the mean of selected metrics
                for metric in ['precision', 'recall', 'f1-score']:
                    vals = defaultdict(lambda: [])
                    for mode in ['lenient', 'strict']:
                        tmp = tmp_dict[mode][metric]
                        vals[mode].append(tmp)
                    
                    ret_dict[dataset][model][metric][method]['lenient'].append(np.mean(vals['lenient']))
                    ret_dict[dataset][model][metric][method]['strict'].append(np.mean(vals['strict']))
                # print("\n\n")
    ## format output
    for dataset in ret_dict.keys():
        for model in ret_dict[dataset].keys():
            for metric in ret_dict[dataset][model].keys():
                if metric != "f1-score":
                    continue
                for method in ret_dict[dataset][model][metric].keys():
                
                    print("###################{}-{}-{}-{}#######################".format(dataset, model, metric, method))
                    metric_list = ret_dict[dataset][model][metric][method]['lenient']
                    print(metric_list)
                    print("{:.3f}±{:.3f}".format(np.mean(metric_list), np.std(metric_list)))
                    # print(ret_dict[model][metric]['strict'])
                    print("\n\n\n")
                
def summarize_2018_n2c2():
    ret_dict = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:defaultdict(lambda: defaultdict(lambda: [])))))
    workspaces = ['workspace1', 'workspace2', 'workspace3']
    
    for ws in workspaces:
        for pd, d, f in os.walk(ws):
            if "2018_n2c2" not in pd:
                    continue
            if "evaluation.json" in f:
                

                model = pd.split("/")[-2] if pd.split("/")[-2] != "baseline" else pd.split("/")[-3]
                method = pd.split("/")[-1]

                json_path = os.path.join(pd, "evaluation.json")
                my_json = json.load(open(json_path))
                tmp_dict = parse_json_full(my_json)
                
                ## calculate the mean of selected metrics
                for metric in ['precision', 'recall', 'f1-score']:
                    vals = defaultdict(lambda: [])
                    for mode in ['lenient', 'strict']:
                        for entity, v in tmp_dict[mode][metric].items():
                            
                            ret_dict[model][metric][method][mode][entity].append(v)
                            
                # print("\n\n")

    ## format output
    for model in ret_dict.keys():
        for metric in ret_dict[model].keys():
            
            if metric != "f1-score":
                continue
            for method in ret_dict[model][metric].keys():
                print("###################{}-{}-{}#######################".format(model, metric, method))
                format_out_mean = dict()
                format_out_std = dict()
                for entity, v in ret_dict[model][metric][method]['lenient'].items():
            
                    
                    metric_list = ret_dict[model][metric][method]['lenient'][entity]
                    # print(metric_list)
                    print("{}: {:.3f}±{:.3f}".format(entity, np.mean(metric_list), np.std(metric_list)))
                    # print(ret_dict[model][metric]['strict'])
                    
                    format_out_mean[entity] = str(round(np.mean(metric_list), 3))
                    format_out_std[entity] = str(round(np.std(metric_list), 3))
                
                format_out_mean = dict(sorted(format_out_mean.items(), key=lambda x:x[0]))
                format_out_mean = dict(sorted(format_out_std.items(), key=lambda x:x[0]))
                    
                print("\t".join(format_out_mean.keys()))
                print("\t".join(format_out_mean.values()))
                print(", ".join(format_out_std.values()))
                print("\n\n\n")
            
if __name__ == "__main__":
    summarize_feature_shift_effect()
    # summarize_fedalg_effect()
    # summarize_2018_n2c2()