import re
import json

def parse_summary(str):
    str = str.strip()
    str = re.sub("\n+", "\t", str)
    tmp_list = str.split("\t")
    
    new_list = [re.split(r"  +", str.strip()) for str in tmp_list][1:]
    
    ret_dict = {}
    for l in new_list:
        ret_dict[l[0]] = {}
        if l[0] == "accuracy":
            ret_dict[l[0]]['precision'] = float(l[1])
            ret_dict[l[0]]['recall'] = float(l[1])
            ret_dict[l[0]]['f1-score'] = float(l[1])
            ret_dict[l[0]]['support'] = float(l[2])
        else:
            ret_dict[l[0]]['precision'] = float(l[1])
            ret_dict[l[0]]['recall'] = float(l[2])
            ret_dict[l[0]]['f1-score'] = float(l[3])
            ret_dict[l[0]]['support'] = float(l[4])
    
    return ret_dict
    
    

if __name__ == "__main__":
    example = '''
precision    recall  f1-score   support

           0       0.00      0.00      0.00        28
           1       0.00      0.00      0.00        47
           2       1.00      0.02      0.04        55
           3       0.00      0.00      0.00         5
           4       0.72      1.00      0.84       251
           5       0.28      0.85      0.42        40
           6       0.00      0.00      0.00        19
           7       0.00      0.00      0.00         4
           8       0.50      0.33      0.40        51

    accuracy                           0.60       500
   macro avg       0.28      0.24      0.19       500
weighted avg       0.55      0.60      0.50       500
    '''
    print(example)
    ret_dict = parse_summary(example)
    
    json_object = json.dumps(ret_dict, indent=4)
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)
    
    
    