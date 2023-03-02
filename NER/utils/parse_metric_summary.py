import re
import json

def parse_summary(str):
    str = str.strip()
    str = re.sub("\n+", "\t", str)
    tmp_list = str.split("\t")
    # print(tmp_list)
    
    new_list = [re.split(r"  +", str.strip()) for str in tmp_list][1:]
    # print(new_list)
    
    ret_dict = {}
    for l in new_list:
        ret_dict[l[0]] = {}
        ret_dict[l[0]]['precision'] = float(l[1])
        ret_dict[l[0]]['recall'] = float(l[2])
        ret_dict[l[0]]['f1-score'] = float(l[3])
        ret_dict[l[0]]['support'] = float(l[4])
    
    return ret_dict
    
    

if __name__ == "__main__":
    example = '''
              precision    recall  f1-score   support

        MISC       0.00      0.00      0.00         1
         PER       1.00      1.00      1.00         1

   micro avg       0.50      0.50      0.50         2
   macro avg       0.50      0.50      0.50         2
weighted avg       0.50      0.50      0.50         2
    '''
    print(example)
    ret_dict = parse_summary(example)
    
    json_object = json.dumps(ret_dict, indent=4)
    with open("sample.json", "w") as outfile:
        outfile.write(json_object)
    
    
    