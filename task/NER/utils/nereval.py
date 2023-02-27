import re
import numpy as np
from collections import defaultdict

TNE = "this_tag_does_not_exists"

def classifcation_report(tags_true: list, tags_pred: list, mode="lenient", scheme="IOB2") -> dict:
    """caculate lenient matching score, including F1-score, precision, and recall

    Args:
        tags_true (list): true tags
        tags_pred (list): predicted tags
        mode (str, optional): matching model, strict or lenient. Defaults to "lenient".
        scheme (str, optional): annotation scheme, currently only support IOB2

    Returns:
        dict: return metrics as a dict
    """    
    tags_true = [r+"-default" if r in ['B', 'I'] else r for r in tags_true ]
    tags_pred = [r+"-default" if r in ['B', 'I'] else r for r in tags_pred ]
    
    assert scheme in ['IOB2'] # sanity check
    
    predict, truth, matched = defaultdict(lambda: 0.), defaultdict(lambda: 0.), defaultdict(lambda: 0.)
    
    cur_matching_tag = TNE  # auxiliary variable for lenient matching
    start_matching = TNE # auxiliary variable for strict matching
    cur_matching_tag_pred = TNE
    for t, p in zip(tags_true, tags_pred):
        ## get the total ground truth number, will be used for recall calculation
        if re.match("^(B-)", t):
            truth[re.sub("(B-)", "", t)] += 1
            cur_matching_tag = re.sub("(B-)", "", t)
        elif re.match("^I-", t) and cur_matching_tag in t: #! continue matching
            pass
        else: #! abort
            cur_matching_tag = TNE 
        
        ## get the total prediction number, will be used for precision calculation
        if re.match("^(B-)", p):
            predict[re.sub("(B-)", "", p)] += 1
            cur_matching_tag_pred = re.sub("(B-)", "", p)
        elif re.match("^(I-)", p) and cur_matching_tag_pred in p:
            pass
        else:
            cur_matching_tag_pred = TNE
        
        if mode == "lenient":
            if cur_matching_tag != TNE and cur_matching_tag == cur_matching_tag_pred:
                matched[cur_matching_tag] += 1
                cur_matching_tag_pred, cur_matching_tag = TNE, TNE
    
        elif mode == "strict":
            ## get the true positives (strict)
            if p == t and re.match("^(B-)", t):
                if start_matching != TNE: ## case: B_entity1 is adjcent to B_entity2 (two entities can be the same)
                    matched[start_matching] += 1
                start_matching = re.sub("(B-)", "", t)
            elif p == t and re.match("^(I-)", t) and start_matching in t:
                pass
            elif t != "I-"+start_matching and p != "I-"+start_matching and start_matching != TNE:
                matched[start_matching] += 1
                start_matching = TNE
            else: #! matching failed
                start_matching = TNE
        else:
            exit("only support strict or lenient mode, please check your input argument")
    matched[start_matching] = matched[start_matching]+1 if start_matching != TNE and mode == "strict" else matched[start_matching]
        
            
    ## calucalte metrics: precision, recall, F1-score
    unique_entities = [re.sub("B-", "", x) for x in set(tags_true) if re.match("^B-", x)]
    
    metrics = defaultdict(lambda: defaultdict(lambda: 0))
    for ue in unique_entities:
        metrics[ue][f'precision'] = matched[ue]/predict[ue] if predict[ue] > 0 else 0
        metrics[ue][f'recall'] = matched[ue]/truth[ue] if truth[ue] > 0 else 0
        metrics[ue][f'f1-score'] = (2*metrics[ue]['precision']*metrics[ue]['recall'])/(metrics[ue]['precision']+metrics[ue]['recall']) if (metrics[ue]['precision']+metrics[ue]['recall'] > 0) else 0
        # print(f"tag: {ue} \t precision:{metrics[ue]['precision']} \t recall:{metrics[ue]['recall']} \t f1-score:{metrics[ue]['f1-score']}")
        # print(matched[ue], truth[ue], predict[ue])
    return metrics