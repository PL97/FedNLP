import os

workspace = ['workspace1', 'workspace2', 'workspace3']

from pathlib import Path

for w in workspace:
    for path in Path(w).rglob(r'*/fedavg/'):
        if os.path.exists(os.path.join(path, "evaluation.json")):
            # print(path)
            pass
        else:
            print(path)
    
    for path in Path(w).rglob(r'*/baseline/site-*'):
        if os.path.exists(os.path.join(path, "evaluation.json")):
            # print(path)
            pass
        else:
            print(path)

