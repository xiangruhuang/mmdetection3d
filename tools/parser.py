import numpy as np
import sys

str = sys.argv[1]
print(str)

keys = str.split(' ')[0::2]
vals = str.split(' ')[1::2]
APs = []
spec_AP = {}
for key, val in zip(keys, vals):
    key = key.split('/')[1].split(':')[0]
    val = float(val.split(',')[0])
    if 'AP' in key:
        name = key.split('_AP')[0]
        if spec_AP.get(name, None) is None:
            spec_AP[name] = []
        spec_AP[name].append(val)
        APs.append(val)
    print(key, val)

for key, val in spec_AP.items():
    print(key, np.mean(val))
print(np.mean(APs))
