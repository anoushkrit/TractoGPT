#%%
import pickle
from dipy.io.streamline import load_tractogram

def load_stream(trk_path):
    bundle_track = load_tractogram(str(trk_path), 'same', bbox_valid_check=False)
    try:
        return list(bundle_track.streamlines)
    except:
        return []

pth = '/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/testset/sub-1049/tractography'

#%%
import os

tot=0
for cls in os.listdir(pth):
    x = len(load_stream(os.path.join(pth,cls)))
    tot+=x
    # print(x)
print(tot)
    # print(cls)
#%%
import json

pth = '/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/dataTrInF/single_streamline/streamlineData_trainset_150_10000/labels.pkl'
import pickle
with open(pth, 'rb') as f:
    labels = pickle.load(f)
import numpy as np
labels = np.array(labels)
labels = np.unique(labels)
labels = np.sort(labels)

dic = {}
for i,lab in enumerate(labels):
    dic[lab] = i

with open('label_int_mapping_TrInF.json', 'w') as f:
    json.dump(dic, f, indent=4)