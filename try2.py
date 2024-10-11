from utils.dice import *
import pickle
from nibabel.streamlines.array_sequence import ArraySequence
import numpy as np
from dipy.io.streamline import load_tractogram
import json
from dipy.io.stateful_tractogram import StatefulTractogram, Space
from utils.dice import hcp_scores

with open('test_pred.pkl', 'rb') as f:
    true_labels, predicted_labels, streamlines = pickle.load(f)

with open('label_int_mapping_TrInF.json', 'r') as f:
    label_int_mapping = json.load(f)
int_label_mapping = {int(v): k for k, v in label_int_mapping.items()}

unique_labels = np.unique(true_labels)

split = 'test'
sub = 1047
tractogram = load_tractogram(f"/scratch/anoushkrit.scee.iitmandi/DATA/TractoInferno/derivatives/{split}set/sub-{sub}/tractography/sub-1047__AF_L.trk", "same", bbox_valid_check=False)
for label in unique_labels:

    true_streamlines = streamlines[true_labels == label].squeeze()
    pred_streamlines = streamlines[predicted_labels == label].squeeze()
    # if len(true_streamlines) >5000:
    #     true_streamlines = true_streamlines[np.random.choice(true_streamlines.shape[0], 5000, replace=False)]
    # if len(pred_streamlines) > 5000:
    #     pred_streamlines = pred_streamlines[np.random.choice(pred_streamlines.shape[0], 5000, replace=False)]
    
    pred_streamlines = pred_streamlines.tolist()
    true_streamlines = true_streamlines.tolist()
    t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram)
    t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram)
    # t1 = StatefulTractogram.from_sft(ArraySequence(true_streamlines), tractogram, space=Space.VOX)
    # t2 = StatefulTractogram.from_sft(ArraySequence(pred_streamlines), tractogram, space=Space.VOX)
    print(int_label_mapping[label], hcp_scores(t1, t2))

