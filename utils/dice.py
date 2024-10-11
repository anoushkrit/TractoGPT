#%%
import numpy as np
from scipy.spatial import distance
import sys
# sys.path.append('/scratch/anoushkrit.scee.iitmandi/ENVS/scilpy/scilpy')
# sys.path.append('/scratch/anoushkrit.scee.iitmandi/ENVS')
import argparse

def hcp_scores(pred_tractogram=None, gt_tractogram=None):
    import json
    import os
    import nibabel as nib
    from dipy.io.streamline import load_tractogram
    import numpy as np
    sys.path.append('/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/scilpy')
    from scilpy.tractanalysis.streamlines_metrics import compute_tract_counts_map
    from dipy.tracking.metrics import length as slength

    if pred_tractogram is None or gt_tractogram is None:
        print("Both predicted_trk and gt_trk are required.")
        return

    def compute_binary_map(sft):
        """Compute the binary map of a tractogram."""
        sft.to_vox()
        sft.to_corner()
        streamlines = sft.streamlines
        _, dimensions, _, _ = sft.space_attributes
        if not streamlines:
            pass
        binary_map = compute_tract_counts_map(streamlines, dimensions)
        binary_map[binary_map > 0] = 1
        return binary_map

    def compute_voxel_pairwise_measures(bundle_fname, gs_fname):
        """Compute comparison measures between two bundles."""
        bundle_binary_map = compute_binary_map(bundle_fname)
        gs_binary_map = compute_binary_map(gs_fname)

        bundle_indices = np.where(bundle_binary_map.flatten() > 0)[0]
        gs_indices = np.where(gs_binary_map.flatten() > 0)[0]

        tp = len(np.intersect1d(bundle_indices, gs_indices))
        fp = len(np.setdiff1d(bundle_indices, gs_indices))
        fn = len(np.setdiff1d(gs_indices, bundle_indices))

        if tp == 0:
            overlap = 0.
            overreach = None
            dice = 0.
        else:
            overlap = tp / float(tp + fn)
            overreach = fp / float(tp + fn)
            dice = 2 * tp / float(2 * tp + fp + fn)

        return {"dice": dice, "overlap": overlap, "overreach": overreach}
    
    return compute_voxel_pairwise_measures(pred_tractogram, gt_tractogram)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute HCP scores for predicted and ground truth tractograms.")
    parser.add_argument("--predicted_trk", type=str, help="Path to the predicted tractogram")
    parser.add_argument("--gt_trk", type=str, help="Path to the ground truth tractogram (eg. /neuro/dMRI_data/105HCP/train/992774/tracts/CG_right.trk)")

    args = parser.parse_args()

    predicted_trk = args.predicted_trk
    gt_trk = args.gt_trk

    hcp_scores(predicted_trk, gt_trk)
# %%
