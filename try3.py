import numpy as np
import os
import pickle
from scipy.interpolate import CubicSpline
from multiprocessing import Pool
from dipy.tracking.distances import approx_polygon_track as apt


def downsample(args):
    streamline, pts = args
    streamline = apt(streamline,.1)
    if len(streamline) != pts: 
        resampled_streamline = interp_streamline((streamline, pts))
    else:
        resampled_streamline = streamline
    return resampled_streamline

def interp_streamline(args):
    streamline, pts = args
    # old_points = np.linspace(0, 1, streamline.shape[0])
    # new_points = np.linspace(0, 1, pts)

    x_old = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), streamline.shape[0])
    y_old = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), streamline.shape[0])
    z_old = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), streamline.shape[0])

    x_new = np.linspace(streamline[:, 0].min(), streamline[:, 0].max(), pts)
    y_new = np.linspace(streamline[:, 1].min(), streamline[:, 1].max(), pts)
    z_new = np.linspace(streamline[:, 2].min(), streamline[:, 2].max(), pts)
    
    x_interp = CubicSpline(x_old, streamline[:, 0], bc_type='natural')
    y_interp = CubicSpline(y_old, streamline[:, 1], bc_type='natural')
    z_interp = CubicSpline(z_old, streamline[:, 2], bc_type='natural')
    
    resampled_streamline = np.vstack([
        x_interp(x_new),
        y_interp(y_new),
        z_interp(z_new)
    ]).T
    
    return resampled_streamline

def choose_upORdownsample(args):
    streamline = args
    pts = 256
    if pts > len(streamline):
        return interp_streamline((streamline, pts))
    else:
        return downsample((streamline, pts))

if __name__ == '__main__':
    data_path = '/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/trainset'
    save_path = '/scratch/anoushkrit.scee.iitmandi/github/TractoGPT/data/dataTrInF/qbx/processed_data/resampled_streamlines_256/trainset'
    os.makedirs(save_path, exist_ok=True)
    subjects = os.listdir(data_path)
    for subject in subjects:
        subject_path = os.path.join(data_path, subject)

        with open(subject_path, 'rb') as f:
            data = pickle.load(f)
        
        streamlines = data[1]

        with Pool() as p:
            resampled_streamlines = p.map(choose_upORdownsample, streamlines)
        
        resampled_streamlines = np.array(resampled_streamlines)

        print(f'{subject.split(".")[0]}: {resampled_streamlines.shape}')
        filename = os.path.join(save_path, f'{subject.split(".")[0]}__{resampled_streamlines.shape[0]}.dat')
        print(f'Saving {filename}')
        memmap = np.memmap(filename, dtype='float32', mode='w+', shape=resampled_streamlines.shape)
        memmap[:] = resampled_streamlines[:]
        memmap.flush()


