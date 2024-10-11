import numpy as np
from dipy.io.streamline import load_trk
from nibabel.streamlines.array_sequence import ArraySequence
import plotly.graph_objects as go
from dipy.segment.bundles import select_random_set_of_streamlines
from dipy.segment.clustering import QuickBundlesX, QuickBundles
from dipy.segment.metric import (
    AveragePointwiseEuclideanMetric, SumPointwiseEuclideanMetric, CosineMetric)
from dipy.segment.featurespeed import VectorOfEndpointsFeature, CenterOfMassFeature
# https://docs.dipy.org/stable/reference/dipy.segment.html#centerofmassfeature
from pathlib import Path
from collections import Counter
import os 
from tqdm import tqdm
from scipy.interpolate import CubicSpline

#%% 

def visualise_bundle(viz_tracks, name, interactive, separate=True ):
    scene = window.Scene()
    scene.SetBackground(0,0,0)
    if separate: 
        viz_tracks = [np.asarray(track) for track in viz_tracks]
        for track in viz_tracks:
            scene.add(actor.line([track], colors= colormap.line_colors, fake_tube=True, linewidth=2))
    else:
        scene.add(actor.line(viz_tracks, fake_tube=True, linewidth=2))
    scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                    position=(-360.11, -30.46, -40.44),
                    view_up=(-0.03, 0.028, 0.89))
    if interactive:
        window.show(scene)
    else:
        window.record(scene, out_path=name+".png", size=(2000,2000))

def get_total_points(tracks):
    return np.vstack(tracks).shape[0]

def merge_tracks(tracks): 
    return

def interpolate_streamline(tracks, pps, c = 3):
    """
    streamlines: list
    pps: int (points per streamline)
    c: channel is 3 
    """
    a = len(tracks)
    resampled_tracks = np.zeros((a,pps,c))

    # TODO: Multiprocessing this loop to speed up
    for i,track in enumerate(tracks):
        # old_points = np.linspace(0, 1, track.shape[0])
        x_old = np.linspace(track[:, 0].min(), track[:, 0].max(), track.shape[0])
        y_old = np.linspace(track[:, 1].min(), track[:, 1].max(), track.shape[0])
        z_old = np.linspace(track[:, 2].min(), track[:, 2].max(), track.shape[0])

        x_new = np.linspace(track[:, 0].min(), track[:, 0].max(), pps)
        y_new = np.linspace(track[:, 1].min(), track[:, 1].max(), pps)
        z_new = np.linspace(track[:, 2].min(), track[:, 2].max(), pps)

        x_interp = CubicSpline(x_old, track[:, 0], bc_type='natural')
        y_interp = CubicSpline(y_old, track[:, 1], bc_type='natural')
        z_interp = CubicSpline(z_old, track[:, 2], bc_type='natural')
        
        # Resample streamline
        resampled_track = np.vstack([
            x_interp(x_new),
            y_interp(y_new),
            z_interp(z_new)
        ]).T
        
        resampled_tracks[i] = resampled_track
    return resampled_tracks


def get_named_colors():
    named_colors = [
        'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
        'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
        'blueviolet', 'brown', 'burlywood', 'cadetblue',
        'chartreuse', 'chocolate', 'coral', 'cornflowerblue',
        'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
        'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen',
        'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
        'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
        'darkslateblue', 'darkslategray', 'darkslategrey',
        'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
        'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
        'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
        'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey',
        'green', 'greenyellow', 'honeydew', 'hotpink',
        'indianred', 'indigo', 'ivory', 'khaki',
        'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon',
        'lightblue', 'lightcoral', 'lightcyan', 'lightgoldenrodyellow',
        'lightgray', 'lightgrey', 'lightgreen', 'lightpink',
        'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray',
        'lightslategrey', 'lightsteelblue', 'lightyellow', 'lime',
        'limegreen', 'linen', 'magenta', 'maroon',
        'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple',
        'mediumseagreen', 'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
        'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose',
        'moccasin', 'navajowhite', 'navy', 'oldlace',
        'olive', 'olivedrab', 'orange', 'orangered',
        'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
        'palevioletred', 'papayawhip', 'peachpuff', 'peru',
        'pink', 'plum', 'powderblue', 'purple',
        'rebeccapurple', 'red', 'rosybrown', 'royalblue',
        'saddlebrown', 'salmon', 'sandybrown', 'seagreen',
        'seashell', 'sienna', 'silver', 'skyblue',
        'slateblue', 'slategray', 'slategrey', 'snow',
        'springgreen', 'steelblue', 'tan', 'teal',
        'thistle', 'tomato', 'turquoise', 'violet',
        'wheat', 'white', 'whitesmoke', 'yellow',
        'yellowgreen'
    ]
    return named_colors
def visualize_tracks(tracks, points, static_color=None, save=False):
    fig = go.Figure()
    named_colors = get_named_colors()
    static_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'black']
    if points is not None:
        for p, point in enumerate(points): 
            x_values = point[:, 0]  # X-axis values
            y_values = point[:, 1]  # Y-axis values
            z_values = point[:, 2]  # Z-axis values
            fig.add_trace(go.Scatter3d(x=x_values, 
                                    y=y_values, 
                                    z=z_values,
                                    mode='markers', 
                                    marker=dict(size=2, color=static_colors[p% len(static_colors)]), 
                                    name='original_points'))

    if tracks is not None:
        for i, track in enumerate(tracks): 
            x_values = track[:, 0]  # X-axis values
            y_values = track[:, 1]  # Y-axis values
            z_values = track[:, 2]  # Z-axis values

            if static_color is None: 
                color_index = i % len(named_colors)  # Calculate the color index based on the current iteration
            else:
                color_index = -1 
            fig.add_trace(go.Scatter3d(x=x_values, 
                                        y=y_values, 
                                        z=z_values,
                                        mode='lines', 
                                        line=dict(color=named_colors[color_index], width=4), 
                                        name=f'track_{i}'))

    if tracks is None and points is None:
        raise ValueError("Both Tracks and Points are None")
    

    fig.update_layout(
        title='3D Plot of Streamlines',
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Z-axis'
        ),
        legend_title='Lines'
    )
    if save is False: 
        fig.show()
    else: 
        if not os.path.exists('plots'):
            os.makedirs('plots')
        fig.write_html(f"plots/track_{np.random.randint(0, 1000)}.html")

def load_TractoInferno(tracts_directory):
    tracks = [] 
    labels = [] 
    import re        
    for path in tqdm(Path(tracts_directory).iterdir()):
        path = path / "tractography"
        for p in tqdm(Path(path).iterdir()):
            subject = re.search(r'\d{4}', p).group()
            print(subject)
            if p.is_file() and path.suffix == '.trk':
                label = re.search(r'__(.*?)\.trk', p).group(1)
                print(label)
                tractogram = load_trk(str(path), "same", bbox_valid_check=False)
                num_streamlines = len(tractogram.streamlines)
        with open(os.path.join(tracts_directory, f"{subject}.txt"), "a") as f:
            
            f.write(f"Number of Streamline in {label} is {num_streamlines}\n")

        for s in tractogram.streamlines:
            labels.append(label)
            tracks.append(s)
    return tracks, labels

def load_HCP(tracts_directory):
    tracks = [] 
    labels = [] 
    import re
    dataset = "HCP"
    subject = re.search(r'\d{6}', tracts_directory).group()
    print(f"Loading streamlines for {subject} for {dataset} dataset")
    for path in tqdm(Path(tracts_directory).iterdir()):
        if path.is_file() and path.suffix == '.trk':
            label = str(path).split('/')[-1].split('.')[0]
            tractogram = load_trk(str(path), "same", bbox_valid_check=False)
            num_streamlines = len(tractogram.streamlines)
            for s in tractogram.streamlines:
                labels.append(label)
                tracks.append(s)
    return tracks, labels



