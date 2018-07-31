import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import operator
from skimage import color
import os
from mpl_toolkits.mplot3d import Axes3D
import pickle
from mpl_toolkits.mplot3d import proj3d
import random
import matlab.engine

def unpickleandplot(pickled_folder,
                    my_newer_folder,
                    my_valid_folder,
                    background=False,
                    amplitudes=[100, 128, 128],
                    chop=True,
                    chop_size=3500,
                    sample=False,
                    sample_size=700,
                    kded=True,
                    scale_points=True,
                    scale_points_factor_no_kded=100,
                    scale_points_factor_kded=80000,
                    axis_scale=False,
                    axis_scale_vals=[2, 1, 1],
                    la_size=7,
                    title='My Title'):

    """Take folder with pickled lists, re-plot 3D histogram with more specifications.
        Use in conjunction with MY_NEW_FOLDER from googleimage_lin_func.py or googleimage_seg.py.

    PICKLED_FOLDER -- Absolute filepath to input folder containing pickled lists; MY_NEW_FOLDER from googleimage_lin_func.py or googleimage_seg_func.py
    MY_NEWER_FOLDER -- Absolute filepath to the folder to where plots from unpickleandplot.py will be exported (must already exist)
    MY_VALID_FOLDER -- Absolute filepath to folder containing valid_lab.pkl and valid_rgb.pkl
    BACKGROUND -- If True, plot background colors instead of foreground. Only to be used in conjunction with googleimage_seg_func.py
    AMPLITUDES -- Amplitudes of each axis [L a b], should match values used in googleimage_lin_func.py
    CHOP -- If True, plot only the first CHOP_SIZE most frequent values (only affects visualization)
    SAMPLE -- If True, use SAMPLE_SIZE to randomly thin out data if plots are too dense (only affects visualization)
    KDED -- If True, assume that KDE was True in googleimage_lin_func.py
    SCALE_POINTS -- If True, scale point sizes by frequency
    SCALE_POINTS_FACTOR_NO_KDED -- Factor by which to downsize point sizes if KDE was False in googleimage_lin_func.py
    SCALE_POINTS_FACTOR_KDED -- Factor by which to upsize point sizes if KDE was True in googleimage_lin_func.py
    AXIS_SCALE -- If True, rescale axes using AXIS_SCALE_VALS [L a b]
    LA_SIZE -- Sets size of labels on 'a' and 'b' axes if AXIS_SCALE=True (rescaling plot can lead to muddy-looking axes)
    TITLE -- Plot title
    """

    assert os.path.exists(my_newer_folder)

    with open(my_valid_folder + '/valid_lab.pkl', 'rb') as pickle_load:
        valid_lab = pickle.load(pickle_load)
    with open(my_valid_folder + '/valid_rgb.pkl', 'rb') as pickle_load:
        valid_rgb = pickle.load(pickle_load)

    L_amp, a_amp, b_amp = amplitudes[0], amplitudes[1], amplitudes[2] # Amplitude of each axis

    if background:
        # unpickle pickled lists; lists already sorted by descending frequency
        with open(pickled_folder + '/' + 'background_colors.pkl', 'rb') as pickle_load:
            s_varlist = pickle.load(pickle_load)

        with open(pickled_folder + '/' + 'background_densities.pkl', 'rb') as pickle_load:
            s_vardensity = pickle.load(pickle_load)
    else:
        # unpickle pickled lists; lists already sorted by descending frequency
        with open(pickled_folder + '/' + 'colors.pkl', 'rb') as pickle_load:
            s_varlist = pickle.load(pickle_load)

        with open(pickled_folder + '/' + 'densities.pkl', 'rb') as pickle_load:
            s_vardensity = pickle.load(pickle_load)

    # sorted and chopped
    if chop:
        s_c_vardensity, s_c_varlist = s_vardensity[:chop_size], s_varlist[:chop_size]
    else:
        s_c_vardensity, s_c_varlist = s_vardensity, s_varlist

    # to thin out plot, use a random sample
    if sample:
        s_c_varlist, s_c_vardensity = zip(*random.sample(list(zip(s_c_varlist, s_c_vardensity)), sample_size))


    x, y, z = [], [], []
    for lab in s_c_varlist:
        x.append(lab[0])
        y.append(lab[1])
        z.append(lab[2])

    colors = []
    for i in range(len(s_c_varlist)):
        lab_color = s_c_varlist[i]
        rgb_color = list(valid_rgb[valid_lab.index(lab_color)][0])
        colors.append(rgb_color)
    colors = np.asarray(colors)

    plt.close()
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    if scale_points:
        if kded:
            ax.scatter(z, y, x, s=[foo*scale_points_factor_kded for foo in s_c_vardensity], c=colors)
        else:
            ax.scatter(z, y, x, s=[foo/scale_points_factor_no_kded for foo in s_c_vardensity], c=colors)
    else:
        ax.scatter(z, y, x, s=10, c=colors)

    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('L')
    ax.set_xlim([-b_amp, b_amp])
    ax.set_ylim([-a_amp, a_amp])
    ax.set_zlim([0, L_amp])
    if background:
        plt.title(title + ' background')
    else:
        plt.title(title)

    if axis_scale: # https://stackoverflow.com/questions/30223161/matplotlib-mplot3d-how-to-increase-the-size-of-an-axis-stretch-in-a-3d-plo/30315313
        """
        Scaling is done from here...
        """
        L_scale, a_scale, b_scale = axis_scale_vals[0], axis_scale_vals[1], axis_scale_vals[2]
        x_scale, y_scale, z_scale = b_scale, a_scale, L_scale

        scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        scale=scale*(1.0/scale.max())
        scale[3,3]=1.0

        def short_proj():
          return np.dot(Axes3D.get_proj(ax), scale)

        ax.get_proj=short_proj
        """
        to here
        """
        matplotlib.rc('xtick', labelsize=la_size)
        matplotlib.rc('ytick', labelsize=la_size)

    plt.savefig(my_newer_folder + '/' + 'histogram 3D (KDE)' + '.svg', format='svg', bbox_inches='tight')
