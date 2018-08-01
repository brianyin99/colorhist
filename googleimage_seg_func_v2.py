import numpy as np
from math import ceil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import operator
from skimage import color
import os
from mpl_toolkits.mplot3d import Axes3D
import pickle
from fastkde import fastKDE
import random
from scipy import spatial
import matlab.engine

def googleimage_seg(my_input_folder,
                    my_valid_folder,
                    seg_folder,
                    my_new_folder,
                    amplitudes=[100, 128, 128],
                    dimensions=[5, 5, 5],
                    kde=True,
                    chop=True,
                    chop_size=3500,
                    sample=False,
                    sample_size=700,
                    title='My Title'):

    """Take folder MY_INPUT_FOLDER of images and SEG_FOLDER of segmented images, create folder MY_NEW_FOLDER containing 3D Histograms of foreground and background pixel distributions.
        Images can be scraped from google images using https://github.com/hardikvasa/google-images-download.
        Images can be segmented in MATLAB using http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/

    MY_INPUT_FOLDER -- Absolute filepath to the folder of images you wish to plot
    MY_VALID_FOLDER -- Absolute filepath to the folder containing valid_lab.pkl and valid_rgb.pkl
    SEG_FOLDER -- Absolute filepath to folder of segmented images (.png files)
    MY_NEW_FOLDER -- Absolute filepath to the folder to where plots and data will be exported
    AMPLITUDES -- Amplitudes of each axis [L a b]; axes extend from 0 to L, -a to a, and -b to b
    DIMENSIONS -- Dimensions of bins in CIELAB space [L a b]; all pixels within confines of a bin take on the same color value
    KDE -- If True, use a Kernel Density Estimate to smooth results in CIELAB space after data collection https://bitbucket.org/lbl-cascade/fastkde
    CHOP -- If True, plot only the first CHOP_SIZE most frequent values (only affects visualization)
    SAMPLE -- If True, use SAMPLE_SIZE to randomly thin out data if plots are too dense (only affects visualization)
    TITLE -- Plot title
    """

    #********************
    """INITIALIZATION"""
    #********************
    assert (not os.path.exists(my_new_folder))
    os.makedirs(my_new_folder)
    Lw, aw, bw = dimensions[0], dimensions[1], dimensions[2] # Bin dimensions (widths)
    L_amp, a_amp, b_amp = amplitudes[0], amplitudes[1], amplitudes[2] # Amplitude of each axis
    Lbins, abins, bbins = L_amp/Lw, a_amp*2/aw, b_amp*2/bw # Number of 1D bins per axis
    L_list, a_list, b_list = [], [], [] # for figures
    L_blist, a_blist, b_blist = [], [], [] # for grounds
    unique_bins, unique_background_bins = {}, {} # Initialize dictionaries
    Lvec, avec, bvec = np.linspace(0, L_amp, Lbins+1), np.linspace(-a_amp, a_amp, abins+1), np.linspace(-b_amp, b_amp, bbins+1) # Vectors for each axis

    with open(my_valid_folder + '/valid_lab.pkl', 'rb') as pickle_load:
        valid_lab = pickle.load(pickle_load)
    with open(my_valid_folder + '/valid_rgb.pkl', 'rb') as pickle_load:
        valid_rgb = pickle.load(pickle_load)

    #************************
    """DEFINING FUNCTIONS"""
    #************************
    def bounder_v2(x, v):
        """Take x and evenly-spaced ordered vector, return list of bin coordinates"""

        x0 = v[0] # minimum value of vector
        w = v[1]-x0 # width of a bin on given axis
        binnum = ceil((x-x0)/w) # number of bins is distance between x & x0, divided by bin width

        # edge case
        if binnum == 0:
            binnum == 1

        return binnum

    def binner_v2(Linput, ainput, binput):
        """Take an LAB value, axis vectors, return linear index of 3D bin"""

        # position of bin on each axis
        Lbin = bounder_v2(Linput, Lvec)
        abin = bounder_v2(ainput, avec)
        bbin = bounder_v2(binput, bvec)

        return [Lbin, abin, bbin]

    def sub2ind(ypos,xpos):
        """Take a 2D matrix coordinates, return linear index"""

        linear_index = imagewidth*ypos+xpos
        return linear_index

    def ind2sub(linear_index):
        """Take linear index, return 2D matrix coordinates"""

        ypos = linear_index // imagewidth
        xpos = linear_index % imagewidth
        return (ypos,xpos)

    def bins2lab(bin_list):
        """Take bin_list [Lbin, abin, bbin], return [L, a, b]"""
        L = Lw*bin_list[0]-Lw/2
        a = -a_amp+aw*bin_list[1]-aw/2
        b = -b_amp+bw*bin_list[2]-bw/2
        return [L, a ,b]

    def uniq(lst):
        last = object()
        for item in lst:
            if item == last:
                continue
            yield item
            last = item


    images_skipped, total_images = 0, 0


    #*********************
    """DATA COLLECTION"""
    #*********************
    # iterate through folder of images
    for my_image in os.listdir(my_input_folder):
        total_images += 1
        my_image_path = my_input_folder + '/' + my_image # reverse engineer file path to image
        try:
            rgb_img = mpimg.imread(my_image_path) # array [height][width][RGB]
        except ValueError as error:
            print(error, ';', '%s was skipped' % my_image)
            images_skipped += 1
            continue
        except OSError as os_error:
            print(os_error, ';', 'No image was skipped') # .DS_store, not an image
            total_images -= 1
            continue
        rgb_img = rgb_img/255
        try:
            lab_img = color.rgb2lab(rgb_img)
        except ValueError as error:
            print(error, ';', '%s was skipped' % my_image)
            images_skipped += 1
            continue

        imshape = lab_img.shape # (height,width,depth)
        imageheight, imagewidth = imshape[0], imshape[1]

        seglist = os.listdir(seg_folder)
        segnum, imagenum, count = 0, my_image[:3], 0
        while segnum != imagenum and count<len(seglist):
            segnum = seglist[count][:3]
            count +=1

        segs_skipped = 0
        if segnum == imagenum:
            exist_segmask = True
        else:
            exist_segmask = False
            segs_skipped += 1

        # iterate through pixels and add to unique bins
        if exist_segmask:
            my_seg_path = seg_folder + '/' + my_image + '.png'
            logicmask = mpimg.imread(my_seg_path) # array [height][width][0 or 1]
            for xpos in range(imagewidth):
                for ypos in range(imageheight):
                    Linput, ainput, binput = lab_img[ypos,xpos][0], lab_img[ypos,xpos][1], lab_img[ypos,xpos][2]
                    bin = str(binner_v2(Linput, ainput, binput)) # string b/c dictionary
                    my_vals = bins2lab(binner_v2(Linput, ainput, binput))
                    if not logicmask[ypos,xpos]: # 0 = grounds, 1 = figures
                        if bin in unique_background_bins:
                            unique_background_bins[bin] += 1
                        else: unique_background_bins[bin] = 1
                        L_blist.append(my_vals[0])
                        a_blist.append(my_vals[1])
                        b_blist.append(my_vals[2])
                    else:
                        if bin in unique_bins:
                            unique_bins[bin] += 1
                        else:
                            unique_bins[bin] = 1
                        L_list.append(my_vals[0])
                        a_list.append(my_vals[1])
                        b_list.append(my_vals[2])
        else:
            for xpos in range(imagewidth):
                for ypos in range(imageheight):
                    Linput, ainput, binput = lab_img[ypos,xpos][0], lab_img[ypos,xpos][1], lab_img[ypos,xpos][2]
                    bin = str(binner_v2(Linput, ainput, binput)) # string b/c dictionary
                    if bin in unique_bins:
                        unique_bins[bin] += 1
                    else:
                        unique_bins[bin] = 1

                    my_vals = bins2lab(binner_v2(Linput, ainput, binput))
                    L_list.append(my_vals[0])
                    a_list.append(my_vals[1])
                    b_list.append(my_vals[2])


    print('Out of %s total images, %d were skipped' % (total_images,images_skipped))
    print('Out of %s images processed, %d were not segmented' % (total_images-images_skipped,segs_skipped))


    #********************
    """VISUALIZATION"""
    #********************

    #for 3D histogram
    plt.close()
    varL, vara, varb = np.asarray(L_list), np.asarray(a_list), np.asarray(b_list)

    if kde:
        myPDF, axes = fastKDE.pdf(varL, vara, varb)
        varL, vara, varb = axes
        varlist, vardensity = [], []
        for L in range(len(varL)):
            for a in range(len(vara)):
                for b in range(len(varb)):
                    varlist.append([varL[L], vara[a], varb[b]])
                    vardensity.append(myPDF[b][a][L])
    else:
        unique_bins_sorted = sorted(unique_bins.items(), key=operator.itemgetter(1), reverse=1) #list of tuples sorted by descending frequency
        varlist, vardensity = [], []
        for unique_bin in unique_bins_sorted:
            varlist.append(bins2lab(eval(unique_bin[0])))
            vardensity.append(unique_bin[1])

    #sorted by descending frequency
    s_vardensity = sorted(vardensity, reverse=True)
    s_varlist = [bin for _,bin in sorted(zip(vardensity,varlist), reverse=True)]
    if 0 in s_vardensity:
        last = s_vardensity.index(0)
        s_varlist, s_vardensity = s_varlist[:last+1], s_vardensity[:last+1]
        print("Color-density pairs with density=0 have been removed")
    my_colors_valid = []

    # For LAB values outside of RGB gamut, use spatial.cKDTree to find nearest LAB values inside of RGB gamut
    # https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    count = 0
    valid_lab_tree = spatial.cKDTree(valid_lab)
    for lab_color in s_varlist:
        if lab_color not in valid_lab:
            lab_color = valid_lab[valid_lab_tree.query(lab_color)[1]]
        my_colors_valid.append(lab_color)
        count += 1
        print(len(s_varlist)+1-count)

    # as of this line, s_varlist and s_vardensity are sorted by density

    #sorting by s_varlist so the while loop works
    my_densities = [bin for _,bin in sorted(zip(my_colors_valid, s_vardensity))]
    my_colors_valid = sorted(my_colors_valid)

    colors_valid, densities = [], []

    #basically a linked list, sums repeated color-density pairs from using spatial.cKDTree above
    my_colors_valid.append("empty")
    my_densities.append("empty")

    currDense = my_densities[0]
    while my_colors_valid[0] != "empty":
        currColor, nextColor = my_colors_valid[0], my_colors_valid[1]
        if nextColor == currColor:
            currDense += my_densities[1]
            my_densities.remove(my_densities[0])
        else:
            colors_valid.append(currColor)
            densities.append(currDense)
            my_densities.remove(my_densities[0])
            currDense = my_densities[0]
        my_colors_valid.remove(currColor)


    s_varlist = [bin for bin,_ in sorted(zip(colors_valid, densities))]
    s_vardensity = sorted(densities)

    # sorted and chopped
    if chop:
        if chop<last:
            s_c_vardensity, s_c_varlist = s_vardensity[:chop_size], s_varlist[:chop_size]
        else:
            print('Chop size of %c >= number of non-zero values %f. No values were chopped.' %(chop, last))
            s_c_vardensity, s_c_varlist = s_vardensity, s_varlist
    else:
        s_c_vardensity, s_c_varlist = s_vardensity, s_varlist

    x, y, z = [], [], []
    for lab in s_c_varlist:
        x.append(lab[0]) #L
        y.append(lab[1]) #a
        z.append(lab[2]) #b

    colors = []
    for i in range(len(s_c_varlist)):
        lab_color = s_c_varlist[i] #already binned
        # rgb_color = list(color.lab2rgb([[lab_color]])[0][0])
        rgb_color = list(valid_rgb[valid_lab.index(lab_color)][0])
        colors.append(rgb_color)
    colors = np.asarray(colors)

    plt.close()
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    if kde:
        ax.scatter(z, y, x, s=[foo*80000 for foo in s_c_vardensity], c=colors)
    else:
        ax.scatter(z, y, x, s=[foo/100 for foo in s_c_vardensity], c=colors)
    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('L')
    ax.set_xlim([-b_amp, b_amp])
    ax.set_ylim([-a_amp, a_amp])
    ax.set_zlim([0, L_amp])
    plt.title(title)
    plt.savefig(my_new_folder + '/' + 'histogram 3D' + '.svg', format='svg', bbox_inches='tight')

    with open(my_new_folder + '/' + 'colors.pkl', 'wb') as pickle_file:
        pickle.dump(s_varlist, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(my_new_folder + '/' + 'densities.pkl', 'wb') as pickle_file:
        pickle.dump(s_vardensity, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    # for 3D histogram of background pixels
    plt.close()
    varL, vara, varb = np.asarray(L_blist), np.asarray(a_blist), np.asarray(b_blist)
    if kde:
        myPDF, axes = fastKDE.pdf(varL, vara, varb)
        varL, vara, varb = axes
        varlist, vardensity = [], []
        for L in range(len(varL)):
            for a in range(len(vara)):
                for b in range(len(varb)):
                    varlist.append([varL[L], vara[a], varb[b]])
                    vardensity.append(myPDF[b][a][L])
    else:
        unique_background_bins_sorted = sorted(unique_background_bins.items(), key=operator.itemgetter(1), reverse=1) #list of tuples sorted by descending frequency
        varlist, vardensity = [], []
        for unique_background_bin in unique_background_bins_sorted:
            varlist.append(bins2lab(eval(unique_background_bin[0])))
            vardensity.append(unique_background_bin[1])

    #sorted by descending frequency
    s_vardensity = sorted(vardensity, reverse=True)
    s_varlist = [bin for _,bin in sorted(zip(vardensity,varlist), reverse=True)]

    if 0 in s_vardensity:
        last = s_vardensity.index(0)
        s_varlist, s_vardensity = s_varlist[:last+1], s_vardensity[:last+1]
        print("Color-density pairs with density=0 have been removed")
    my_colors_valid = []

    # For LAB values outside of RGB gamut, use spatial.cKDTree to find nearest LAB values inside of RGB gamut
    # https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
    count = 0
    valid_lab_tree = spatial.cKDTree(valid_lab)
    for lab_color in s_varlist:
        if lab_color not in valid_lab:
            lab_color = valid_lab[valid_lab_tree.query(lab_color)[1]]
        my_colors_valid.append(lab_color)
        count += 1
        print(len(s_varlist)+1-count)

    # as of this line, s_varlist and s_vardensity are sorted by density

    #sorting by s_varlist so the while loop works
    my_densities = [bin for _,bin in sorted(zip(my_colors_valid, s_vardensity))]
    my_colors_valid = sorted(my_colors_valid)

    colors_valid, densities = [], []

    # basically a linked list, sums repeated color-density pairs from using spatial.cKDTree above
    my_colors_valid.append("empty")
    my_densities.append("empty")

    currDense = my_densities[0]
    while my_colors_valid[0] != "empty":
        currColor, nextColor = my_colors_valid[0], my_colors_valid[1]
        if nextColor == currColor:
            currDense += my_densities[1]
            my_densities.remove(my_densities[0])
        else:
            colors_valid.append(currColor)
            densities.append(currDense)
            my_densities.remove(my_densities[0])
            currDense = my_densities[0]
        my_colors_valid.remove(currColor)


    s_varlist = [bin for bin,_ in sorted(zip(colors_valid, densities))]
    s_vardensity = sorted(densities)

    # sorted and chopped
    if chop:
        if chop<last:
            s_c_vardensity, s_c_varlist = s_vardensity[:chop_size], s_varlist[:chop_size]
        else:
            print('Chop size of %c >= number of non-zero values %f. No values were chopped.' %(chop, last))
            s_c_vardensity, s_c_varlist = s_vardensity, s_varlist
    else:
        s_c_vardensity, s_c_varlist = s_vardensity, s_varlist

    # sorted, chopped, and sampled
    if sample:
        s_c_varlist, s_c_vardensity = zip(*random.sample(list(zip(s_c_varlist, s_c_vardensity)), sample_size))

    x, y, z = [], [], []
    for lab in s_c_varlist:
        x.append(lab[0]) #L
        y.append(lab[1]) #a
        z.append(lab[2]) #b

    colors = []
    for i in range(len(s_c_varlist)):
        lab_color = s_c_varlist[i] #already binned
        rgb_color = list(valid_rgb[valid_lab.index(lab_color)][0])
        colors.append(rgb_color)
    colors = np.asarray(colors)

    plt.close()
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    # resizing points
    if kde:
        ax.scatter(z, y, x, s=[foo*80000 for foo in s_c_vardensity], c=colors)
    else:
        ax.scatter(z, y, x, s=[foo/100 for foo in s_c_vardensity], c=colors)
    ax.set_xlabel('b')
    ax.set_ylabel('a')
    ax.set_zlabel('L')
    ax.set_xlim([-b_amp, b_amp])
    ax.set_ylim([-a_amp, a_amp])
    ax.set_zlim([0, L_amp])
    plt.title(title + ' background')
    plt.savefig(my_new_folder + '/' + 'background histogram 3D' + '.svg', format='svg', bbox_inches='tight')

    with open(my_new_folder + '/' + 'background_colors.pkl', 'wb') as pickle_file:
        pickle.dump(s_varlist, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(my_new_folder + '/' + 'background_densities.pkl', 'wb') as pickle_file:
        pickle.dump(s_vardensity, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
