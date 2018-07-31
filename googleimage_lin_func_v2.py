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

def googleimage_lin(my_input_folder,
                    my_new_folder,
                    my_valid_folder,
                    amplitudes=[100, 128, 128],
                    dimensions=[5, 5, 5],
                    seg=True,
                    tau=3,
                    kde=True,
                    chop=True,
                    chop_size=3500,
                    sample=False,
                    sample_size=700,
                    title='My Title'):

    """Take folder MY_INPUT_FOLDER of images, create folder MY_NEW_FOLDER containing segmented images and/or 3D Histogram of pixel distribution.
        Images can be scraped from google images using https://github.com/hardikvasa/google-images-download.
        Improves upon v1 -- after KDE, use scipy.spatial.cKDTree to find CIELAB values with corresponding RGB values.

    MY_INPUT_FOLDER -- Absolute filepath to the folder of images you wish to segment and/or plot
    MY_NEW_FOLDER -- Absolute filepath to the folder to where segmented images and/or plots and data will be exported
    MY_VALID_FOLDER -- Absolute filepath to folder containing valid_lab.pkl and valid_rgb.pkl
    AMPLITUDES -- Amplitudes of each axis [L a b]; axes extend from 0 to L, -a to a, and -b to b
    DIMENSIONS -- Dimensions of bins in CIELAB space [L a b]; all pixels within confines of a bin take on the same color value
    SEG -- If True, segment each image using an approximation of the Lin et. al, 2013 method http://vis.stanford.edu/papers/semantically-resonant-colors
    TAU -- Segmentation parameter, in CIELAB units
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
    L_list, a_list, b_list = [],[],[] # Destination for pixel values
    unique_bins = {} # Dictionary for unique bins (2D histogram). Key is CIELAB value, value is absolute frequency.
    total_images, images_skipped = 0, 0

    # Vector for each axis
    L_vec, a_vec, b_vec = np.linspace(0, L_amp, Lbins+1), np.linspace(-a_amp, a_amp, abins+1), np.linspace(-b_amp, b_amp, bbins+1)

    with open(my_valid_folder + '/valid_lab.pkl', 'rb') as pickle_load:
        valid_lab = pickle.load(pickle_load)
    with open(my_valid_folder + '/valid_rgb.pkl', 'rb') as pickle_load:
        valid_rgb = pickle.load(pickle_load)
    #************************
    """DEFINING FUNCTIONS"""
    #************************
    def bounder_v2(x, v):
        """Take x and evenly-spaced ordered vector, return list of bin coordinates."""

        x0 = v[0] # minimum value of vector
        w = v[1]-x0 # width of a bin on given axis
        binnum = ceil((x-x0)/w) # number of bins is distance between x & x0, divided by bin width

        # edge case
        if binnum == 0:
            binnum = 1 # check to make sure this works

        return binnum

    def binner_v2(L_input, a_input, b_input):
        """Take an LAB value, axis vectors, return linear index of 3D bin."""

        # position of bin on each axis
        L_bin = bounder_v2(L_input, L_vec)
        a_bin = bounder_v2(a_input, a_vec)
        b_bin = bounder_v2(b_input, b_vec)

        return [L_bin, a_bin, b_bin]

    def sub2ind(ypos, xpos):
        """Take a 2D matrix coordinates, return linear index."""

        linear_index = imagewidth * ypos + xpos # imagewidth is defined on a per-image basis
        return linear_index

    def ind2sub(linear_index):
        """Take linear index, return 2D matrix coordinates."""

        ypos = linear_index // imagewidth
        xpos = linear_index % imagewidth
        return (ypos,xpos)

    def neighbors(ypos, xpos):
        """Find all 8 neighboring coordinates to given coordinates."""

        # could do this programmatically
        top_left = [ypos-1, xpos-1]
        top = [ypos-1,xpos]
        top_right = [ypos-1, xpos+1]
        left = [ypos,xpos-1]
        right = [ypos,xpos+1]
        bottom_left = [ypos+1, xpos-1]
        bottom = [ypos+1, xpos]
        bottom_right = [ypos+1, xpos+1]
        return [top_left, top, top_right, left, right, bottom_left, bottom, bottom_right]

    def bins2lab(bin_list):
        """Take bin_list [Lbin, abin, bbin] and return [L, a, b]."""

        L = Lw*bin_list[0]-Lw/2
        a = -a_amp+aw*bin_list[1]-aw/2
        b = -b_amp+bw*bin_list[2]-bw/2
        return [L, a ,b]

    def grouper(ypos, xpos):
        """Group pixels using approximation of Lin et al., 2013 method. Variables unspecified in this function body are nonlocally defined."""

        my_ind = sub2ind(ypos,xpos) # get linear index of coordinate
        if my_ind in linear_array: # if pixel is still ungrouped
            neighbor_list = neighbors(ypos,xpos) # find neigboring pixels
            for neighbor in neighbor_list:
                n_ypos, n_xpos = neighbor[0], neighbor[1]
                if 0 <= n_ypos < imageheight and 0 <= n_xpos < imagewidth: # if neighboring pixels in image dimensions
                    dist = np.linalg.norm(np.array(lab_array[ypos,xpos])-np.array(lab_array[n_ypos,n_xpos])) #calculate distance between neighbor and given pixel
                    if dist <=tau and my_array[n_ypos,n_xpos] == num_groups: # if distance smaller than tau and there is currently one connected component
                        linear_array.remove(my_ind) # remove grouped pixel from to-be-grouped
                        my_array[ypos,xpos] = my_array[n_ypos,n_xpos] # give neighboring pixel same value as given pixel
                        nonlocal num_grouped
                        num_grouped +=1
                        break # don't need to look at any other neighbors

    def no_seg_bin(ypos, xpos):
        """Bin pixels from given image without segmentation. Variables unspecified in this function body are nonlocally defined."""

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


    #*********************
    """DATA COLLECTION"""
    #*********************
    for my_image in os.listdir(my_input_folder): # iterate through folder of images
        total_images += 1
        border_pixels = [] # initialize list for border pixels
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
        rgb_img = rgb_img/255 # np.array(height, width, [RGB])
        try:
            lab_img = color.rgb2lab(rgb_img) # np.array(height, width, [LAB])
        except ValueError as error:
            print(error, ';', '%s was skipped' % my_image)
            images_skipped += 1
            continue


        imshape = lab_img.shape # (height, width, depth)
        imageheight, imagewidth = imshape[0], imshape[1]

        # initialize imageheightximagewidth array of lab values
        lab_array = np.zeros((imageheight,imagewidth), dtype="object")

        # iterate through all pixels, add CIELAB values to unique bins and to lab_array
        for xpos in range(imagewidth):
            for ypos in range(imageheight):
                Linput, ainput, binput = lab_img[ypos,xpos][0], lab_img[ypos,xpos][1], lab_img[ypos,xpos][2]
                lab_array[ypos,xpos] = [Linput, ainput, binput]

                #***************************************************************
                foo = color.lab2rgb(np.array([[[Linput, ainput, binput]]]))[0][0]
                if foo[0] < 0 or foo[0] > 1 or foo[1] < 0 or foo[1] > 1 or foo[2] < 0 or foo[2] > 1:
                    print('HELP')
                #***************************************************************

        if seg:
            # Iterate through border pixels |=|, add CIELAB values to border_pixels list
            for xpos in range(1,imagewidth-1):
                for ypos in [0,imageheight-1]:
                    Linput, ainput, binput = lab_img[ypos,xpos][0], lab_img[ypos,xpos][1], lab_img[ypos,xpos][2]
                    border_pixels.append((Linput, ainput, binput))
            for ypos in range(imageheight):
                for xpos in [0,imagewidth-1]:
                    Linput, ainput, binput = lab_img[ypos,xpos][0], lab_img[ypos,xpos][1], lab_img[ypos,xpos][2]
                    border_pixels.append((Linput, ainput, binput))

            # b/w background exists if >= 75% of border pixels are within tau=3 of b/w (Lin et. al)
            white, black = np.array((100,0,0)), np.array((0,0,0))
            border_dist_white = [np.linalg.norm(pixel-white) for pixel in border_pixels]
            border_dist_black = [np.linalg.norm(pixel-black) for pixel in border_pixels]
            border_bool_white = [dist<=tau for dist in border_dist_white]
            border_bool_black = [dist<=tau for dist in border_dist_black]
            whiteborder = sum(border_bool_white)/len(border_bool_white) >= 0.75 #boolean
            blackborder = sum(border_bool_black)/len(border_bool_black) >= 0.75 #boolean
            border = whiteborder or blackborder

            # segmentation if a border exists
            if border:
                print('%s has a border' % my_image)
                #lab_array = np.zeros((imageheight,imagewidth), dtype="object")
                linear_array = list(range(imageheight*imagewidth)) # linear indices of coordinates for to-be-grouped pixels
                my_array = np.zeros((imageheight,imagewidth), dtype="int") # representation of segmentation
                lab_array[ypos,xpos] = [Linput,ainput,binput]
                my_array[0,0] = 1 # leftmost, topmost pixel assumed to have the same color as the background
                my_array[imageheight-1,imagewidth-1] = 1
                linear_array.remove(0) # have just "looked" at leftmost, topmost pixel
                linear_array.remove(imageheight*imagewidth-1)
                num_groups = 1 # of different connected components (1, 2,...)
                num_grouped = 2 # of pixels in a connected component (left-topmost, bottom-rightmost)

                #start from top left of image
                for ypos in range(imageheight):
                        for xpos in range(imagewidth):
                            grouper(ypos, xpos)

                #start from bottom right of image
                for ypos in list(range(imageheight))[::-1]:
                        for xpos in list(range(imagewidth))[::-1]:
                            grouper(ypos, xpos)

                # if there are still ungrouped values, there must be at least 2 connected components
                if linear_array:
                    num_groups += 1

            # if there is a background and there are at least two connected components
            if border and num_groups > 1:
                for xpos in range(imagewidth):
                    for ypos in range(imageheight):
                        if my_array[ypos, xpos] == 1: # ignore background pixels
                            continue
                        else:
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

                plt.close()
                plt.imshow(my_array)
                try:
                    plt.savefig(my_new_folder + '/' + my_image + '.svg', format='svg')
                except ValueError as error:
                    print(my_image + ' was processed but not saved')

            # image fails to meet two conditions (background and >=2 connected components)
            else:
                for xpos in range(imagewidth):
                    for ypos in range(imageheight):
                        no_seg_bin(ypos, xpos)

        # seg==False
        else:
            for xpos in range(imagewidth):
                for ypos in range(imageheight):
                    no_seg_bin(ypos, xpos)

    print('Out of %s total images seen, %d were skipped' % (total_images,images_skipped))


    #*******************
    """VISUALIZATION"""
    #*******************
    # for 3D histogram
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

    # sorted by descending frequency
    s_vardensity = sorted(vardensity, reverse=True)
    s_varlist = [bin for _,bin in sorted(zip(vardensity,varlist), reverse=True)]

    if 0 in s_vardensity:
        last = s_vardensity.index(0)
        s_varlist, s_vardensity = s_varlist[:last+1], s_vardensity[:last+1]
        print("Color-density pairs with density=0 have been removed")
    my_colors_valid = []

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

    #basically a linked list
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

    # to thin out plot, use a random sample
    if sample:
        s_c_varlist, s_c_vardensity = zip(*random.sample(list(zip(s_c_varlist, s_c_vardensity)), sample_size))

    x, y, z = [], [], []
    for lab in s_c_varlist:
        x.append(float(lab[0])) #L
        y.append(float(lab[1])) #a
        z.append(float(lab[2])) #b

    # color points by position in CIELAB space
    colors = []
    for i in range(len(s_c_varlist)):
        lab_color = s_c_varlist[i] # already binned
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
    ax.set_xlim([-b_amp,b_amp])
    ax.set_ylim([-a_amp,a_amp])
    ax.set_zlim([0, L_amp])
    plt.title(title)
    plt.savefig(my_new_folder + '/' + 'histogram 3D' + '.svg', format='svg', bbox_inches='tight')

    with open(my_new_folder + '/' + 'colors.pkl', 'wb') as pickle_file:
        pickle.dump(s_varlist, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(my_new_folder + '/' + 'densities.pkl', 'wb') as pickle_file:
        pickle.dump(s_vardensity, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
