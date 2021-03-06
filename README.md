# Building Color Histograms

## Background
Color-concept assignment is the process by which people build associations between concepts (things and ideas, e.g. "banana", "freedom", "car"), and colors. In order to estimate the strengths of the connections between concepts and colors to serve as a baseline for future experiments, we analyze the frequencies of colors from images scraped from Google queries of concepts (scroll to the bottom for examples).

This repository contains functions that create color histograms from input images over CIELAB space. We use [figure-ground segmentation](https://en.wikipedia.org/wiki/Figure%E2%80%93ground_(perception)) by [Kuettal & Ferrari](http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/) and previous work by [Lin et al.](http://vis.stanford.edu/papers/semantically-resonant-colors) as a framework for our project.


## Dependencies
- [Figure-ground segmentation by transferring window masks](http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/) - required for figureground.m segmentation
- [MATLAB API for Python](https://www.mathworks.com/help/matlab/matlab-engine-for-python.html) - `matlab.engine` module in googleimage_lin_func_v2.py and googleimage_seg_func_v2.py
- [Google Images Download](https://github.com/hardikvasa/google-images-download) - We use the output folder of thumbnail images from this tool as `MY_FOLDER` in segmenter_func.m and as `MY_INPUT_FOLDER` in googleimage_lin_func_v2.py and googleimage_seg_func_v2.py
- [fastKDE](https://bitbucket.org/lbl-cascade/fastkde) - `fastkde.fastKDE` module in googleimage_lin_func_v2.py and googleimage_seg_func_v2.py
- Various other Python packages: numpy, scipy, matplotlib, skimage, mpl_toolkits

## Window Segmentation Setup
1. Follow README provided [here](http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/)
2. If on macOS:
   - In ../segtrans/objectness/MEX/computeScoreContrast.c, change ~line 72: `mxERR…` to `mexERR…`
   - In ../segtrans/objectness/computeScores.m, change ~line 159: `I = imread([imgBase ‘.ppm’]);` to 
   `I = imread([imgBase ‘.jpg’]); imwrite(I, [imgBase ‘.ppm’]);`
   - Run maxflow_make.m (../segtrans/maxflow/maxflow_make.m)
3. Download voc10.zip from above website, place contents into ../segtrans/data/xps

## Usage

- For landscape queries (ex. ‘ocean’, ‘forest’, ‘avalanche’), use googleimage_lin_func_v2.py, an approximation of the Lin et al. method. Most images won’t be segmented, but using googleimage_seg_func_v2.py can lead to mostly meaningless figure-ground segmentation.
- For object queries (ex. ‘lemonade’, ‘scissors’, ‘watch’), default to using window segmentation (segmenter.m, googleimage_seg_func_v2.py), as it does a better job of segmenting. If the background plots seem to have large amounts of salient color values (not white/black), try using googleimage_lin_func_v2.py to include background colors.
- Note: valid_lab.pkl contains a pickled list of all CIELAB values that have corresponding values in the RGB gamut. valid_rgb.pkl contains a pickled list of all RGB values for lookups.

<img width="511" alt="artboard 1 2x" src="https://user-images.githubusercontent.com/41968577/43601194-5be9b00a-9652-11e8-8f7d-214dd5ea1f94.png">

## Parameter Explanations
`TAU` -- googleimage_lin_func_v2.py uses this value for segmentation. [Lin et al.](http://vis.stanford.edu/papers/semantically-resonant-colors) use `TAU=3`, but we recommend `TAU=2`, as meaningful colors closer to white are not removed in queries such as "milkshake" or "lemonade."

`DIMENSIONS` -- googleimage_lin_func_v2.py and googleimage_seg_func_v2.py use this value. Represents dimensions of bins in CIELAB space; all pixels within confines of a bin take on the same color value. Ex. setting `DIMENSIONS=[5, 5, 5]` will give all 125 pixels in any given CIELAB bin the same CIELAB value.

## Future Work
- Plotting KDE over values only found in original binning proccess could be faster than using `spatial.cKDTree` on valid_lab.pkl. Helpful links: [1](https://stackoverflow.com/questions/40756024/python-fastkde-beyond-limits-of-data-points), [2](https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates)
- The [exact methods](https://github.com/StanfordHCI/semantic-colors) of Lin et al. seem to depend on .dll files missing from [Google CustomSearch C# Library v1.3.0](https://github.com/google/google-api-dotnet-client/releases/tag/1.3.0-beta). Getting their methods working would provide valuable distribuitions to which our results could be compared.

## Acknowledgments

Special thanks to Karen Schloss and Laurent Lessard at University of Wisconsin - Madison

## Extras

#### Example results
![artboard 1 1 5x](https://user-images.githubusercontent.com/41968577/43612803-0051b348-9673-11e8-9db9-bf8040cb1e4e.png)

#### Figure-Ground Segmentation for Separating Backgrounds
![artboard 1](https://user-images.githubusercontent.com/41968577/43655670-3873c502-9715-11e8-8721-9605c6bd4819.png)
