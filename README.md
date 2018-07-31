# colorhist

## GET WINDOW SEGMENTATION WORKING FOR GOOGLEIMAGE_SEG_FUNC.PY

1. Follow README provided [here](http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/)
2. If on macOS:
   - In ../segtrans/objectness/MEX/computeScoreContrast.c, change ~line 72: `mxERR…` to `mexERR…`
   - In ../segtrans/objectness/computeScores.m, change ~line 159: `I = imread([imgBase ‘.ppm’]);` to 
   `I = imread([imgBase ‘.jpg’]); imwrite(I, [imgBase ‘.ppm’])`
   - Run maxflow_make.m (segtrans>>maxflow>>maxflow_make.m)
3. Place segmenter_func.m and figureground.m into your MATLAB folder
4. Download voc10.zip from above website, place contents into ../segtrans/data/xps
5. Run segmenter_func() with specified arguments
6. Use NEWFOLDER filepath from segmenter_func.m as SEG_FOLDER filepath in googleimage_seg_func.py


## WHICH METHOD TO USE?
Landscape queries (ex. ‘Ocean’, ‘forest’, ‘avalanche’): use googleimage_lin_func.py (set tau=2).
Not a big difference between t=2 and t=3 as most images won’t be segmented, but
using googleimage_seg_func.py will lead to mostly meaningless figure-ground segmentation.

Object queries (ex. ‘Lemonade’, ‘scissors’, ‘watch’): default to using window
segmentation (segmenter.m, googleimage_seg_func.py), as it does a better job of
segmenting than Lin’s color segmentation method. If you suspect that the background
colors may be meaningful, check the background plots that googleimage_seg_func.py produces.
If the background plots seem to have large amounts of salient color values (not white/black),
use googleimage_lin_func.py (set tau=2) to include background colors.


## PARAMETER EXPLANATIONS
TAU -- googleimage_lin_func.py uses this value for segmentation. In Lin et al., 2013,
an image is described as having a black or white background if at least 75% of
the border pixels of the image are within TAU of black or white, respectively.
Backgrounds were only removed from images that had "at least two connected components...
adjacent pixels are part of the same connected component if their colors are within
distance threshold [TAU]. We [Lin et al.], set [TAU] to 3 CIELAB units..." We
recommend setting TAU=2 instead, as meaningful colors closer to white are not removed
in queries such as "milkshake" or "lemonade."


DIMENSIONS -- googleimage_lin_func.py and googleimage_seg_func.py use this value.
Represents dimensions of bins in CIELAB space; all pixels within confines of a bin
take on the same color value. Ex. setting DIMENSIONS=[5, 5, 5] will give all 125
pixels in any given CIELAB bin the same CIELAB value.

