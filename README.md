# colorhist

## Dependencies
[Figure-ground segmentation by transferring window masks](http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/)

[Google Images Download](https://github.com/hardikvasa/google-images-download)

## Window segmentation for googleimage_seg_func_v2.py
1. Follow README provided [here](http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/)
2. If on macOS:
   - In ../segtrans/objectness/MEX/computeScoreContrast.c, change ~line 72: `mxERR…` to `mexERR…`
   - In ../segtrans/objectness/computeScores.m, change ~line 159: `I = imread([imgBase ‘.ppm’]);` to 
   `I = imread([imgBase ‘.jpg’]); imwrite(I, [imgBase ‘.ppm’])`
   - Run maxflow_make.m (../segtrans/maxflow/maxflow_make.m)
3. Place segmenter_func.m and figureground.m into your MATLAB folder
4. Download voc10.zip from above website, place contents into ../segtrans/data/xps
5. Run `segmenter_func()` in segmenter_func.m
6. Use NEW_FOLDER filepath from segmenter_func.m as SEG_FOLDER filepath in googleimage_seg_func_v2.py

## Which method to use?
For landscape queries (ex. ‘ocean’, ‘forest’, ‘avalanche’), use googleimage_lin_func_v2.py (set `TAU=2`). Most images won’t be segmented, but using googleimage_seg_func_v2.py can lead to mostly meaningless figure-ground segmentation.

For object queries (ex. ‘lemonade’, ‘scissors’, ‘watch’), default to using window segmentation (segmenter.m, googleimage_seg_func_v2.py), as it does a better job of segmenting. If the background plots seem to have large amounts of salient color values (not white/black), try using googleimage_lin_func_v2.py (`TAU=2`) to include background colors.

## Parameter explanations
`TAU` -- googleimage_lin_func_v2.py uses this value for segmentation. [Lin et. al](http://vis.stanford.edu/papers/semantically-resonant-colors) use `TAU=3`, but we recommend `TAU=2`, as meaningful colors closer to white are not removed in queries such as "milkshake" or "lemonade."

`DIMENSIONS` -- googleimage_lin_func_v2.py and googleimage_seg_func_v2.py use this value. Represents dimensions of bins in CIELAB space; all pixels within confines of a bin take on the same color value. Ex. setting DIMENSIONS=[5, 5, 5] will give all 125 pixels in any given CIELAB bin the same CIELAB value.

