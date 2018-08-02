function [prediction] = figureground(rgbimage, segtrans_path)
%{ Take image (imageheightximagewidthxRGB), return image segmented. Must be on same path as segmenter_func.m 
%}

addpath(genpath(segtrans_path)) % path to segtrans folder (must be run from segtrans)

%% Load the prepared training data of voc10.
source = load_value('data/xps/voc10/train_source.mat');

%% Segment a single image
image = rgbimage; % cell of RGB uint8 images
windows = st_windows(image,100);
features = st_features_gist(image,windows);
weights = ones(1,100)/100;
mask = st_transfer(source,50,windows,features,@distance_L2,weights);
prediction = st_segment(image,mask,0.2,20);

%% Display the results.
% show_prediction(prediction,image,mask);

end

