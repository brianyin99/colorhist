function segmenter_func(myfolder, new_folder, segtrans_path)
%{ 
Take folder of images, segment using: http://calvin.inf.ed.ac.uk/software/figure-ground-segmentation-by-transferring-window-masks/
Use figureground.mA

MYFOLDER -- Absolute filepath to folder of images to segment
NEWFOLDER -- Name of new folder (will appear in current MATLAB folder)
SEGTRANS_PATH -- Absolute filepath to segtrans folder downloaded from website above.

%}

mkdir(new_folder)

% loops through every image in myfolder
myfiles = dir(myfolder);
num_images = length(myfiles) - 2;
for i = 3:num_images + 2
    
    % reverse-engineer the file path for each image
    myfile = myfiles(i);
    myname = myfile.name;
    inputimage = strcat(myfolder,'/', myname);
    
    % for when myname is '.DS_Store' (info mac stores in folders)
    if myname(2) == 'D'
        continue
    end
    
    % read-in rgb image
    rgbimage = imread(inputimage);
    
    % takes care of indexed images
    if class(rgbimage) == 'uint8'
        [X, map] = imread(inputimage);
        if ~isempty(map)
            rgbimage = ind2rgb(X,map);
        end
    end
    
    % find prediction for figure-ground, imageheightximagewidth logical
    prediction = figureground(rgbimage, segtrans_path);
    imwrite(prediction, fullfile(new_folder, strcat(myname,'.png')))
end
