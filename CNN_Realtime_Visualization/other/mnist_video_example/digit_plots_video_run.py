import params
import os
import digit_plots_video

imgfolder = params.IMAGES_FOLDER
i = 0;
for filename in os.listdir(imgfolder):
    digit_plots_video.main(filename)
    i += 1
    print(i)
    if (i == 20):
        break;

