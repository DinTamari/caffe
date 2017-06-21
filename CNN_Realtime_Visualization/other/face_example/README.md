(See citations belows)

# Convolutional Neural Network Visualization of a Realtime Classifier

Din Tamari  
Bachelor Project, USI Lugano, 2017  
Advisor: Luca Gambardella  
Tutor/Assistant: Alessandro Giusti  

# Face-Age classification example, see: https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/


1. Need to download the 'Apparent age estimation model' .caffemodel file into directory
2. age.prototxt (age_deploy.prototxt) file already available in directory
3. 4 images supplied for demo (for more, see 00 directory)  
   If using other images, may need to preprocess before classification.  
   See classify function in age_test.py
4. run: > python age_test.py


For any questions or issues, please contact me:  
tamari.din@gmail.com


-------------------------------------------------------------------------------------------------

Using ETHZ Data Vision's DEX: Deep EXpectation of apparent age from a single image
@article{Rothe-IJCV-2016,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {Deep expectation of real and apparent age from a single image without facial landmarks},
  journal = {International Journal of Computer Vision (IJCV)},
  year = {2016},
  month = {July},
}

@InProceedings{Rothe-ICCVW-2015,
  author = {Rasmus Rothe and Radu Timofte and Luc Van Gool},
  title = {DEX: Deep EXpectation of apparent age from a single image},
  booktitle = {IEEE International Conference on Computer Vision Workshops (ICCVW)},
  year = {2015},
  month = {December},
}
