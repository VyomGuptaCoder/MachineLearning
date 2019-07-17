Machine Learning CS 6375 Project On Deep Dark Light.

Team members:
Swapnil Bansal sxb180020
Harshel Jain hxj170009
Vyom Gupta fxv180000
Lipsa Senapati lxs180002

The Folder has 3 files:
accuracy_metrics.py - To calculate the PSNR and SSIM of the output image vs the real image.
test_Dark_Images.py - To test the model on testing dataset.
train_Dark_Images.py - To train the model on the training dataset.

Dataset can be downloaded from the link : https://drive.google.com/drive/folders/1_Tq5bOGQx2cm7tb_LSEQmZOVoDUpcmhj?usp=sharing
Trained Model can be downloaded from: https://drive.google.com/drive/folders/1OmgVO-RESKJALZUfJb1XiN9aICP4uP1G?usp=sharing
The output images and intermediate Images can be viewed at this link: https://drive.google.com/drive/folders/1pS8-0E7NXXtsv1-KDQNHoJZQeAMNw2E5?usp=sharing

Install the required Libraries by using the python pip command.
Library required:
os
scipy.io
tensorflow
tensorflow.contrib.layers
tensorflow.contrib.slim
numpy
rawpy
glob
PIL
skimage.io
skimage.measure
math
cv2

How to Run and Compile:
1. Download the folder and extract it to Desktop.
2. Download the dataset from the link provided. (Extract it in the same format and keep it inside the Canon folder only)
3. Start the terminal instance on the folder.
4. Start the training of the model by running the command: python train_Dark_Images.py
The model will be trained and will be available at the checkpoint folder. (Our trained model can also be downloaded from the link provided above)
5. Start the model test and prediction by running the command: python train_Dark_Images.py
The output will be generated in 4 different folders as Dark Image(Raw Input image scaled with white balance), Scaled Image(Intermediate Output), Output Image(The predicted image from the model), Groundtruth Image(Real Image or expected output)
6. Test the model and accuracy metrics by running the command: python accuracy_metrics.py

The output obtained by running the accuracy file is attached along with the code file.
