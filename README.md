This project is still a work in progress and follows the textbook "Deep Learning With PyTorch"'s chapters 9-14. It seeks to create a tool for malignant tumor detection in lungs using a CT scan of a patient's chest as input.
The general process can be broken down into 5 steps:
  1) Load the CT scan data into a form compatible with PyTorch
  2) Identify voxels of potential tumors in the lungs using segmentation
  3) Group interesting voxels into candidate nodules
  4) Classify candidate nodules as actual nodules or non-nondules using 3D convolution
  5) Provide a diagnosis using the per-nodule classifications

The code files are broken up into different pieces as the project has progressed. The models can be found in model.py, while dataset implementations can be found in the various dsets files.
dsets_pre_seg.py contains the original dataset for testing the classificaiton model. The split for training and validation sets were done on the list of nodules, while the segmentation datasets split the list of CT scans.
Because of this, an updated set for classification is found in dsets_classification.py to use when unifying the two models. The original training code for the classification model and the segmentation model training code is found in training.py, 
while an updated version uses the new dataset split. 

The unification of the two models is the final step, and I am currently working through the final chapter of the project to complete this goal. Please note
that this is still a work in progress, and jupyter notebooks still need to be completed and ran once the whole pipeline is working, and various data files still need to be pushed.

Textbook github: https://github.com/deep-learning-with-pytorch/dlwpt-code
