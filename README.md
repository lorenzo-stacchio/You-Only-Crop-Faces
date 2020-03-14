# Single Face Detector
Deep neural network model developed in colab with [@Lemonzino](https://github.com/Lemonzino) as our Machine Learning course project at Alma Mater Studiorum University.
This model is able to recognize the presence of a face in a pic and guess the area of photo in which it is.

## Model architecture
The final model is a parallel Convolutional Neural Network composed by two submodules:
- Face presence classifier;
- Face area detector;

### Face presence classifier
This submodel has the duty to classify if there is a face or not in the examined pic.
The classification is a real number in range `[0,1]` and the confidence that i chosen for this detector  `>= 0.5`.
This model is composed by a CNN followed by a layer dense to solve the classification task.
This model is a TF model, that use [MobileNetV2 architecture](https://arxiv.org/pdf/1801.04381.pdf) pretrained with [image-net](http://www.image-net.org/).
It was trained using two datasets: 
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html);
- [256 Object categories](http://www.vision.caltech.edu/Image_Datasets/Caltech256/).


### Face area detector
This submodel is a regressor that guess a rectangle in the photo that contains an eventual face.
The guess has the following format: 
`[x_top_left, y_top_left, width, height]`
This model was built using a Fully Convolution approach.
This model is a TF model, that use [MobileNetV2 architecture](https://arxiv.org/pdf/1801.04381.pdf) pretrained with [image-net](http://www.image-net.org/).
It was trained using a single dataset: 
- [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

## Requirements
- Support for tensorflow GPU (this should probably work on tensorflow CPU too, but not tested);
- Python 3.6 or greater;
- All pip packages listed in [requirements](requirements.txt).

## How to test the model with a webcam
After you got a camera you only need to access the root project directory and launch
`python webcam_face_detector.py`.

## How to test the model without a webcam
You only need to access the root project directory and launch:
`python image_face_detector.py input_image_path outputdir`.

# Problems and future improvement
The classifier submodel hasn't a great accuracy because of the low variation of faces that were feed to it in training phase.
Indeed you can test the model without the classifier by simply commenting and moving a few lines of code in 
`webcam_face_detector.py` or in `input_image_path outputdir` and you should see a greater performance in detection.
The future improvement is to retrain the classifier submodule with an increased dataset that contains a greater variation in faces position and orientation.


