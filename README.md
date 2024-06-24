# Detecting Plant Disease
Can we use machine learning to detect diseases in plants?

![An example of a disease detection](images/example_detection.png)

## Overview
This is a production-grade machine learning application that uses object detection to localize and classify diseases in plants. This is meant to be an educational project for machine learning engineers
to learn how to create an end-to-end computer vision application using Metaflow as the infrastructure abstraction tool. Inspiration was taken from the notable 
[You Don't Need a Bigger Boat](https://github.com/jacopotagliabue/you-dont-need-a-bigger-boat) repo, which shows a more complicated flow based on training a recommendation model.

## Data Source
The data for this project is from the [PlantDoc dataset](https://public.roboflow.com/object-detection/plantdoc?ref=blog.roboflow.com), published on the Roboflow website. 
It contains 2,569 images across 13 plant species and 30 classes (diseased and healthy) for image classification and object detection. It is under a CC BY 4.0 license, which allows
for the remixing, transforming, and building upon the images for any purpose, even commercially.

## Metaflow
This project uses Metaflow as the orchestration tool and infrastructure abstraction tool. Below you can see what tools are used for this project.

![Flow used for this project](images/flow_image.png)

- **Data transformation: KerasCV**. This is a horizontal extension of the Keras library that includes many helpful abstractons for computer vision tasks, in this
  case object detection. The pre-trained RetinaNet model is also obtained from the KerasCV library.
- **Model training: AWS Batch**. Compute for fine-tuning is provided via AWS Batch. This is a fully managed service by Amazon that can dynamically provision the
  appropriate compute instances for our training task. If you are using the provided CloudFormation template, this will be accomplished via a
  [p3.2xlarge instance](https://aws.amazon.com/ec2/instance-types/).
- **Model evaluation: Weights and Biases**. All training metrics including MaP, training loss, and ground truth vs. predicted bounding boxes are logged in WandB.
- **Model deployment: AWS Sagemaker**. The trained model is sent to an AWS Sagemaker endpoint, where it can provide predictions for images submitted into the application.
