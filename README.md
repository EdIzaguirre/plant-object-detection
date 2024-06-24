# Detecting Plant Disease
This is a production-grade machine learning application that uses object detection to localize and classify diseases in plants. 

# Data Source
The data for this project was found on 

# Metaflow
This project uses Metaflow as the orchestration tool and infrastructure abstraction tool. Below you can see what tools are used for this project.

- **Data transformation: KerasCV**. This is a horizontal extension of the Keras library that includes many
  helpful abstractons for computer vision tasks, in this case object detection. The pre-trained RetinaNet model is also obtained from the KerasCV library.
- **Model training: AWS Batch**: Compute for fine-tuning is provided via AWS Batch. This is a fully managed service by Amazon that can dynamically provision the
  appropriate compute instances for our training task. Using you are using the provided CloudFormation template, this will be accomplished via a
  [p3.2xlarge instance]([url](https://aws.amazon.com/ec2/instance-types/)).
- **Model evaluation: Weights and Biases:** All training metrics including MaP, training loss, and ground truth vs. predicted bounding boxes are logged in WandB.
- **Model deployment: AWS Sagemaker:** Trained model is sent to an AWS Sagemaker endpoint, where it can provide predictions for images submitted into the application.
