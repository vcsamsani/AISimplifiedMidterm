# Malware Detection App

## Project Overview

Welcome to our mid-term project for DSCI 6015, focusing on Cloud-based PE Malware Detection App. This project forms a crucial part of our Artificial Intelligence and Cybersecurity course for the Spring 2024 semester at the University of New Haven, under the guidance of Professor Vahid Behzadan.

## Project Description

Our project revolves around implementing and deploying a machine learning model designed for classifying Portable Executable (PE) files as either malware or benign. We have chosen to utilize the Random Forest Classifier for this purpose. The project is structured into three primary tasks:

### Task 1 – Model Development and Training

The initial task involves building and training the model on AWS SageMaker. This entails data preparation, model definition, and training utilizing SageMaker's distributed computing capabilities. Post-training, the model's performance is assessed, and if satisfactory, it's deployed as an endpoint for real-time predictions. SageMaker streamlines this process by providing managed infrastructure, performance monitoring, and automation tools.

### Task 2 - Model Deployment as a Sagemaker Endpoint (API)

Upon successful training, the model is deployed as a cloud API using AWS Sagemaker. This facilitates other applications to utilize the model in real-time for malware classification.

### Task 3 – Web Client Development

The final task involves developing a user-friendly web application employing Streamlit. Users can upload PE files (.exe), which are then converted into a feature vector. This feature vector is forwarded to the deployed API, and the results (Malware or Benign) are promptly displayed to the user.

## Getting Started

To embark on this journey, clone this repository to your local machine and meticulously follow the instructions provided for detailed implementation steps.

## Additional Resources

Our report contains additional resources for reference purposes. Feel free to explore and tweak the code to further enhance the project.

Video link of presentation : https://youtu.be/gBJi2cwvYgk
