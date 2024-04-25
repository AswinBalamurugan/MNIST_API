Please refer: [My Project Collection](https://github.com/AswinBalamurugan/Machine_Learning_Projects/blob/main/README.md)

# Objective
Understand and implement FastAPI for handwritten digit classification. 
FastAPI is an open-source web framework for creating APIs with Python that's used for data science and e-commerce applications.

# Purpose
The purpose of this project is to build a FastAPI application that exposes the functionality of an MNIST digit classification model over a RESTful API. 
Users can upload images of handwritten digits, and the API will predict the digit based on the trained model.

# Dataset
- Task 1 images were downloaded from the attached hugging face [link](https://huggingface.co/datasets/mnist)
- Task 2 images were created by taking screenshots of hand-drawn digits on a touch screen.

# Steps to execute the API
- Open the command line terminal in the working directory of containing all the python file and the model.
- For task 1, type `python ch20b018_task1.py best_model.keras` and for task 2, type `python ch20b018_task2.py best_model.keras`
- Then open your web browser and type `http://0.0.0.0:8000/docs`.
- Based on the chosen task, upload the acceptable images.

# Brief Description
This project consists of a FastAPI application that loads a pre-trained MNIST digit classification model. 
The application provides an endpoint (/predict) where users can upload an image of a handwritten digit. 
The image is preprocessed, and the model predicts the digit based on the processed image data. 
The predicted digit is then returned to the user as a JSON response.

- For `ch20b018_task1.py`, upload images of size *28 x 28*.
- For `ch20b018_task2.py`, any image with a single digit will work.

# Conclusions
The API is able to classify the digit images which were used. 
Since the model was built on mnist dataset which has images with white text and black background, it was not able to identify the **digit** image correctly.
The model was also unable to accurately predict the images **digit-4** & **digit-7**.
Refer [link]() for the prediction results.
