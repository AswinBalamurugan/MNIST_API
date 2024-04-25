Please refer: [My Project Collection](https://github.com/AswinBalamurugan/Machine_Learning_Projects/blob/main/README.md)

# Objective
Understand and implement FastAPI for handwritten digit classification. 
FastAPI is an open-source web framework for creating APIs with Python that's used for data science and e-commerce applications.

# Purpose
This project aims to build a FastAPI application that exposes the functionality of an MNIST digit classification model over a RESTful API. 
Users can upload images of handwritten digits, and the API will predict the digit based on the trained model.

# Dataset
- Task 1 images were downloaded from the attached hugging face [link](https://huggingface.co/datasets/mnist).
- Task 2 images were created by taking screenshots of hand-drawn digits on a touch screen.

# Steps to execute the API
- Open the command line terminal in the working directory containing all the Python files and the model.
- For task 1, type `python ch20b018_task1.py best_model.keras`. 
- For task 2, type `python ch20b018_task2.py best_model.keras`.
- Then open your web browser and type `http://0.0.0.0:8000/docs`.
- Based on the chosen task, upload the acceptable images.

# Brief Description
This project is a FastAPI application that loads a pre-trained MNIST digit classification model. 
The application provides an endpoint (/predict) where users can upload an image of a handwritten digit. 
The image is preprocessed, and the model predicts the digit based on the processed image data. 
The predicted digit is then returned to the user as a JSON response.

- For `ch20b018_task1.py`, upload images of size *28 x 28*.
- For `ch20b018_task2.py`, any image with a single digit will work.

# Conclusions
The API can classify the digit images that were used. 
The model performed well on images for task 1 since those resembled the images on which the model had been trained.
Since the model was built on the MNIST dataset, which has images with white text and black backgrounds, it could not correctly identify the **digit** image.
The model could also not accurately predict the images **digit-4** & **digit-7**.
- Refer [link1](https://github.com/AswinBalamurugan/MNIST_API/tree/main/predictions/task1) for task 1 prediction results.
- Refer [link2](https://github.com/AswinBalamurugan/MNIST_API/tree/main/predictions/task2) for task 2 prediction results.

# FastAPI interface

![img1](https://github.com/AswinBalamurugan/MNIST_API/blob/main/predictions/interface-1.png)
![img2](https://github.com/AswinBalamurugan/MNIST_API/blob/main/predictions/interface-2.png)
