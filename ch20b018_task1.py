import sys
import uvicorn
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from keras.models import Sequential
from keras.models import load_model as keras_model
import unittest
from unittest.mock import patch, MagicMock

app = FastAPI()

# 2. Take the path of the model as a command line argument
if len(sys.argv) < 2:
    raise ValueError("Please provide the path to the model file as a command-line argument.")
model_path = sys.argv[1]

# 3. Create a function "def load_model(path:str) -> Sequential" which will load the model saved at the supplied path on the disk and return the keras.src.engine.sequential.Sequential model.
def load_model(path: str) -> Sequential:
    """
    Load the pre-trained Keras model from the specified path.
    
    Args:
        path (str): The path to the saved model file.
        
    Returns:
        Sequential: The loaded Keras Sequential model.
    """
    return keras_model(path)

# Load the model
model = load_model(model_path)

# 4. Create a function "def predict_digit(model:Sequential, data_point:list) -> str" that will take the image serialized as an array of 784 elements and returns the predicted digit as string.
def predict_digit(model: Sequential, image_data: list) -> str:
    """
    Predict the digit in the given image data using the loaded model.
    
    Args:
        model (Sequential): The loaded Keras Sequential model.
        image_data (list): The input image serialized as a list of 784 elements.
        
    Returns:
        str: The predicted digit as a string.
    """
    # Reshape and normalize the image data
    image_data = np.array(image_data) / 255.0

    # Make prediction
    prediction = model.predict(image_data)
    digit = str(np.argmax(prediction))
    return digit

# 5. Create an API endpoint "@app.post('/predict')" that will read the bytes from the uploaded image to create a serialized array of 784 elements. The array shall be sent to 'predict_digit' function to get the digit. The API endpoint should return {"digit":digit"} back to the client.
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    API endpoint to predict the digit in an uploaded image.
    
    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: A dictionary containing the predicted digit.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Read image as grayscale
    image_data = image.reshape(1, 784)  # Serialize image as a list of 784 elements
    digit = predict_digit(model, image_data)
    return {"digit": digit}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)


########################### Unit tests ###########################
class TestFunctions(unittest.TestCase):
    @patch('keras.models.load_model')
    def test_load_model(self, mock_load_model):
        # Mock the behavior of the load_model function
        mock_model = MagicMock(spec=Sequential)
        mock_load_model.return_value = mock_model

        # Call the load_model function with a test path
        test_path = '/path/to/model.h5'
        loaded_model = load_model(test_path)

        # Check if the load_model function was called with the correct path
        mock_load_model.assert_called_with(test_path)
        self.assertEqual(loaded_model, mock_model)

    def test_predict_digit(self):
        # Create a mock model with a pre-defined prediction
        mock_model = MagicMock(spec=Sequential)
        mock_model.predict.return_value = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        # Create a test image data
        test_image_data = [0] * 784

        # Call the predict_digit function with the mock model and test image data
        predicted_digit = predict_digit(mock_model, test_image_data)

        # Check if the predicted digit is correct
        self.assertEqual(predicted_digit, '2')

if __name__ == '__main__':
    unittest.main()