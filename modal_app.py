import os
import modal
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import numpy as np
from io import BytesIO

# --- 1. Modal Setup ---

# Define the Modal App
app = modal.App("image-classifier-app")

# Define the Modal Image with necessary packages and add the model file
modal_image = (
    modal.Image.debian_slim()
    .pip_install("torch", "fastapi", "uvicorn", "Pillow", "numpy", "python-multipart")
    .add_local_file("model.pth", remote_path="/model.pth")
)

# --- 2. Model Definition (Must be the same as in the training script) ---

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 16 * 16)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Create a FastAPI Web Endpoint ---

web_app = FastAPI()

# --- 4. Link the FastAPI app to Modal ---

@app.function(image=modal_image) # The mounts argument is no longer needed here
@modal.asgi_app()
def fastapi_app():
    # --- Model Loading and Prediction Logic (Moved Inside) ---
    # This code now runs ONLY inside the Modal container when the app starts.
    
    # Load the model from the mounted path
    model_path = "/model.pth"
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    class_names = ['Cat', 'Dog']

    @web_app.post("/predict")
    async def predict(file: UploadFile = File(...)):
        contents = await file.read()
        img = Image.open(BytesIO(contents)).resize((64, 64)).convert('RGB')
        
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted_idx = torch.max(outputs.data, 1)
            prediction = class_names[predicted_idx.item()]
            
        return JSONResponse(content={"prediction": prediction})

    return web_app
