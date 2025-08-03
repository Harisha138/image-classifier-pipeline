# End-to-End Image Classification Pipeline

This project demonstrates a simple, end-to-end machine learning pipeline for binary image classification. It includes scripts for training a PyTorch model, deploying it as a serverless API on [Modal](https://modal.com/), and interacting with it through a [Streamlit](https://streamlit.io/) web interface.

## 🎥 Short video

![Demo](short_video.gif)

## Project Overview
The pipeline consists of three main components:

1.  **Training (`ml_pipeline_script.py`):** A Python script to train a simple Convolutional Neural Network (CNN) on a local dataset (e.g., Cats vs. Dogs) and save the model weights.
2.  **Deployment (`modal_app.py`):** A Modal application that deploys the trained model as a serverless API endpoint. It uses FastAPI to handle web requests.
3.  **User Interface (`streamlit_ui.py`):** A Streamlit web app that allows users to upload an image, which is then sent to the deployed Modal API for inference. The predicted class is displayed to the user.


## Step-by-Step Instructions

### Step 1: Project Setup

1.  **Clone or Create Project:** Create a new folder for your project.

2.  **Create a Virtual Environment:** Open a terminal in your project folder and create a Python virtual environment. This isolates your project's dependencies.
    ```bash
    python -m venv venv
    ```

3.  **Activate the Environment:**
    * **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Requirements:** Install all the necessary Python packages using the provided `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

### Step 2: Prepare the Dataset

1.  **Download Data:** For this project, we recommend the [Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data) dataset from Kaggle. Download the `train.zip` file.

2.  **Create a Small Subset (for speed):**
    * Create a main folder named `cats_vs_dogs`.
    * Inside it, create two subfolders: `cats` and `dogs`.
    * Unzip `train.zip` and copy a small number of images (e.g., 500 of each) into the corresponding folders. This will make training much faster.

3.  **Update Script Path:** Make sure the `dataset_path` variable in `ml_pipeline_script.py` points to your `cats_vs_dogs` folder.

### Step 3: Train the Model

1.  Run the training script from your terminal. This will create the `model.pth` file in your project directory.
    ```bash
    python ml_pipeline_script.py
    ```

### Step 4: Deploy the Model to the Cloud

1.  **Set up Modal:** If you don't have a Modal account, sign up for free and get your API token.
    ```bash
    modal token new
    ```

2.  **Deploy the API:** With `model.pth` in your project folder, run the deployment command.
    ```bash
    modal deploy modal_app.py
    ```

3.  **Copy the API URL:** After a successful deployment, Modal will provide a public URL for your API endpoint. It will look something like `https://your-org--image-classifier-app-fastapi-app.modal.run`. Copy this URL.

### Step 5: Run the User Interface

1.  **Update the UI Script:** Open `streamlit_ui.py` and paste your Modal URL into the `MODAL_API_URL` variable. **Remember to add `/predict` to the end of the URL.**

2.  **Launch the App:** Run the Streamlit app from your terminal.
    ```bash
    streamlit run streamlit_ui.py
    ```

Your browser will open a new tab with the image classifier. You can now upload an image and get a prediction from your cloud-hosted model!


## 📸 Screenshots

A walkthrough of the project pipeline from training to prediction.

### 1. Training the Model in Colab
![Training the model in Google Colab](screenshots/Screenshot%20(197).png)

### 2. Deploying the Model to Modal
![Deploying the model to the cloud using Modal](screenshots/Screenshot%20(185).png)
![Screenshot](screenshots/Screenshot%202025-08-01%20234636.png)

### 3. Running the Streamlit UI
![Launching the Streamlit user interface](screenshots/Screenshot%20(191).png)

### 4. Predicting an Image
![Uploading a cat image and receiving a successful prediction](screenshots/Screenshot%20(196).png)






