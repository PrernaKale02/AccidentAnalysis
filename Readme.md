# Traffic Accident Severity Analysis & Prediction

This is a full-stack data mining project built for the "Data Warehousing and Mining" course. The application analyzes a dataset of traffic accidents to predict the severity of an injury, identify patterns in accident data, and present these findings in an interactive web interface.

## Project Overview

The core of this project is to take a large, raw dataset of traffic accident reports and apply data mining techniques to build a predictive model. This model is then served via a Python Flask API and consumed by a simple frontend, allowing a user to input hypothetical accident conditions and receive a real-time prediction of the likely outcome.

The main tasks performed are:

1.  **Data Cleaning & Preparation:** Handling thousands of rows with missing data.
2.  **Data Preprocessing:** Using one-hot encoding to convert categorical features (like weather and road conditions) into a format a machine can understand.
3.  **Data Transformation (Data Mining):** Addressing the severe class imbalance in the dataset (where "No Injury" far outnumbered "Fatal" accidents) using the **SMOTE (Synthetic Minority Over-sampling TEchnique)**.
4.  **Classification:** Training a Decision Tree model to predict the most severe injury.
5.  **Clustering:** Using K-Means to group accidents into distinct types based on their features.
6.  **Deployment:** Creating a web application (HTML/CSS/JS + Flask) that allows a user to interact with the trained models.

## Features

- **Severity Prediction:** Predicts the most likely injury severity (e.g., Fatal, Incapacitating, No Injury) based on user-selected inputs.
- **Prediction Probability:** Shows the model's "confidence" for each of the 5 injury classes in a bar chart.
- **Cluster Analysis:** Identifies which "type" of accident the user's input belongs to (e.g., "Afternoon Rush-Hour Accident") based on a K-Means clustering model.
- **Web Interface:** A clean, responsive frontend to interact with the models.

## Tech Stack

- **Backend:**
  - **Python 3**
  - **Flask:** For the web server and API.
  - **scikit-learn:** For all machine learning (Decision Tree, K-Means, StandardScaler, OneHotEncoder).
  - **pandas:** For data loading and manipulation.
  - **imbalanced-learn:** For using SMOTE.
- **Frontend:**
  - **HTML5**
  - **CSS3**
  - **JavaScript (ES6+):** For API calls (`fetch`) and DOM manipulation.
  - **Chart.js:** For the dynamic probability chart.
- **Dataset:**
  - [Traffic Accidents Dataset on Kaggle](https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents/data)

## How to Run This Project

### 1. Backend Setup

1.  **Navigate to the `backend` folder:**

    ```bash
    cd backend
    ```

2.  **Create and activate a Python virtual environment:**

    ```bash
    # Create the environment
    python -m venv venv

    # Activate on Windows
    .\venv\Scripts\activate

    # Activate on Mac/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
   use either the requirements.txt or install manually

   ```bash
        pip install -r requirements.txt
   ```

  ```bash
        pip install flask flask-cors pandas scikit-learn imbalanced-learn
  ```

5.  **Place the Dataset:**

    - Download `traffic_accidents.csv` from the Kaggle link above.
    - Create a folder `backend/data/` and place the CSV file inside it.

6.  **Train the Models:**

    - This is a one-time step. Run the training script to clean the data, apply SMOTE, and save the model files (`.pkl`).

    ```bash
    cd backend
    python train.py
    ```

7.  **Run the Flask API Server:**
    - Once the models are trained, start the server.
    ```bash
    python app.py
    ```
    - The API will now be running at `http://127.0.0.1:5000/`. Leave this terminal running.

### 2. Frontend Setup

1.  **Open the Frontend:**

    - In a _new_ terminal, navigate to the `frontend` folder.
    - You don't need a server. Simply open the `index.html` file in your web browser.

    ```bash
    # On Windows
    start index.html

    # On Mac
    open index.html
    ```

    OR use the live server option on vscode.

2.  **Use the Application:**
    - The web page will load. Fill out the form and click "Analyze Accident" to get a live prediction from your running Flask backend.

## Data Mining Insights

The most significant challenge in this project was the **highly imbalanced dataset**. The initial dataset had over 150,000 "No Injury" reports but only ~350 "Fatal" reports.

When a model was trained on this raw data, it achieved a high "fake" accuracy of 74% by simply learning to _always_ predict "No Injury." It had 0% recall for all other injury types.

The solution was to apply **SMOTE**, which generated synthetic data for the minority classes (Fatal, Incapacitating, etc.) until all 5 classes were equally represented. This forced the model to learn the actual patterns for severe accidents, resulting in a more honest (though lower) accuracy and, most importantly, the ability to successfully predict all 5 outcomes.

## Project Structure

```
AccidentAnalysis/
├── backend/
│   ├── data/
│   │   └── traffic_accidents.csv   (Must be added manually)
│   ├── venv/                     (Virtual environment)
│   │
│   ├── accident_severity_model.pkl (Generated by train.py)
│   ├── cluster_analysis.json     (Generated by train.py)
│   ├── encoder.pkl               (Generated by train.py)
│   ├── kmeans_model.pkl          (Generated by train.py)
│   ├── model_columns.pkl         (Generated by train.py)
│   ├── scaler.pkl                (Generated by train.py)
│   │
│   ├── app.py                    (The Flask API server)
│   └── train.py                  (The main data mining script)
│
└── frontend/
    ├── index.html                (The main webpage)
    ├── style.css                 (All styles)
    └── script.js                 (All JavaScript logic)
```
