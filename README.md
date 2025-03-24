
---

# Diabetes Prediction App

This is a Streamlit-based web application that predicts whether a person has diabetes based on input features such as pregnancies, glucose levels, blood pressure, and more. The app uses a pre-trained RandomForestClassifier model and provides visualizations to compare user data with a dataset of other patients.

## Features
- **User Input**: Enter patient data via sliders in the sidebar (e.g., Pregnancies, Glucose, Blood Pressure, etc.).
- **Prediction**: Predicts whether the user is diabetic using a pre-trained machine learning model.
- **Visualizations**: Displays scatter plots comparing the user's data with a dataset of other patients, color-coded by diabetes outcome.
- **Dataset**: Uses the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) for visualization.

## Project Structure
The repository contains the following files:
- `streamlit_interface.py`: The main Streamlit app script for the web interface.
- `diabetes.csv`: The dataset used for visualization (Pima Indians Diabetes Dataset).
- `diabetes_model.pkl`: The pre-trained RandomForestClassifier model.
- `requirements.txt`: Lists the Python dependencies required to run the app.

## Prerequisites
- Python 3.7 or higher
- Git (for cloning the repository)
- A GitHub account (for deployment on Streamlit Community Cloud)

## Setup Instructions

### 1. Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/hemasriram111/diabetes-prediction.git
cd diabetes-prediction-app
```

### 2. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

### 3. Run the App Locally
Run the Streamlit app on your local machine:
```bash
streamlit run streamlit_interface.py
```
- This will open the app in your default web browser (e.g., `http://localhost:8501`).

## Deployment on Streamlit Community Cloud
You can deploy this app on [Streamlit Community Cloud](https://streamlit.io/

## How to Use the App
1. Open the app in your browser (either locally or via the deployed URL).
2. Use the sliders in the sidebar to input patient data:
   - Pregnancies (0-17)
   - Glucose (0-200)
   - Blood Pressure (0-122)
   - Skin Thickness (0-100)
   - Insulin (0-846)
   - BMI (0-67)
   - Diabetes Pedigree Function (0.0-2.4)
   - Age (21-88)
3. View the patient data summary and prediction result ("You are not Diabetic" or "You are Diabetic").
4. Explore the visualizations comparing your data with the dataset, with points color-coded by outcome (blue for non-diabetic, red for diabetic).



## Model Training
The model was trained using a RandomForestClassifier on the Pima Indians Diabetes Dataset. The training script (`app.py`) is not included in the deployment but was used to generate `diabetes_model.pkl`. To retrain the model:
1. Use the `app.py` script (or similar) to load the dataset, train the model, and save it as `diabetes_model.pkl`.
2. Replace the existing `diabetes_model.pkl` in the repository with the new one.

## Dependencies
The app relies on the following Python libraries (listed in `requirements.txt`):
- `streamlit`: For the web interface
- `pandas`: For data manipulation
- `numpy`: For numerical operations
- `matplotlib`: For plotting
- `seaborn`: For enhanced visualizations
- `scikit-learn`: For the machine learning model

## Limitations
- The app uses a pre-trained model, so it cannot be retrained via the web interface.
- The dataset (`diabetes.csv`) is static and used only for visualization purposes.
- The model’s accuracy depends on the quality of the training data and may not generalize to all populations.

## Contributing
Contributions are welcome! If you’d like to improve the app, fix bugs, or add features:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Make your changes and commit (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments
- The dataset used in this project is the Pima Indians Diabetes Dataset, available on [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).
- Built with [Streamlit](https://streamlit.io/), a fantastic tool for creating data apps.

-
