# 🚢 Titanic Survival Prediction – Machine Learning & Web Application

This project predicts the survival of passengers on the **Titanic** using machine learning techniques. The model uses features such as **age**, **sex**, **class**, and other variables to predict the likelihood of survival for each passenger. Additionally, a simple **HTML web page** is created to interact with the model and allow users to input their data to get predictions.

---

## 🎯 Objective

- Build a machine learning model to predict Titanic passenger survival
- Create a web interface to allow users to input data and get survival predictions
- Provide insights into the factors affecting survival probability

---

## 📂 Dataset

The dataset used for this project is the **Titanic dataset** available from **Kaggle**:

- **Source**: [Kaggle Titanic dataset](https://www.kaggle.com/c/titanic/data)
- **Features**:
  - `Pclass`: Passenger class (1st, 2nd, 3rd)
  - `Name`: Passenger's name
  - `Sex`: Gender of the passenger
  - `Age`: Age of the passenger
  - `SibSp`: Number of siblings/spouses aboard
  - `Parch`: Number of parents/children aboard
  - `Fare`: Passenger fare
  - `Embarked`: Port of embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
  - `Survived`: Target variable indicating if the passenger survived (1 = Yes, 0 = No)

> 📌 Note: Ensure the dataset is placed in a `data/` directory for easy access during model training.

---

## 🚀 Project Workflow

1. **Data Preprocessing**
   - Handle missing values (e.g., fill or drop missing values for `Age`, `Embarked`)
   - Convert categorical variables (`Sex`, `Embarked`) into numerical formats (e.g., one-hot encoding)
   - Feature engineering: Create new features like family size or age group

2. **Model Training**
   - Split data into training and test sets
   - Experiment with machine learning models:
     - Logistic Regression
     - Random Forest
     - Support Vector Machine (SVM)
     - XGBoost
   - Evaluate the model using metrics like **accuracy**, **precision**, **recall**, and **F1-score**

3. **Model Deployment (Web Interface)**
   - **Web Page**: Create an HTML interface with an input form for the user to input Titanic passenger details.
   - **Prediction**: Allow the model to predict survival based on user input and display the result on the web page.

4. **Model Evaluation**
   - Evaluate the model on the test set and check its accuracy.
   - Fine-tune model parameters using cross-validation and grid search.

---

## 🛠️ Technologies Used

| Technology      | Purpose                                     |
|-----------------|---------------------------------------------|
| pandas          | Data preprocessing and manipulation        |
| numpy           | Numerical operations                        |
| scikit-learn    | Machine learning model training and evaluation |
| flask           | Web framework for model deployment         |
| jinja2          | Templating engine for rendering HTML pages  |
| html, css       | Front-end web page development             |
| Bootstrap       | Styling for web page                       |

---

## 📁 Project Structure

titanic-survival-prediction/
├── data/
│ ├── train.csv # Titanic dataset
├── app/
│ ├── templates/
│ │ └── index.html # HTML web page for user input and display
│ ├── app.py # Flask web application
│ ├── model.pkl # Saved model
├── notebooks/
│ └── titanic_survival_prediction.ipynb # Jupyter notebook for model training
├── requirements.txt
└── README.md # Project documentation

yaml
Copy
Edit

---

## 📊 Model Evaluation

To evaluate the performance of the model, we use several metrics:

- **Accuracy**: Percentage of correctly predicted outcomes
- **Precision**: The proportion of positive predictions that are actually correct
- **Recall**: The proportion of actual positives that are correctly identified by the model
- **F1-Score**: Harmonic mean of precision and recall

---

## 📄 Requirements

To run this project locally, install the necessary dependencies with:

bash
pip install -r requirements.txt
Typical libraries in requirements.txt:

txt
Copy
Edit
pandas
numpy
scikit-learn
flask
xgboost
matplotlib
jinja2

---

💡 Future Improvements

🛠 User Interface Enhancements: Improve the web interface by adding more form validation and styling.
🔮 Advanced Model: Try ensemble methods or neural networks to improve prediction accuracy.
🧑‍💻 Model API: Deploy the model as a REST API with Flask or FastAPI to allow integration with other applications.
🌍 Real-time Data: Collect real-time Titanic passenger data for predictions in future voyages.

---

## 👨‍💻 Author

Developed by Rakhi Yadav
Feel free to fork, contribute, or suggest improvements!

---
