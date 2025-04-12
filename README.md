# ML Model Comparator with Hyperparameter Tuning

## Project Overview
The **ML Model Comparator** is a Streamlit-based web application designed to simplify the process of comparing machine learning models. It allows users to upload a dataset, select a target column, and evaluate multiple classification models. The application performs hyperparameter tuning to optimize model performance and provides visualizations for easy comparison. This tool is particularly useful for data scientists and machine learning practitioners who want to quickly identify the best-performing model for their dataset.

## Key Features & Technologies

### Key Features
- **Dataset Upload**: Upload CSV files for analysis.
- **Automatic Data Cleaning**: Automatically removes columns with missing values.
- **Feature Scaling**: Standardizes features using `StandardScaler`.
- **Model Comparison**: Evaluates multiple classification models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine (SVM)
- **Hyperparameter Tuning**: Uses `GridSearchCV` to optimize model parameters.
- **Performance Metrics**: Calculates and displays:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- **Visualizations**:
  - Bar plots for metric comparison.
  - ROC AUC curves for tuned models.
- **Downloadable Results**: Export model comparison results as a CSV file.

### Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/) for interactive UI.
- **Backend**: Python for data processing and machine learning.
- **Libraries**:
  - `pandas` and `numpy` for data manipulation.
  - `scikit-learn` for machine learning models, preprocessing, and evaluation.
  - `matplotlib` and `seaborn` for data visualization.

### Setup Instructions
- Download the project file
- Install **important library files** in Command Prompt :
`pip install streamlit pandas numpy matplotlib seaborn scikit-learn`
- In CMD **change destination** to the downloaded folder (For example) :
`cd C:\Users\sujit\Desktop\25_April_Hackathon`
- Then using the streamlit command to run the Program :
`streamlit run ML_Model_HPT.py`
- In the program (Browser Mode) drag and drop the **Recommended Dataset** (included in the project folder also drive link provided in the streamlit program)
- See all the analytics of the machine learning models before and after Hyper-parameter tuning
- Download the csv file of the **model-comparisons** by the download button at the very last

## Why It Matters
Choosing the right machine learning model and tuning its hyperparameters can be a time-consuming process. This project automates these tasks, enabling users to focus on insights and decision-making. By providing a user-friendly interface and robust evaluation metrics, the ML Model Comparator ensures that users can make informed choices about their models with minimal effort.
