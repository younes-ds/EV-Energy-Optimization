# Electric Vehicle Trip Energy Consumption Analysis

This project focuses on analyzing and predicting electric vehicle trip energy consumption using machine learning models. The workflow includes data preprocessing, feature engineering, model training, and evaluation with models such as Linear Regression and XGBoost. This project can be expanded to optimize EV charging schedules and infrastructure efficiency by leveraging additional data, such as charging power, state of charge, station locations... Integrating these insights would enable peak time scheduling, cost-effective energy use, and improved charging station availability.

## **Goal**
Predict energy consumption for electric vehicle trips based on factors like trip distance, battery temperature, and efficiency.

---

## **Steps**
1. **Data Loading and Preprocessing**:
   - Handle missing values.
   - Apply log transformation to the target variable (`Trip Energy Consumption`) for better modeling.
   - Engineer features such as efficiency (energy per distance).
   - Categorize battery temperature into "Cold," "Moderate," or "Hot."

2. **Feature Scaling**:
   - Scale numerical features using `StandardScaler` for better model performance.

3. **Model Training and Evaluation**:
   - Train models using Linear Regression and XGBoost.
   - Evaluate models with Root Mean Squared Error (RMSE) for training and testing sets.

4. **Visualizations**:
   - Residual analysis for model performance.
   - Feature importance plot from the XGBoost model.
   - Scatter plot (e.g., speed vs energy consumption).

---

## **Features**
- `Trip Distance`: Distance of the trip (km).
- `Maximum Cell Temperature of Battery`: Maximum battery temperature during the trip (°C).
- `Efficiency`: Energy consumed per unit distance.
- `Temperature_Category`: Categorized temperature (`Cold`, `Moderate`, `Hot`).

---

## **Evaluation Metrics**
- **RMSE**: Measures prediction error for training and test data.
- **Residual Plot**: Visualizes the difference between true and predicted values.
- **Feature Importance**: Highlights the most influential features in the XGBoost model.

---

## **Requirements**
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

### **Installation**
Install dependencies with the following command:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
