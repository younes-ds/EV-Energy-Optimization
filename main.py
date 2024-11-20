import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


def load_and_process_data(file_path):
    """Load the dataset and preprocess it."""
    # Load the dataset
    data = pd.read_csv(file_path)

    # Log transformation of 'Trip Energy Consumption' to handle skewness and outliers
    data['Log_Trip_Energy_Consumption'] = np.log1p(data['Trip Energy Consumption'])

    # Handle missing values in 'Trip Distance' by replacing zeros with NaN, then dropping rows with NaN
    data['Trip Distance'].replace(0, np.nan, inplace=True)
    data.dropna(subset=['Trip Distance'], inplace=True)

    # Create a new feature for efficiency (energy consumption per distance)
    data['Efficiency'] = data['Log_Trip_Energy_Consumption'] / data['Trip Distance']

    # Convert 'Temperature_Category' to numerical values based on temperature ranges
    data['Temperature_Category'] = pd.cut(data['Maximum Cell Temperature of Battery'], 
                                          bins=[-np.inf, 10, 25, np.inf], 
                                          labels=['Cold', 'Moderate', 'Hot'])
    label_encoder = LabelEncoder()
    data['Temperature_Category'] = label_encoder.fit_transform(data['Temperature_Category'])

    return data


def scale_data(X_train, X_test):
    """Scale the features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test):
    """Train and evaluate the model, returning performance metrics."""
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict and evaluate
    train_predictions = model.predict(X_train_scaled)
    test_predictions = model.predict(X_test_scaled)

    # Calculate RMSE for the predictions
    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    return train_rmse, test_rmse


def plot_feature_importance(model, features):
    """Plot the feature importance of the model."""
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 8))
    plt.bar(features, feature_importances)
    plt.title("Feature Importance from XGBoost Model")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.xticks(rotation=90)
    plt.tight_layout()  # Adjust layout to prevent clipping of feature names
    plt.show()


def plot_residuals(y_test, predictions):
    """Plot the residuals of the model."""
    sns.residplot(x=y_test, y=predictions, line_kws={'color': 'red'})
    plt.title('Residuals for XGBoost')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()


def plot_scatter(x, y, x_label, y_label, title):
    """Plot a scatter plot for given data."""
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, alpha=0.5)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.show()


def main():
    # Load and process the data
    data = load_and_process_data('Electric Vehicle Trip Energy Consumption Data.csv')

    # Define features (X) and target (y)
    X = data[['Trip Distance', 'Maximum Cell Temperature of Battery', 'Efficiency', 'Temperature_Category']]
    y = data['Log_Trip_Energy_Consumption']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    # Linear Regression model
    lin_reg = LinearRegression()
    train_rmse, test_rmse = evaluate_model(lin_reg, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"Linear Regression RMSE (Train): {train_rmse}")
    print(f"Linear Regression RMSE (Test): {test_rmse}")

    # XGBoost model
    xgb_model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    train_rmse_xgb, test_rmse_xgb = evaluate_model(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test)
    print(f"XGBoost RMSE (Train): {train_rmse_xgb}")
    print(f"XGBoost RMSE (Test): {test_rmse_xgb}")

    # Plot residuals for XGBoost
    plot_residuals(y_test, xgb_model.predict(X_test_scaled))

    # Plot feature importance from XGBoost
    plot_feature_importance(xgb_model, X.columns)

    # Plot scatter plot for speed vs energy consumption (example)
    plot_scatter(data['Speed'], data['Trip Energy Consumption'], 'Speed (km/h)', 'Energy Consumption (kWh)', "Speed vs Energy Consumption")


if __name__ == "__main__":
    main()
