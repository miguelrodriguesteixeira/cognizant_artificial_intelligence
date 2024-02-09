import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Specify the path to your CSV files
path_sales = "/Users/miguelteixeira/cognizant/Cognizant - Data Analysis/sample_sales_data.csv"
path_sensor_temperature = "/Users/miguelteixeira/cognizant/sensor_storage_temperature.csv"
path_sensor_stock_levels = "/Users/miguelteixeira/cognizant/sensor_stock_levels.csv"

# Read the CSV files into DataFrames
df_sales = pd.read_csv(path_sales)
df_sensor_temperature = pd.read_csv(path_sensor_temperature)
df_sensor_stock_levels = pd.read_csv(path_sensor_stock_levels)

# Convert timestamp to hourly format
def convert_timestamp_to_hourly(data: pd.DataFrame = None, column: str = None):
    dummy = data.copy()
    dummy[column] = pd.to_datetime(dummy[column])  # Convert to datetime
    new_ts = dummy[column].tolist()
    new_ts = [i.strftime('%Y-%m-%d %H:00:00') for i in new_ts]
    new_ts = [datetime.strptime(i, '%Y-%m-%d %H:00:00') for i in new_ts]
    dummy[column] = new_ts
    return dummy

df_sales = convert_timestamp_to_hourly(df_sales, 'timestamp')
df_sensor_temperature = convert_timestamp_to_hourly(df_sensor_temperature, 'timestamp')
df_sensor_stock_levels = convert_timestamp_to_hourly(df_sensor_stock_levels, 'timestamp')

# Merge datasets based on 'timestamp'
df_merged = pd.merge(df_sales, df_sensor_temperature, on='timestamp', how='left')
df_merged = pd.merge(df_merged, df_sensor_stock_levels, on='timestamp', how='left')

# Drop irrelevant columns for modeling
df_merged.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')

# Choose features (X) and target variable (y)
X = df_merged[['feature1', 'feature2', ...]]  # Replace with actual feature names
y = df_merged['target_variable']  # Replace with actual target variable name

# Instantiate algorithm and scaler
model = RandomForestRegressor()
scaler = StandardScaler()

# Number of folds for cross-validation
K = 10
split = 0.75
accuracy = []

# Cross-validation loop
for fold in range(0, K):
    # Create training and test samples
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

    # Scale X data
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    trained_model = model.fit(X_train, y_train)

    # Generate predictions on the test sample
    y_pred = trained_model.predict(X_test)

    # Compute accuracy, using mean absolute error
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    accuracy.append(mae)
    print(f"Fold {fold + 1}: MAE = {mae:.3f}")

# Display average MAE
print(f"Average MAE: {(sum(accuracy) / len(accuracy)):.2f}")
