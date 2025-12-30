import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Trader Performance & Sentiment Analysis", layout="wide")

st.title("Trader Performance and Sentiment Analysis")
st.write("This application analyzes historical trading data, daily sentiment, and uses machine learning models to explore their relationship.")

# --- 1. Data Loading ---
st.header("1. Data Loading and Initial Inspection")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

df_sentiment = load_data("historical_data.csv.gz")
st.success("Data loaded successfully from 'historical_data.csv'")

# Display first 5 rows
st.subheader("First 5 rows of the historical data:")
st.dataframe(df_sentiment.head())

# Display data types before conversion
st.subheader("Data types before 'Timestamp IST' conversion:")
buffer = io.StringIO() # Create a buffer to capture info() output
df_sentiment.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# --- 2. Data Preprocessing ---
st.header("2. Data Preprocessing")

# Convert 'Timestamp IST' column to datetime objects
# Using errors='coerce' to turn unparseable dates into NaT (Not a Time)
df_sentiment['Timestamp IST'] = pd.to_datetime(df_sentiment['Timestamp IST'], errors='coerce', format='%d-%m-%Y %H:%M')

# Display the data types again to confirm the conversion of 'Timestamp IST'
st.subheader("Data types after 'Timestamp IST' column conversion:")
buffer = io.StringIO() # Create a buffer to capture info() output
df_sentiment.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

# Rename df_sentiment to df_trader to align with the original notebook's flow
df_trader = df_sentiment.copy()

st.subheader("Descriptive statistics of all columns (including categorical):")
st.dataframe(df_trader.describe(include='all'))

st.subheader("Sum of missing values per column:")
st.dataframe(df_trader.isnull().sum().to_frame(name='Missing Values'))

import io # Import io for capturing info() output

# --- 3. Daily Aggregation and Sentiment Calculation ---
st.header("3. Daily Aggregation and Sentiment Calculation")

# 1. Extract the date component from the 'Timestamp IST' column and store it in a new column named 'Date'
df_trader['Date'] = df_trader['Timestamp IST'].dt.date

# Convert the 'Date' column to datetime objects to ensure proper grouping and indexing later
df_trader['Date'] = pd.to_datetime(df_trader['Date'])

# Group the df_trader DataFrame by this new 'Date' column and calculate metrics
df_daily_trader_performance = df_trader.groupby('Date').agg(
    Daily_PnL=('Closed PnL', 'sum'),
    Daily_Trade_Count=('Order ID', lambda x: x.nunique()),
    Daily_Volume_USD=('Size USD', 'sum'),
    Daily_Volume_Tokens=('Size Tokens', 'sum'),
    Daily_Avg_Execution_Price=('Execution Price', 'mean')
).reset_index()

st.subheader("Daily Trader Performance:")
st.dataframe(df_daily_trader_performance.head())
st.write(f"Shape of Daily Trader Performance DataFrame: {df_daily_trader_performance.shape}")

# Aggregate df_trader to create df_daily_sentiment
df_daily_sentiment = df_trader.groupby('Date').agg(
    Daily_Buy_Count=('Side', lambda x: (x == 'BUY').sum()),
    Daily_Sell_Count=('Side', lambda x: (x == 'SELL').sum())
).reset_index()

# Calculate Daily_Net_Buy_Ratio, handling potential division by zero by setting to 0 if no trades
def calculate_net_buy_ratio(row):
    total_trades = row['Daily_Buy_Count'] + row['Daily_Sell_Count']
    if total_trades == 0:
        return 0
    else:
        return (row['Daily_Buy_Count'] - row['Daily_Sell_Count']) / total_trades

df_daily_sentiment['Daily_Net_Buy_Ratio'] = df_daily_sentiment.apply(calculate_net_buy_ratio, axis=1)

st.subheader("Daily Sentiment Data (Net Buy Ratio):")
st.dataframe(df_daily_sentiment.head())
st.write(f"Shape of Daily Sentiment DataFrame: {df_daily_sentiment.shape}")

# Merge df_daily_trader_performance with df_daily_sentiment
df_merged = pd.merge(df_daily_trader_performance, df_daily_sentiment, on='Date', how='inner')
st.subheader("Merged Daily Performance and Sentiment Data (df_merged):")
st.dataframe(df_merged.head())
st.write(f"Shape of Merged DataFrame: {df_merged.shape}")

st.subheader("Data types of df_merged:")
buffer = io.StringIO()
df_merged.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Missing values in df_merged:")
st.dataframe(df_merged.isnull().sum().to_frame(name='Missing Values'))

# --- 4. Feature Engineering ---
st.header("4. Feature Engineering")

# Ensure the DataFrame is sorted by Date before creating time-series features
df_merged = df_merged.sort_values(by='Date').reset_index(drop=True)

# 1. Create lagged features for 'Daily_Net_Buy_Ratio'
df_merged['Lag_1_Daily_Net_Buy_Ratio'] = df_merged['Daily_Net_Buy_Ratio'].shift(1)
df_merged['Lag_3_Daily_Net_Buy_Ratio'] = df_merged['Daily_Net_Buy_Ratio'].shift(3)
df_merged['Lag_7_Daily_Net_Buy_Ratio'] = df_merged['Daily_Net_Buy_Ratio'].shift(7)

# 2. Calculate the daily change in 'Daily_Net_Buy_Ratio'
df_merged['Daily_Net_Buy_Ratio_Change'] = df_merged['Daily_Net_Buy_Ratio'].diff()

# 3. Calculate 3-day and 7-day rolling means (moving averages) for performance metrics
performance_cols = ['Daily_PnL', 'Daily_Volume_USD', 'Daily_Trade_Count']

for col in performance_cols:
    df_merged[f'MA_3_{col}'] = df_merged[col].rolling(window=3, min_periods=1).mean()
    df_merged[f'MA_7_{col}'] = df_merged[col].rolling(window=7, min_periods=1).mean()

st.subheader("DataFrame with New Features:")
st.dataframe(df_merged.head())
st.write(f"Shape of DataFrame after Feature Engineering: {df_merged.shape}")

st.subheader("Data types of df_merged after Feature Engineering:")
buffer = io.StringIO()
df_merged.info(buf=buffer)
s = buffer.getvalue()
st.text(s)

st.subheader("Missing values in df_merged after Feature Engineering:")
st.dataframe(df_merged.isnull().sum().to_frame(name='Missing Values'))

# --- 5. Prepare Data for Machine Learning ---
st.header("5. Prepare Data for Machine Learning")

# 1. Define the target variable y as 'Daily_PnL'
y = df_merged['Daily_PnL']

# 2. Define the feature set X
# Exclude 'Date', 'Daily_PnL' and other specified columns
columns_to_exclude = [
    'Date',
    'Daily_PnL',
    'Daily_Volume_Tokens',
    'Daily_Buy_Count',
    'Daily_Sell_Count',
    'Lag_3_Daily_Net_Buy_Ratio',
    'Lag_7_Daily_Net_Buy_Ratio',
    'Daily_Net_Buy_Ratio_Change',
    'MA_3_Daily_PnL',
    'MA_7_Daily_PnL',
    'MA_3_Daily_Volume_USD',
    'MA_7_Daily_Volume_USD',
    'MA_3_Daily_Trade_Count',
    'MA_7_Daily_Trade_Count'
]

X = df_merged.drop(columns=columns_to_exclude)

st.subheader("Initial Feature Set (X):")
st.write(f"Shape of X: {X.shape}")
st.dataframe(X.head())
st.subheader("Target Variable (y):")
st.write(f"Shape of y: {y.shape}")
st.dataframe(y.head())

# 3. Drop any rows from X and y that contain missing values (NaNs)
# First, align X and y by index to ensure consistent dropping
df_combined_ml = pd.concat([X, y], axis=1)
df_combined_ml.dropna(inplace=True)

X = df_combined_ml.drop(columns=['Daily_PnL'])
y = df_combined_ml['Daily_PnL']

st.subheader("Feature Set and Target Variable after dropping NaNs:")
st.write(f"Shape of X after dropping NaNs: {X.shape}")
st.write(f"Shape of y after dropping NaNs: {y.shape}")

# 4. Split the cleaned data into training and testing sets using a temporal split
split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

st.subheader("Data Split into Training and Testing Sets:")
st.write(f"Training set size: {len(X_train)} samples")
st.write(f"Testing set size: {len(X_test)} samples")
st.write(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
st.write(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

st.dataframe(X_train.head())
st.dataframe(X_test.head())

# --- 6. Train Machine Learning Models ---
st.header("6. Train Machine Learning Models")

# 1. Initialize StandardScaler and fit it to X_train to scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames with original column names for better readability/consistency
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

st.subheader("Scaled Training and Testing Features:")
st.write("X_train_scaled head:")
st.dataframe(X_train_scaled.head())
st.write("X_test_scaled head:")
st.dataframe(X_test_scaled.head())

# 2. Initialize the three regression models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42)
}

# 3. Train each model and measure training time
st.subheader("Model Training Progress:")
for name, model in models.items():
    st.write(f"Training {name}...")
    start_time = time.time() # time is already imported
    model.fit(X_train_scaled, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    st.success(f"{name} trained in {training_time:.4f} seconds.")

st.write("All models trained successfully.")

# --- 7. Evaluate Model Performance and Interpret Results ---
st.header("7. Evaluate Model Performance and Interpret Results")

# Create an empty dictionary to store evaluation results for display
model_results = {}

st.subheader("Model Evaluation Metrics:")
for name, model in models.items():
    st.write(f"Evaluating {name}...")
    y_pred = model.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    model_results[name] = {
        'R2': r2,
        'MAE': mae,
        'MSE': mse,
        'predictions': y_pred # Store predictions for later visualization
    }

    st.write(f"  **{name}**")
    st.write(f"    R2 Score: {r2:.4f}")
    st.write(f"    Mean Absolute Error (MAE): {mae:.4f}")
    st.write(f"    Mean Squared Error (MSE): {mse:.4f}")

# Display feature importances for tree-based models
st.subheader("Feature Importances (Tree-based Models):")
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        st.write(f"  **{name}**")
        importances = pd.Series(model.feature_importances_, index=X_train.columns)
        st.dataframe(importances.nlargest(5).to_frame(name='Importance')) # Display top 5 features

st.write("Model evaluation and interpretation complete.")

# --- 8. Visualize Key Findings and Insights ---
st.header("8. Visualize Key Findings and Insights")

st.subheader("Actual vs. Predicted Daily PnL for each model")

# Create a figure to display plots
fig1, axes1 = plt.subplots(1, len(models), figsize=(18, 5))

for i, (name, model_info) in enumerate(model_results.items()):
    y_pred = model_info['predictions']

    # Use the appropriate axis for the subplot
    ax = axes1[i] if len(models) > 1 else axes1

    # Create a scatter plot comparing actual vs. predicted values
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.6, label='Predictions', ax=ax)

    # Add a diagonal line for perfect predictions (y=x)
    max_val = max(y_test.max(), y_pred.max())
    min_val = min(y_test.min(), y_pred.min())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    # Label axes and set title
    ax.set_xlabel('Actual Daily PnL')
    ax.set_ylabel('Predicted Daily PnL')
    ax.set_title(f'Actual vs. Predicted Daily PnL - {name}')
    ax.legend()

plt.tight_layout()
st.pyplot(fig1)

st.subheader("Feature Importances for tree-based models")

# Create a figure to display feature importance plots
fig2, axes2 = plt.subplots(1, 2, figsize=(15, 6))
plot_index = 0
for name, model in models.items():
    if hasattr(model, 'feature_importances_'):
        # Extract feature importances
        importances = pd.Series(model.feature_importances_, index=X_train.columns)

        # Sort feature importances in descending order
        importances = importances.sort_values(ascending=False)

        # Create a bar plot
        sns.barplot(x=importances.index, y=importances.values, palette='viridis', hue=importances.index, legend=False, ax=axes2[plot_index])
        axes2[plot_index].set_title(f'Feature Importances - {name}')
        axes2[plot_index].set_xlabel('Feature')
        axes2[plot_index].set_ylabel('Importance Score')
        axes2[plot_index].tick_params(axis='x', rotation=45, ha='right') # Rotate labels for better readability
        plot_index += 1

plt.tight_layout()
st.pyplot(fig2)

st.write("Visualizations displayed.")
