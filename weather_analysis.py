# weather_analysis.py
from io import StringIO

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import os


def clean_data(df):
    """
    Handles missing values in the dataset
    """
    print("\nDEBUG: Cleaning data...")
    print(f"Original shape: {df.shape}")
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Set the index to datetime for time-based interpolation
    print("\nDEBUG: Setting datetime index...")
    df = df.set_index('day')

    # Fill missing weather data with appropriate methods
    print("DEBUG: Interpolating weather data...")
    weather_cols = ['temperatureMax', 'humidity', 'windSpeed']
    for col in weather_cols:
        print(f"DEBUG: Interpolating {col}...")
        df[col] = df[col].interpolate(method='time')
        # Forward fill any remaining NaN at the edges
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')

    # Reset index to get 'day' back as a column
    df = df.reset_index()

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())
    print(f"Final shape: {df.shape}")

    # Drop the 'time' column if it exists as we no longer need it
    if 'time' in df.columns:
        df = df.drop('time', axis=1)
        print("DEBUG: Dropped 'time' column")

    return df


def plot_correlation_matrix(df, cols_to_plot):
    """
    Creates and saves correlation matrix heatmap
    """
    print("DEBUG: Creating correlation matrix...")
    plt.figure(figsize=(10, 8))
    correlation = df[cols_to_plot].corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig('correlation_matrix.png')
    plt.close()
    return correlation


def analyze_weather_relationships(df):
    """
    Analyzes relationships between weather conditions and energy consumption
    """
    print("\nDEBUG: Starting weather relationship analysis...")

    # Create output directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
        print("DEBUG: Created plots directory")

    # Save data description
    print("DEBUG: Saving data description...")
    with open('plots/data_description.txt', 'w') as f:
        f.write("Data Description:\n\n")
        f.write(df.describe().to_string())
        f.write("\n\nData Info:\n")
        buffer = StringIO()
        df.info(buf=buffer)
        f.write(buffer.getvalue())

    # 1. Temperature vs Energy with Secondary Y-Axis
    print("DEBUG: Creating temperature vs energy plot with secondary y-axis...")
    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Temperature', color=color)
    ax1.plot(df.day, df.temperatureMax, color=color, label='Temperature')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # Create a secondary y-axis
    color = 'tab:blue'
    ax2.set_ylabel('Energy Consumption', color=color)
    ax2.plot(df.day, df.energy_mean, color=color, label='Energy Consumption')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Energy Consumption and Temperature Over Time')
    fig.tight_layout()
    plt.savefig('plots/temp_energy_time.png')
    plt.close()

    # 2. Humidity vs Energy with Secondary Y-Axis
    print("DEBUG: Creating humidity vs energy plot with secondary y-axis...")
    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = 'tab:green'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Humidity', color=color)
    ax1.plot(df.day, df.humidity, color=color, label='Humidity')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Energy Consumption', color=color)
    ax2.plot(df.day, df.energy_mean, color=color, label='Energy Consumption')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Energy Consumption and Humidity Over Time')
    fig.tight_layout()
    plt.savefig('plots/humidity_energy_time.png')
    plt.close()

    # 3. Wind Speed vs Energy with Secondary Y-Axis
    print("DEBUG: Creating wind speed vs energy plot with secondary y-axis...")
    fig, ax1 = plt.subplots(figsize=(15, 6))

    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Wind Speed', color=color)
    ax1.plot(df.day, df.windSpeed, color=color, label='Wind Speed')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Energy Consumption', color=color)
    ax2.plot(df.day, df.energy_mean, color=color, label='Energy Consumption')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Energy Consumption and Wind Speed Over Time')
    fig.tight_layout()
    plt.savefig('plots/wind_energy_time.png')
    plt.close()

    # 4. Scatter plots with regression lines
    print("DEBUG: Creating scatter plots with regression lines...")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.regplot(ax=axes[0], x=df.temperatureMax, y=df.energy_mean, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[0].set_title('Energy vs Temperature')
    axes[0].set_xlabel('Temperature')
    axes[0].set_ylabel('Energy Consumption')

    sns.regplot(ax=axes[1], x=df.humidity, y=df.energy_mean, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[1].set_title('Energy vs Humidity')
    axes[1].set_xlabel('Humidity')
    axes[1].set_ylabel('Energy Consumption')

    sns.regplot(ax=axes[2], x=df.windSpeed, y=df.energy_mean, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    axes[2].set_title('Energy vs Wind Speed')
    axes[2].set_xlabel('Wind Speed')
    axes[2].set_ylabel('Energy Consumption')

    plt.tight_layout()
    plt.savefig('plots/weather_scatter_plots.png')
    plt.close()

    # Create correlation matrix
    cols_to_plot = ['energy_mean', 'temperatureMax', 'humidity', 'windSpeed']
    correlation = plot_correlation_matrix(df, cols_to_plot)

    return correlation



def create_weather_clusters(df, n_clusters=3):
    """
    Creates weather clusters based on temperature, humidity, and wind speed
    """
    print("\nDEBUG: Starting weather clustering...")

    # Select features
    weather_features = ['temperatureMax', 'humidity', 'windSpeed']

    # Check for NaN values
    print("\nDEBUG: Checking for NaN values in clustering features:")
    print(df[weather_features].isnull().sum())

    # Scale features
    scaler = MinMaxScaler()
    weather_scaled = scaler.fit_transform(df[weather_features])

    print(f"\nDEBUG: Scaled features shape: {weather_scaled.shape}")

    # Find optimal number of clusters
    print("DEBUG: Finding optimal number of clusters...")
    inertias = []
    K = range(1, 10)
    for k in K:
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(weather_scaled)
            inertias.append(kmeans.inertia_)
        except Exception as e:
            print(f"ERROR with k={k}: {str(e)}")

    # Plot elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(K, inertias, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.savefig('plots/elbow_curve.png')
    plt.close()

    # Create clusters
    print(f"DEBUG: Creating {n_clusters} weather clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['weather_cluster'] = kmeans.fit_predict(weather_scaled)

    # Visualize clusters
    print("DEBUG: Creating cluster visualizations...")
    fig = plt.figure(figsize=(20, 5))

    plt.subplot(131)
    plt.scatter(df.weather_cluster, df.temperatureMax, c=df.weather_cluster)
    plt.title('Temperature by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Temperature')

    plt.subplot(132)
    plt.scatter(df.weather_cluster, df.humidity, c=df.weather_cluster)
    plt.title('Humidity by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Humidity')

    plt.subplot(133)
    plt.scatter(df.weather_cluster, df.windSpeed, c=df.weather_cluster)
    plt.title('Wind Speed by Cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Wind Speed')

    plt.tight_layout()
    plt.savefig('plots/cluster_analysis.png')
    plt.close()

    return df

def validate_data(df):
    """
    Validates the required columns and data types
    """
    print("\nDEBUG: Validating data...")

    required_columns = [
        'day', 'energy_mean', 'temperatureMax',
        'humidity', 'windSpeed'
    ]

    # Check columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check data types
    if not pd.api.types.is_datetime64_any_dtype(df['day']):
        print("WARNING: Converting 'day' to datetime")
        df['day'] = pd.to_datetime(df['day'])

    # Check value ranges
    print("\nDEBUG: Value ranges:")
    for col in ['energy_mean', 'temperatureMax', 'humidity', 'windSpeed']:
        if col in df.columns:  # Only check if column exists
            print(f"{col}: {df[col].min():.2f} to {df[col].max():.2f}")

    return df


def main():
    print("DEBUG: Starting weather analysis...")

    try:
        # Load the processed data
        print("DEBUG: Loading processed data...")
        df = pd.read_csv('processed_data.csv', parse_dates=['day'])
        print(f"DEBUG: Loaded data shape: {df.shape}")
        print("\nDEBUG: Columns in data:")
        print(df.columns.tolist())

        # Validate data
        df = validate_data(df)

        # Clean data
        df = clean_data(df)

        # Analyze weather relationships
        print("\nDEBUG: Analyzing weather relationships...")
        correlations = analyze_weather_relationships(df)
        print("\nCorrelations with energy consumption:")
        print(correlations['energy_mean'])

        # Create weather clusters
        print("\nDEBUG: Creating weather clusters...")
        df = create_weather_clusters(df)

        # Save enhanced dataset
        print("\nDEBUG: Saving enhanced dataset...")
        df.to_csv('enhanced_data.csv', index=False)
        print("DEBUG: Enhanced dataset saved successfully")

        print("\nDEBUG: Analysis complete! Check the 'plots' directory for visualizations.")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("Please ensure processed_data.csv exists and contains the required columns")

    finally:
        plt.close('all')  # Close any open plots

if __name__ == "__main__":
    from io import StringIO
    main()
