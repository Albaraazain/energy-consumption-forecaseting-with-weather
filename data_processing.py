import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns

def check_directory_structure():
    """
    Debug function to check directory structure
    """
    print("\nDEBUG: Checking directory structure...")
    print(f"Current working directory: {os.getcwd()}")

    # Check main directories
    main_dirs = ['daily_dataset', 'halfhourly_dataset', 'hhblock_dataset']
    for dir_name in main_dirs:
        if os.path.exists(dir_name):
            print(f"\nDEBUG: Found {dir_name} directory")
            print(f"Contents of {dir_name}/daily_dataset:")
            nested_path = os.path.join(dir_name, dir_name)
            if os.path.exists(nested_path):
                try:
                    files = os.listdir(nested_path)
                    print(f"Number of files: {len(files)}")
                    print("Sample files:", files[:5] if files else "No files found")
                except Exception as e:
                    print(f"Error reading directory: {e}")
        else:
            print(f"\nDEBUG: {dir_name} directory not found")

def load_energy_data(block_range=range(112)):
    """
    Loads and combines block data files
    """
    print("\nDEBUG: Starting energy data loading process...")
    print(f"DEBUG: Will attempt to load {len(block_range)} blocks")

    # Correct path for your structure
    base_path = os.path.join('daily_dataset', 'daily_dataset')
    print(f"DEBUG: Using base path: {base_path}")

    all_blocks = []
    for block in block_range:
        try:
            file_path = os.path.join(base_path, f'block_{block}.csv')
            print(f"DEBUG: Attempting to load: {file_path}")

            if os.path.exists(file_path):
                print(f"DEBUG: Found file: {file_path}")
                df = pd.read_csv(file_path)
                print(f"DEBUG: Successfully loaded block_{block}.csv with shape {df.shape}")
                all_blocks.append(df)
            else:
                print(f"WARNING: {file_path} not found")

        except Exception as e:
            print(f"ERROR loading block_{block}.csv: {str(e)}")
            continue

    if not all_blocks:
        print("ERROR: No data files were loaded.")
        return pd.DataFrame()

    combined_df = pd.concat(all_blocks, ignore_index=True)
    print(f"\nDEBUG: Combined DataFrame shape: {combined_df.shape}")
    print("DEBUG: Combined DataFrame columns:", combined_df.columns.tolist())
    return combined_df

def load_weather_holidays():
    """
    Loads weather and holiday data
    """
    print("\nDEBUG: Starting weather and holiday data loading...")

    # Weather data
    print("DEBUG: Loading weather data...")
    weather = pd.DataFrame()
    weather_path = 'weather_daily_darksky.csv'

    if os.path.exists(weather_path):
        print(f"DEBUG: Found weather file at {weather_path}")
        weather = pd.read_csv(weather_path)
        print(f"DEBUG: Weather data shape: {weather.shape}")
    else:
        print(f"ERROR: Weather file not found at {weather_path}")

    # Holiday data
    print("\nDEBUG: Loading holiday data...")
    holidays = pd.DataFrame()
    holiday_path = 'uk_bank_holidays.csv'

    if os.path.exists(holiday_path):
        print(f"DEBUG: Found holiday file at {holiday_path}")
        holidays = pd.read_csv(holiday_path)
        print(f"DEBUG: Holiday data shape: {holidays.shape}")
    else:
        print(f"ERROR: Holiday file not found at {holiday_path}")

    return weather, holidays

def preprocess_data(energy_df, weather_df, holidays_df):
    """
    Combines and preprocesses all data sources
    """
    print("\nDEBUG: Starting data preprocessing...")
    print(f"DEBUG: Input shapes - Energy: {energy_df.shape}, Weather: {weather_df.shape}, Holidays: {holidays_df.shape}")

    if energy_df.empty:
        print("WARNING: Energy DataFrame is empty. Exiting preprocessing.")
        return pd.DataFrame()

    try:
        # Convert dates
        print("DEBUG: Converting date columns...")
        energy_df['day'] = pd.to_datetime(energy_df['day'])
        weather_df['time'] = pd.to_datetime(weather_df['time'])
        holidays_df['Bank holidays'] = pd.to_datetime(holidays_df['Bank holidays'])

        # Calculate daily stats
        print("DEBUG: Calculating daily statistics...")
        daily_stats = energy_df.groupby(['day', 'LCLid']).agg({
            'energy_sum': 'sum',
            'energy_mean': 'mean'
        }).reset_index()
        print(f"DEBUG: Daily stats shape: {daily_stats.shape}")

        # Average across households
        print("DEBUG: Calculating household averages...")
        daily_avg = daily_stats.groupby('day').agg({
            'energy_sum': 'mean',
            'energy_mean': 'mean'
        }).reset_index()
        print(f"DEBUG: Daily averages shape: {daily_avg.shape}")

        # Merge with weather and holidays
        print("DEBUG: Merging datasets...")
        df = daily_avg.merge(
            weather_df[['time', 'temperatureMax', 'humidity', 'windSpeed']],
            left_on='day',
            right_on='time',
            how='left'
        )
        print(f"DEBUG: Shape after weather merge: {df.shape}")

        df['is_holiday'] = df['day'].isin(holidays_df['Bank holidays']).astype(int)
        print(f"DEBUG: Final shape: {df.shape}")

        # Save processed data
        print("\nDEBUG: Saving processed data to CSV...")
        df.to_csv('processed_data.csv', index=False)
        print("DEBUG: Data saved successfully to processed_data.csv")

        return df

    except Exception as e:
        print(f"ERROR during preprocessing: {str(e)}")
        return pd.DataFrame()

def main():
    print("DEBUG: Starting main function...")

    # Check directory structure
    check_directory_structure()

    # Load data
    print("\nStep 1: Load energy data")
    energy_data = load_energy_data()
    print(f"Energy data shape: {energy_data.shape}")

    print("\nStep 2: Load weather and holiday data")
    weather_data, holidays_data = load_weather_holidays()
    print(f"Weather data shape: {weather_data.shape}")
    print(f"Holidays data shape: {holidays_data.shape}")

    # Process data
    print("\nStep 3: Preprocess data")
    processed_data = preprocess_data(energy_data, weather_data, holidays_data)

    if not processed_data.empty:
        print("\nDEBUG: Processing successful!")
        print("\nSample of processed data:")
        print(processed_data.head())

        # Save data info
        print("\nDEBUG: Saving data info...")
        with open('data_info.txt', 'w') as f:
            # Write DataFrame info
            f.write("Data Information:\n\n")
            buffer = StringIO()
            processed_data.info(buf=buffer)
            f.write(buffer.getvalue())

            # Write DataFrame description
            f.write("\n\nData Statistics:\n")
            f.write(processed_data.describe().to_string())

            # Write columns information
            f.write("\n\nColumns and Non-null Counts:\n")
            f.write(processed_data.isnull().sum().to_string())

            # Write data types
            f.write("\n\nData Types:\n")
            f.write(processed_data.dtypes.to_string())

        print("DEBUG: Data info saved to data_info.txt")

    else:
        print("ERROR: No processed data available.")

if __name__ == "__main__":
    from io import StringIO
    main()
