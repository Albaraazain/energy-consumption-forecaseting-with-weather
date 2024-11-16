import os
import kaggle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def explore_dataset():
    """
    Explores the downloaded dataset structure
    """
    print("Available files in the dataset:")
    for dirname, _, filenames in os.walk('./'):
        for filename in filenames:
            if filename.endswith('.csv'):
                print(os.path.join(dirname, filename))
                # Print first few lines of each CSV
                try:
                    df = pd.read_csv(os.path.join(dirname, filename), nrows=2)
                    print(f"\nColumns in {filename}:")
                    print(df.columns.tolist())
                    print("\nFirst 2 rows:")
                    print(df.head(2))
                    print("\n" + "="*50 + "\n")
                except Exception as e:
                    print(f"Could not read {filename}: {e}\n")

explore_dataset()