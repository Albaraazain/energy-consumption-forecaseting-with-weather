# time_series_modeling.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def check_stationarity(df):
    """
    Tests for stationarity and performs differencing if needed
    """
    print("\nDEBUG: Checking stationarity...")

    # Perform Dickey-Fuller test
    t = sm.tsa.adfuller(df.energy_mean, autolag='AIC')
    print('ADF Statistic:', t[0])
    print('p-value:', t[1])

    # If not stationary (p > 0.05), perform differencing
    if t[1] > 0.05:
        print("DEBUG: Series is not stationary, performing differencing...")
        diff = df.energy_mean.diff().dropna()

        # Check stationarity of differenced series
        t_diff = sm.tsa.adfuller(diff, autolag='AIC')
        print('Differenced ADF Statistic:', t_diff[0])
        print('Differenced p-value:', t_diff[1])

        return diff
    return df.energy_mean

def analyze_acf_pacf(series):
    """
    Analyzes ACF and PACF plots to determine model parameters
    """
    print("\nDEBUG: Creating ACF/PACF plots...")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    plot_acf(series, lags=50, ax=ax1)
    ax1.set_title('Autocorrelation Function')

    plot_pacf(series, lags=50, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function')

    plt.tight_layout()
    plt.savefig('plots/acf_pacf.png')
    plt.close()

def seasonal_decomposition(series):
    """
    Performs seasonal decomposition of the time series
    """
    print("\nDEBUG: Performing seasonal decomposition...")

    decomposition = sm.tsa.seasonal_decompose(series, period=12)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))

    decomposition.observed.plot(ax=ax1)
    ax1.set_title('Observed')

    decomposition.trend.plot(ax=ax2)
    ax2.set_title('Trend')

    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title('Seasonal')

    decomposition.resid.plot(ax=ax4)
    ax4.set_title('Residual')

    plt.tight_layout()
    plt.savefig('plots/seasonal_decomposition.png')
    plt.close()

    return decomposition

def fit_arimax_model(df):
    """
    Fits ARIMAX model with exogenous variables
    """
    print("\nDEBUG: Fitting ARIMAX model...")

    # Prepare exogenous variables
    exog = sm.add_constant(df[['weather_cluster', 'is_holiday']])

    # Split data into train and test
    train_size = int(len(df) * 0.9)
    train = df[:train_size]
    test = df[train_size:]

    # Fit model
    model = SARIMAX(
        endog=train['energy_mean'],
        exog=exog[:train_size],
        order=(7,1,1),
        seasonal_order=(1,1,0,12)
    )

    print("DEBUG: Training model...")
    results = model.fit()
    print("\nModel Summary:")
    print(results.summary())

    # Make predictions
    print("\nDEBUG: Making predictions...")
    predictions = results.predict(
        start=len(train),
        end=len(df)-1,
        exog=exog[train_size:]
    )

    # Calculate metrics
    mae = mean_absolute_error(test['energy_mean'], predictions)
    mape = np.mean(np.abs((test['energy_mean'] - predictions) / test['energy_mean'])) * 100

    print(f"\nTest MAE: {mae:.4f}")
    print(f"Test MAPE: {mape:.2f}%")

    # Plot results
    plt.figure(figsize=(15, 6))
    plt.plot(test.index, test['energy_mean'], label='Actual')
    plt.plot(test.index, predictions, label='Predicted')
    plt.title('Energy Consumption: Actual vs Predicted')
    plt.legend()
    plt.savefig('plots/arimax_predictions.png')
    plt.close()

    return results, predictions

def main():
    print("DEBUG: Starting time series modeling...")

    try:
        # Load enhanced data
        print("DEBUG: Loading enhanced data...")
        df = pd.read_csv('enhanced_data.csv', parse_dates=['day'])
        df.set_index('day', inplace=True)
        print(f"DEBUG: Data shape: {df.shape}")

        # Check stationarity
        series = check_stationarity(df)

        # Analyze ACF/PACF
        analyze_acf_pacf(series)

        # Perform seasonal decomposition
        decomp = seasonal_decomposition(df.energy_mean)

        # Fit ARIMAX model
        model, predictions = fit_arimax_model(df)

        print("\nDEBUG: Modeling complete! Check the 'plots' directory for visualizations.")

    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()