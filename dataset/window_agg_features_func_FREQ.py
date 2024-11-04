from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pandas as pd
import numpy as np
from dataset.logging_tools import default_logger
from pathlib import Path
from dataset.configs.history_data_crawlers_config import root_path, symbols_dict
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from numpy import rms
from scipy.signal import cwt, ricker



def cal_window_max(array, window_size):
    res = np.zeros([array.shape[0], 4])
    res[:window_size, :] = np.nan
    import numpy as np
from scipy.signal import stft
from scipy.fftpack import fft

def calculate_log_returns(array):
    # Calculate the logarithmic returns of the array
    return np.log(array[1:] / array[:-1])

def cal_frequency_features(array, window_size, method #="stft"
                           ):
    # Initialize result matrix with NaNs for compatibility with rolling windows
    res = np.full([array.shape[0], 4], np.nan)  
    
    # Calculate log returns on close prices
    log_returns = calculate_log_returns(array[:, 0])  # Assuming array[:, 0] contains "close" prices
    log_returns = np.insert(log_returns, 0, 0)  # Insert 0 for the first value to match the length

    for i in range(window_size, array.shape[0]):
        selected_slice = log_returns[i - window_size + 1 : i + 1]

        if method == "stft":
            # Compute STFT on selected slice
            f, t, Zxx = stft(selected_slice, nperseg=window_size)
            power_spectrum = np.abs(Zxx) ** 2
            max_freq_power = np.max(power_spectrum)  # Max power in frequency spectrum
            mean_freq_power = np.mean(power_spectrum)  # Mean power in frequency spectrum
            
            # Store in res matrix
            res[i, :] = [
                max_freq_power,
                mean_freq_power,
                np.max(selected_slice),  # Max log return for reference
                np.mean(selected_slice)  # Mean log return for reference
            ]
        
        elif method == "fft":
            # Compute FFT on selected slice
            fft_result = fft(selected_slice)
            power_spectrum = np.abs(fft_result) ** 2
            max_freq_power = np.max(power_spectrum)
            mean_freq_power = np.mean(power_spectrum)
            
            res[i, :] = [
                max_freq_power,
                mean_freq_power,
                np.max(selected_slice),
                np.mean(selected_slice)
            ]

        elif method == "wavelet":
            #from scipy.signal import cwt, ricker
            # Define scales to represent the different 'levels' as in wavedec
            scales_lvl2 = [1, 2]
            scales_lvl3 = [1, 2, 3]
            scales_lvl4 = [1, 2, 3, 4]
            scales_lvl5 = [1, 2, 3, 4, 5]

            # Perform CWT using the Ricker wavelet
            coeffs_ricker_lvl2 = cwt(selected_slice, ricker, scales_lvl2)
            coeffs_ricker_lvl3 = cwt(selected_slice, ricker, scales_lvl3)
            coeffs_ricker_lvl4 = cwt(selected_slice, ricker, scales_lvl4)
            coeffs_ricker_lvl5 = cwt(selected_slice, ricker, scales_lvl5)

            # Extract statistical features for each level
            res = []

            # Level 2
            for coeff in coeffs_ricker_lvl2:
                res.extend([
                    np.max(np.abs(coeff)),
                    np.mean(np.abs(coeff)),
                    np.std(coeff),
                    np.min(coeff),
                    np.max(coeff),
                    np.mean(coeff),
                    np.var(coeff),
                    np.sqrt(np.mean(coeff**2)),  # RMS
                    np.mean(coeff**3) / (np.std(coeff)**3),  # Skewness
                    np.mean(coeff**4) / (np.var(coeff)**2) - 3  # Kurtosis
                ])

            # Repeat for Level 3
            for coeff in coeffs_ricker_lvl3:
                res.extend([
                    np.max(np.abs(coeff)),
                    np.mean(np.abs(coeff)),
                    np.std(coeff),
                    np.min(coeff),
                    np.max(coeff),
                    np.mean(coeff),
                    np.var(coeff),
                    np.sqrt(np.mean(coeff**2)),  # RMS
                    np.mean(coeff**3) / (np.std(coeff)**3),  # Skewness
                    np.mean(coeff**4) / (np.var(coeff)**2) - 3  # Kurtosis
                ])

            # Repeat for Level 4
            for coeff in coeffs_ricker_lvl4:
                res.extend([
                    np.max(np.abs(coeff)),
                    np.mean(np.abs(coeff)),
                    np.std(coeff),
                    np.min(coeff),
                    np.max(coeff),
                    np.mean(coeff),
                    np.var(coeff),
                    np.sqrt(np.mean(coeff**2)),  # RMS
                    np.mean(coeff**3) / (np.std(coeff)**3),  # Skewness
                    np.mean(coeff**4) / (np.var(coeff)**2) - 3  # Kurtosis
                ])

            # Repeat for Level 5
            for coeff in coeffs_ricker_lvl5:
                res.extend([
                    np.max(np.abs(coeff)),
                    np.mean(np.abs(coeff)),
                    np.std(coeff),
                    np.min(coeff),
                    np.max(coeff),
                    np.mean(coeff),
                    np.var(coeff),
                    np.sqrt(np.mean(coeff**2)),  # RMS
                    np.mean(coeff**3) / (np.std(coeff)**3),  # Skewness
                    np.mean(coeff**4) / (np.var(coeff)**2) - 3  # Kurtosis
                ])


# Main function to add CWT features to DataFrame

def add_win_fe_base_func_FREQ( 
    df, symbol, raw_features, timeframes, window_sizes, round_to=3, fe_prefix="fe_WIN_FREQ"
):
    for tf in timeframes:
        for w_size in window_sizes:
            assert tf == 5, "!!! For now, this code only works with 5M timeframe, tf must be 5."

            # Define column names for STFT, FFT, and Wavelet features with all metrics
            cols = {}
            metrics = ['max', 'min', 'mean', 'std', 'skew', 'kurtosis', 'rms']
            methods = ['stft', 'fft', 'wavelet']

            for method in methods:
                cols[method] = {metric: f"{fe_prefix}_{method}_{metric}_W{w_size}_M{tf}" for metric in metrics}

            # Convert the DataFrame column to a NumPy array for calculations
            array = df[raw_features].to_numpy()
            
            # Calculate frequency domain features for each method
            stft_res = cal_frequency_features(array, w_size, method="stft")
            fft_res = cal_frequency_features(array, w_size, method="fft")
            wavelet_res = cal_frequency_features(array, w_size, method="wavelet")

            # Add the STFT features
            df[cols['stft']['max']] = np.round(stft_res[:, 0], round_to)
            df[cols['stft']['min']] = np.round(stft_res[:, 1], round_to)
            df[cols['stft']['mean']] = np.round(stft_res[:, 2], round_to)
            df[cols['stft']['std']] = np.round(stft_res[:, 3], round_to)
            df[cols['stft']['skew']] = np.round(stft_res[:, 4], round_to)
            df[cols['stft']['kurtosis']] = np.round(stft_res[:, 5], round_to)
            df[cols['stft']['rms']] = np.round(stft_res[:, 6], round_to)

            # Add the FFT features
            df[cols['fft']['max']] = np.round(fft_res[:, 0], round_to)
            df[cols['fft']['min']] = np.round(fft_res[:, 1], round_to)
            df[cols['fft']['mean']] = np.round(fft_res[:, 2], round_to)
            df[cols['fft']['std']] = np.round(fft_res[:, 3], round_to)
            df[cols['fft']['skew']] = np.round(fft_res[:, 4], round_to)
            df[cols['fft']['kurtosis']] = np.round(fft_res[:, 5], round_to)
            df[cols['fft']['rms']] = np.round(fft_res[:, 6], round_to)

            # Add the Wavelet features
            df[cols['wavelet']['max']] = np.round(wavelet_res[:, 0], round_to)
            df[cols['wavelet']['min']] = np.round(wavelet_res[:, 1], round_to)
            df[cols['wavelet']['mean']] = np.round(wavelet_res[:, 2], round_to)
            df[cols['wavelet']['std']] = np.round(wavelet_res[:, 3], round_to)
            df[cols['wavelet']['skew']] = np.round(wavelet_res[:, 4], round_to)
            df[cols['wavelet']['kurtosis']] = np.round(wavelet_res[:, 5], round_to)
            df[cols['wavelet']['rms']] = np.round(wavelet_res[:, 6], round_to)

    return df



def history_fe_WIN_features_FREQ(feature_config, logger=default_logger):

    logger.info("- " * 25)
    logger.info("--> start history_fe_WIN_FREQ sfeatures func:")
    try:

        fe_prefix = "fe_WIN_FREQ"
        features_folder_path = f"{root_path}/data/features/{fe_prefix}/"
        Path(features_folder_path).mkdir(parents=True, exist_ok=True)
        base_candle_folder_path = f"{root_path}/data/realtime_candle/"
        round_to = 4

        for symbol in list(feature_config.keys()):
            logger.info(f"---> symbol: {symbol}")
            logger.info("= " * 40)
            

            base_cols = feature_config[symbol][fe_prefix]["base_columns"]
            raw_features = [f"M5_{base_col}" for base_col in base_cols]
            needed_columns = ["_time", "minutesPassed", "symbol"] + raw_features
            file_name = base_candle_folder_path + f"{symbol}_realtime_candle.parquet"
            df = pd.read_parquet(file_name, columns=needed_columns)
            df.sort_values("_time", inplace=True)
     
            df["_time"] = df["_time"].dt.tz_localize(None)
            df.drop(columns=["symbol"])
            df.sort_values("_time", inplace=True)

            df = add_win_fe_base_func_FREQ(
                df,
                symbol,
                raw_features=raw_features,
                timeframes=feature_config[symbol][fe_prefix]["timeframe"],
                window_sizes=feature_config[symbol][fe_prefix]["window_size"],
                round_to=round_to,
                fe_prefix="fe_WIN_FREQ",
            )
            
            # ??
            df.drop(columns=raw_features + ["minutesPassed"], inplace=True)
            df["symbol"] = symbol
            df.to_parquet(f"{features_folder_path}/{fe_prefix}_{symbol}.parquet")
        logger.info("--> history_fe_WIN_features_FREQ run successfully.")
    except Exception as e:
        logger.exception("--> history_fe_WIN_features_FREQ error.")
        logger.exception(f"--> error: {e}")
        raise ValueError("!!!")


if __name__ == "__main__":
    from configs.feature_configs_general import generate_general_config
    config_general = generate_general_config()
    history_fe_WIN_features_FREQ(config_general)
    default_logger.info(f"--> history_fe_WIN_features_FREQ DONE.")
