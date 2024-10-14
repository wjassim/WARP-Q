"""

WARP-Q: Quality Prediction For Generative Neural Speech Codecs

This code implements the WARP-Q speech quality metric, as outlined in the following papers:

[1] W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “Speech quality assessment
    with WARP‐Q: From similarity to subsequence dynamic time warp cost,” 
    IET Signal Processing, 1– 21 (2022)

[2] W. A. Jassim, J. Skoglund, M. Chinen, and A. Hines, “WARP-Q: Quality prediction 
    for generative neural speech codecs,” ICASSP 2021 - 2021 IEEE International 
    Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021, pp. 401-405
    
Warning: Although this code has been tested and documented, it currently lacks comprehensive 
exception handling and input validation. Providing invalid input may cause the code to fail 
or produce incorrect results without clear error messages.


Dr. Wissam Jassim
Email: wissam.a.jassim@gmail.com
Date: September 30, 2024

"""

import librosa
import numpy as np
from numpy.lib.stride_tricks import as_strided
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
import os
import matplotlib.pyplot as plt
import pandas as pd
import warnings


def load_audio(ref_path, deg_path, sr=None, native_sr=True, verbose=True):
    """
    Load reference and degraded audio files.

    Parameters:
    -----------
    ref_path : str
        Path to the reference speech file.

    deg_path : str
        Path to the degraded speech file.

    sr : float, optional
        Sampling frequency to load the signals. Must be provided if native_sr is False.
        Ignored if native_sr is True.

    native_sr : bool, optional
        If True, load the audio with its native sampling rate.
        If False, load the audio resampled to the provided sampling rate (sr).
        User must provide a valid sr when native_sr is False.

    verbose : bool, optional
        If True, print messages explaining how the audio is loaded, whether with its native sampling rate
        or resampled to the provided sampling rate. Default is True.

    Returns:
    --------
    ref_sig : numpy.ndarray
        The loaded reference audio data.

    deg_sig : numpy.ndarray
        The loaded degraded audio data.

    ref_sr : int
        The sampling rate of the reference signal.

    deg_sr : int
        The sampling rate of the degraded signal.
    """

    # Ensure a sampling rate is provided if native_sr is False
    if not native_sr and sr is None:
        raise ValueError("When native_sr is False, you must provide a valid sampling rate (sr).")

    # 1) Load the reference speech signal
    try:
        if native_sr:
            ref_sig, ref_sr = librosa.load(ref_path, sr=None)  # Load with native sampling rate
            if verbose:
                print(f"Reference signal is loaded with its native sampling rate of {ref_sr} Hz.")
        else:
            ref_sig, ref_sr = librosa.load(ref_path, sr=sr)  # Resample to provided sampling rate
            if verbose:
                print(f"Reference signal is loaded and resampled to {sr} Hz.")
    except Exception as e:
        raise ValueError(f"An error occurred while loading reference speech file '{ref_path}': {e}")

    # 2) Load the degraded speech signal
    try:
        if native_sr:
            deg_sig, deg_sr = librosa.load(deg_path, sr=None)  # Load with native sampling rate
            if verbose:
                print(f"Degraded signal is loaded with its native sampling rate of {deg_sr} Hz.")
        else:
            deg_sig, deg_sr = librosa.load(deg_path, sr=sr)  # Resample to provided sampling rate
            if verbose:
                print(f"Degraded signal is loaded and resampled to {sr} Hz.")
    except Exception as e:
        raise ValueError(f"An error occurred while loading degraded speech file '{deg_path}': {e}")

    # 3) Warn if reference and degraded signals have different sampling rates
    if ref_sr != deg_sr:
        warnings.warn(f"Reference file sampling rate {ref_sr} differs from degraded file sampling rate {deg_sr}.")

    return ref_sig, deg_sig, ref_sr, deg_sr


def get_audio_duration(signal, sr):
    """
    Calculate the duration of an audio signal.

    Parameters:
    -----------
    signal : numpy.ndarray
        The audio signal array.

    sr : float
        Sampling rate of the audio signal.

    Returns:
    --------
    duration : float
        Duration of the audio signal in seconds.
    """
    return librosa.get_duration(y=signal, sr=sr)


def cmvnw(vec, win_size=301, variance_normalization=False):
    """
    This function is aimed to perform local cepstral mean and
    variance normalization on a sliding window. The code assumes that
    there is one observation per row.

    Adopted from https://github.com/astorfi/speechpy/blob/master/speechpy/processing.py

    Parameters
    ----------
    vec : array_like
        Input feature matrix
        (size:(num_observation,num_features))
    win_size : int
        The size of sliding window for local normalization.
        Default=301 which is around 3s if 100 Hz rate is
        considered(== 10ms frame stide)
    variance_normalization : bool
        If the variance normilization should be performed or not.

    Returns
    -------
    array_like
        The mean(or mean+variance) normalized feature vector.
    """
    # Get the shapes
    eps = 2**-30
    rows, cols = vec.shape

    # Windows size must be odd.
    assert isinstance(win_size, int), "Size must be of type 'int'!"
    assert win_size % 2 == 1, "Windows size must be odd!"

    # Padding and initial definitions
    pad_size = int((win_size - 1) / 2)
    vec_pad = np.lib.pad(vec, ((pad_size, pad_size), (0, 0)), "symmetric")
    mean_subtracted = np.zeros(np.shape(vec), dtype=np.float32)

    for i in range(rows):
        window = vec_pad[i : i + win_size, :]
        window_mean = np.mean(window, axis=0)
        mean_subtracted[i, :] = vec[i, :] - window_mean

    # Variance normalization
    if variance_normalization:
        variance_normalized = np.zeros(np.shape(vec), dtype=np.float32)
        vec_pad_variance = np.lib.pad(mean_subtracted, ((pad_size, pad_size), (0, 0)), "symmetric")

        # Looping over all observations.
        for i in range(rows):
            window = vec_pad_variance[i : i + win_size, :]
            window_variance = np.std(window, axis=0)
            variance_normalized[i, :] = mean_subtracted[i, :] / (window_variance + eps)
        output = variance_normalized
    else:
        output = mean_subtracted

    return output


def sliding_window_mfcc_strided(mfcc, window_shape, step, time_stamps, pad_with_zeros=False):
    """
    Create sliding windows from the MFCC features using as_strided for efficiency,
    and return the MFCC patches, the time stamps, and the frame indices for each patch.

    Args:
        mfcc (np.ndarray): MFCC feature matrix (n_mfcc, time_steps).
        window_shape (tuple): Shape of the sliding window (n_mfcc, cols_per_patch).
        step (int): Step size for the sliding window.
        time_stamps (np.ndarray): Array of time stamps in seconds for each column (frame) of the MFCC matrix.
        pad_with_zeros (bool): If True, pad the MFCC array with zeros to ensure all columns are included.

    Returns:
        np.ndarray: 3D array of sliding MFCC windows (n_mfcc, num_patches, cols_per_patch).
        list: List of frame time ranges (start_time, end_time) in seconds for each patch.
        list: List of frame indices (start_index, end_index) corresponding to the time array for each patch.
    """
    n_mfcc, time_steps = mfcc.shape

    # If padding is enabled and time_steps do not align with step and window size
    if pad_with_zeros:
        padding_needed = (time_steps - window_shape[1]) % step
        if padding_needed != 0:
            padding_amount = step - padding_needed
            # Pad the MFCC matrix with zeros along the time_steps (columns)
            mfcc = np.pad(mfcc, ((0, 0), (0, padding_amount)), mode="constant", constant_values=0)
            time_steps = mfcc.shape[1]  # Update time_steps after padding

            # Calculate time increment based on the last two values of the time_stamps array
            time_increment = time_stamps[-1] - time_stamps[-2]

            # Extend the time_stamps array to account for the padded zeros
            extra_times = [time_stamps[-1] + (i + 1) * time_increment for i in range(padding_amount)]
            time_stamps = np.concatenate([time_stamps, extra_times])

    # Define the shape and strides for the sliding windows
    new_shape = (n_mfcc, (time_steps - window_shape[1]) // step + 1, window_shape[1])
    new_strides = (mfcc.strides[0], step * mfcc.strides[1], mfcc.strides[1])

    # Use as_strided to generate the sliding windows
    window_mfcc = as_strided(mfcc, shape=new_shape, strides=new_strides)

    # Calculate the time range and frame indices for each patch
    patch_frame_timestamps = []  # For time ranges (start_time, end_time)
    patch_frame_indices = []  # For frame indices (start_index, end_index)

    for i in range(new_shape[1]):  # new_shape[1] is the number of patches
        start_frame = i * step
        end_frame = start_frame + window_shape[1] - 1

        start_time = time_stamps[start_frame]
        end_time = time_stamps[end_frame] if end_frame < len(time_stamps) else time_stamps[-1]  # Handle case where end_frame exceeds

        patch_frame_timestamps.append((start_time, end_time))  # Add start and end timestamps
        patch_frame_indices.append((int(start_frame), int(end_frame)))  # Add start and end frame indices

    # Return the MFCC patches, their corresponding time stamps, and the frame indices
    return window_mfcc, patch_frame_timestamps, patch_frame_indices


def group_dataframe_by_columns(df=None, group_cols=None, agg_cols=None, agg_func="mean", output_csv=None):
    """
    Groups a DataFrame by specified columns, applies an aggregation function to specified columns, and saves the result if requested.

    Parameters:
    -----------
    df : pd.DataFrame or str, required
        A DataFrame or the path to a CSV file to group. If a DataFrame is provided, it will be used directly. If a CSV file path is
        provided, the data will be loaded from the CSV file.

    group_cols : list, required
        A list of column names to group the DataFrame by. The order of the columns defines the hierarchy of grouping.
        For example, grouping by `['Database','Condition', 'Degradation']` will first group by 'con', then within each 'Database' group,
        it will group by 'Condition', and so on. The sequence of columns affects how the groups are structured.

    agg_cols : list, required
        A list of column names to apply the aggregation function to. For instance, `['mos', 'Raw WARP-Q Score']` means
        that the aggregation function will be applied to both of these columns.

    agg_func : str, optional, default='mean'
        The aggregation function to apply to the columns in `agg_cols`. By default, this is 'mean', but you can also apply
        other functions like 'sum', 'min', 'max', etc.

    output_csv : str, optional
        If provided, the resulting DataFrame will be saved as a CSV file with this name.

    Returns:
    --------
    grouped_df : pd.DataFrame
        The resulting DataFrame after grouping and applying the aggregation function.

    Example:
    --------
    >>> group_dataframe_by_columns(df="data.csv",
                                   group_cols=['Database','Condition', 'Degradation'],
                                   agg_cols=["mos", "Raw WARP-Q Score"],
                                   agg_func="mean",
                                   output_csv="grouped_results.csv")
    """

    # Validate input data
    if df is None:
        raise ValueError("'data_or_csv' must be provided.")

    # Load data if a CSV file path is provided
    if isinstance(df, str):
        try:
            data = pd.read_csv(df)
        except Exception as e:
            raise FileNotFoundError(f"Error reading CSV file: {e}")
    else:
        data = df

    # Validate grouping columns and aggregation inputs
    if group_cols is None or agg_cols is None:
        raise ValueError("'group_cols' and 'agg_cols' must be provided.")

    # Create an aggregation dictionary using the same function for all columns
    agg_dict = {col: agg_func for col in agg_cols}

    # Group the DataFrame by the specified columns and apply aggregations
    grouped_df = data.groupby(group_cols).agg(agg_dict).reset_index()

    # Save the grouped DataFrame to a CSV file if output_csv is provided
    if output_csv:
        # Ensure the directory exists
        save_dir = os.path.dirname(output_csv)
        if not os.path.exists(save_dir) and save_dir:
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        # Check if the file has a .csv extension, and append if necessary
        if not output_csv.endswith(".csv"):
            output_csv += ".csv"  # Append .csv if not present

        grouped_df.to_csv(output_csv, index=False)
        print(f"Data saved to {output_csv}")

    return grouped_df


def plot_warpq_scores(
    df,
    mos_col,
    warpq_col="Raw WARP-Q Score",
    hue_col=None,
    style_col=None,
    title="MOS vs WARP-Q",
    plot_width=None,
    plot_height=None,
    font_size=12,
    marker_size=150,
    legend_cols=1,
    legend_position="best",
    bbox_to_anchor=None,
    legend_font_size=None,
    handletextpad=0.5,
    columnspacing=1.0,
    save_path=None,
):
    """
    Plot the WARP-Q scores against the MOS scores and calculate the Pearson and Spearman correlation coefficients.
    The scores are assigned different markers and styles to enhance the visualization of score distribution, highlighting the
    various categories and degradation levels of the database.

    Args:
        df (pd.DataFrame or str): DataFrame or path to a CSV file containing MOS and WARP-Q scores.
        mos_col (str): Column name for MOS (Mean Opinion Score).
        warpq_col (str): Column name for WARP-Q scores. Default is 'Raw WARP-Q Score'.
        hue_col (str, optional): Column name used for color encoding in the scatter plot.
        style_col (str, optional): Column name used to differentiate marker styles in the scatter plot.
        title (str, optional): Title of the plot. Default is 'MOS vs WARP-Q'.
        plot_width (int, optional): Width of the plot. If not provided, uses default size.
        plot_height (int, optional): Height of the plot. If not provided, uses default size.
        font_size (int, optional): Font size for the plot title and labels. Default is 12.
        marker_size (int, optional): Size of the markers in the scatter plot. Default is 100.
        save_path (str, optional): Path to save the plot, including filename. If None, the plot is not saved.
                                   The .png extension will be added if not provided.

        Note: When there are many degradation types or grouping categories, the legend may exceed the plot space.
        In such cases, changining the following parameters can help manage the space.

        legend_cols (int, optional): Number of columns in the legend. Default is 1.
        legend_position (str, optional): The location of the legend (e.g., 'best', 'upper right').
                                         This parameter works the same as in Seaborn and Matplotlib.
                                         Default is 'best'.
        bbox_to_anchor (tuple, optional): Tuple for the bbox_to_anchor parameter to fine-tune the legend location.
                                          This works the same as in Seaborn and Matplotlib.
                                          For example, bbox_to_anchor=(1, 0.5) will place the legend outside the plot
                                          on the right, halfway up. Default is None.
        legend_font_size (int, optional): Font size for the legend. If None, defaults to font_size.
        handletextpad (float, optional): Space between the legend marker and the text. Default is 0.5.
        columnspacing (float, optional): Space between columns in the legend. Default is 1.0.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    # Set theme for seaborn
    # sns.color_palette()
    sns.set_theme()

    # Check if df is a path to a CSV or already a DataFrame
    if isinstance(df, str):
        try:
            df = pd.read_csv(df)
        except Exception as e:
            raise FileNotFoundError(f"Error reading CSV file: {e}")

    # Validate the presence of required columns in the DataFrame
    if mos_col not in df.columns:
        raise ValueError(f"MOS column '{mos_col}' does not exist in the DataFrame.")
    if warpq_col not in df.columns:
        raise ValueError(f"WARP-Q score column '{warpq_col}' does not exist in the DataFrame.")
    if hue_col is not None and hue_col not in df.columns:
        raise ValueError(f"Hue column '{hue_col}' does not exist in the DataFrame.")
    if style_col is not None and style_col not in df.columns:
        raise ValueError(f"Style column '{style_col}' does not exist in the DataFrame.")

    # Calculate Pearson and Spearman correlations
    pearson_r, _ = pearsonr(df[warpq_col], df[mos_col])
    spearman_r, _ = spearmanr(df[warpq_col], df[mos_col])

    # Round correlation values to 2 digits
    pearson_r = round(pearson_r, 2)
    spearman_r = round(spearman_r, 2)

    # Create the scatter plot
    if plot_width is not None and plot_height is not None:
        fig, ax = plt.subplots(figsize=(plot_width, plot_height), layout="constrained")  # Use user-defined size
    else:
        fig, ax = plt.subplots(layout="constrained")

    sns.scatterplot(x=mos_col, y=warpq_col, hue=hue_col, style=style_col, data=df, s=marker_size, ax=ax, palette="deep")

    # Set title and labels with user-defined font size
    ax.set_title(f"{title} - Pearson: {pearson_r}, Spearman: {spearman_r}", fontsize=font_size)
    ax.set_xlabel(mos_col, fontsize=font_size)
    ax.set_ylabel(warpq_col, fontsize=font_size)

    # Set default legend font size if not provided
    if legend_font_size is None:
        legend_font_size = font_size

    # Move the legend if hue_col is defined
    if hue_col is not None:
        sns.move_legend(
            ax,
            loc=legend_position,
            bbox_to_anchor=bbox_to_anchor,
            ncol=legend_cols,
            title=hue_col,
            fontsize=legend_font_size,
            handletextpad=handletextpad,
            columnspacing=columnspacing,
        )

    # Save the plot if a save path is provided
    if save_path:
        # Ensure the directory exists
        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir) and save_dir:
            os.makedirs(save_dir)  # Create the directory if it doesn't exist

        # Add .png extension if not provided
        if not save_path.lower().endswith(".png"):
            save_path += ".png"

        fig.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path}")

    return fig
