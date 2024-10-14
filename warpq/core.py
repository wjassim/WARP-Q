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
Date: October 6, 2024

"""

import librosa
import pandas as pd
import numpy as np
from pyvad import vad
from .utils import load_audio, cmvnw, get_audio_duration
from tqdm import tqdm
import warnings
import multiprocessing
from joblib import Parallel, delayed
import os


class warpqMetric:
    """
    WARP-Q Metric: Main class to estimate quality scores for speech codecs.
    It contains methods to evaluate scores between two audio files or multiple files from a CSV.

    Initialize the WARP-Q metric object.

    Args:
        sr (int): Sampling frequency of audio signals in Hertz (Hz). This sampling rate will be used
                  for designing the metric and its frequency representation (e.g., MFCCs, framing). Default is 16000.
        native_sr (bool): If True, audio files will be loaded using their native sampling rate. If False,
                          the class-defined `sr` will be used to resample the audio files.
        frame_ms (float): Length of audio frame in milliseconds for framing.
        overlap_ms (float): Length of overlap between consecutive frames in milliseconds.
        n_mfcc (int): Number of MFCCs to compute.
        fmax (int): Cutoff frequency for MFCC.
        patch_size (float): Size of each patch in seconds for processing.
        patch_hop (float): Hop size between patches in seconds.
        sigma (list): Step size condition for DTW.
        apply_vad (bool): Whether to apply Voice Activity Detection (VAD).
        score_fn (str): Function to compute the final score ('mean' or 'median').
        cmvnw_win_time (float): The size of the window (in seconds) used for computing local
                                Cepstral Mean and Variance Normalization (CMVN) for MFCC features.
        max_score (float): Maximum raw score for normalization. Normalization is done such that
                           the highest raw WARP-Q score is mapped to 0 and the lowest value (close to 0)
                           is mapped to 1. Default is 3.5.
        n_jobs (int): Number of cores to use for parallel processing. If None or -1, use all available cores.

    """

    def __init__(
        self,
        sr=16000,  # Sampling frequency of audio signals in Hertz (Hz)
        native_sr=False,  # If True, load audio with native sampling rate, otherwise resample to sr
        frame_ms=32,  # Length of audio frame in milliseconds for framing
        overlap_ms=4,  # Length of overlap between consecutive frames in milliseconds
        n_mfcc=13,  # Number of Mel-Frequency Cepstral Coefficients to compute
        fmax=5000,  # Cutoff frequency for MFCC computation
        patch_size=0.4,  # Size of each patch in seconds for processing
        patch_hop=0.2,  # Hop size between patches in seconds
        sigma=[[1, 0], [0, 3], [1, 3]],  # Step size conditions for Subsequence Dynamic Time Warping (SDTW)
        apply_vad=True,  # Flag to determine if Voice Activity Detection (VAD) should be applied
        score_fn="median",  # Function to compute the final score ('mean' or 'median')
        cmvnw_win_time=0.836,  # The size of sliding window for local normalization (in seconds)
        max_score=3.5,  # Maximum raw score for normalization
        n_jobs=-1,  # Number of cores to use for parallel processing. If None or -1, use all available cores
    ):

        self.sr = sr
        self.native_sr = native_sr

        self.n_mfcc = n_mfcc

        # Calculate the Nyquist frequency
        nyquist = sr / 2
        if fmax > nyquist:
            raise ValueError(
                f"Invalid fmax value: {fmax}. It must be less than or equal to the Nyquist frequency ({nyquist} Hz) based on the sampling rate of {self.sr} Hz."
            )
        self.fmax = fmax

        self.patch_size = patch_size
        if patch_hop >= self.patch_size:
            raise ValueError(f"Invalid patch_hop: {patch_hop}")
        self.patch_hop = patch_hop

        self.sigma = sigma
        self.apply_vad = apply_vad

        # MFCC and DTW parameters
        # self.win_length = int(0.032 * self.sr)  # 32 ms frame
        # self.hop_length = int(0.004 * self.sr)  # 4 ms overlap
        if overlap_ms >= frame_ms:
            raise ValueError("overlap_length_ms must be less than frame_length_ms.")
        # self.win_length = int(frame_ms * self.sr / 1000)  # Convert ms to samples
        self.win_length = librosa.time_to_samples(frame_ms / 1000, sr=self.sr)  # Convert ms to samples
        # self.hop_length = int(overlap_ms * self.sr / 1000)  # Convert ms to samples
        self.hop_length = librosa.time_to_samples(overlap_ms / 1000, sr=self.sr)  # Convert ms to samples
        self.dtw_metric = "euclidean"
        self.n_fft = 2 * self.win_length
        self.lifter = 3

        # cols_per_patch = int(self.patch_size / (self.hop_length / self.sr))
        # step = int(cols_per_patch / 2)  # Step size for sliding window
        self.cols_per_patch = librosa.time_to_frames(self.patch_size, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)
        self.sliding_window_shape = (self.n_mfcc, self.cols_per_patch)  # (n_mfcc, cols_per_patch)
        self.step = librosa.time_to_frames(self.patch_hop, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)

        # VAD parameters
        self.hop_size_vad = 30
        self.sr_vad = self.sr
        self.aggresive = 0

        # Sliding window for Cepstral Mean and Variance Normalization (CMVN)
        cmvnw_win_size = librosa.time_to_frames(cmvnw_win_time, sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)
        if cmvnw_win_size % 2 == 0:  # Check if it's even
            # If it's even, adjust it to the nearest odd number
            cmvnw_win_size += 1
        self.cmvnw_win_size = int(cmvnw_win_size)

        # Function to compute the final score (median or mean)
        if score_fn not in ["median", "mean"]:
            raise ValueError("score_fn must be either 'mean' or 'median'")
        self.score_fn = score_fn

        # Define normalization parameter
        if max_score <= 0:
            raise ValueError(f"max_score must be greater than 0, but got {max_score}")
        self.max_score = max_score

        # Detect available CPU cores
        if n_jobs is None or n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()  # Use all available cores
        else:
            self.n_jobs = n_jobs

        # self.print_msg = True

    def evaluate(self, ref_audio, deg_audio, arr_sr=None, save_csv_path=None, verbose=False):
        """
        Compute WARP-Q score between two input speech signals.

        Accepts file paths or arrays.

        Args:
            ref_audio (str or np.ndarray): Path to reference speech file or reference audio array.
            deg_audio (str or np.ndarray): Path to degraded speech file or degraded audio array.
            arr_sr (int, optional): Sampling rate. Required if audio arrays are provided.
            save_csv_path (str, optional): Path to save detailed results to a CSV file. If None, no file is saved.
                                           If a valid path is provided, the results will be saved in CSV format,
                                           with columns including reference and degraded audio descriptions, WARP-Q scores,
                                           alignment costs, and timing information for each patch. If the file already exists,
                                           new results will be appended without the header.
            verbose (bool, optional): If True, print messages about the process and results.

        Returns:
            dict: Dictionary with WARP-Q results and detailed alignment information:
                - 'raw_warpq_score': The computed WARP-Q score between the reference and degraded audio.

                - 'normalized_warpq_score': The normalized WARP-Q score between 0 and 1, where 1 indicates best quality.

                - 'total_patch_count': The total number of patches generated from the degraded signal's MFCC, representing the
                    number of segments in the degraded signal after applying the sliding window.

                - 'alignment_costs': A list of DTW alignment costs for each degraded MFCC patch, representing how well each
                    patch matches its aligned subsequence in the reference MFCC. Length is equal to 'total_patch_count'.

                - 'aligned_ref_time_ranges': List of (start_time, end_time) tuples containing the start and end time stamps
                    (in seconds) for the best matching subsequences in the reference MFCC, as aligned to each patch in the
                    degraded signal using DTW. Length is equal to 'total_patch_count'.

                - 'aligned_ref_frame_indices': List of (a_ast, b_ast) tuples containing the start and end frame indices for
                    the best matching subsequences in the reference MFCC, corresponding to the aligned subsequences. Length is equal to
                    'total_patch_count'.

                - 'deg_patch_time_ranges': List of (start_time, end_time) tuples containing the start and end time stamps
                    (in seconds) for each patch in the degraded signal's MFCC, generated using a sliding window approach. Length is equal to
                    'total_patch_count'.

                - 'deg_patch_frame_indices': List of (start_frame, end_frame) tuples containing the start and end frame indices
                    for each patch in the degraded signal's MFCC, corresponding to the patches created by the sliding window process. Length is equal
                    to 'total_patch_count'.

        Additional Info:

            - Normalization of scores:
                Since lower WARP-Q scores represent better quality and higher WARP-Q scores represent worse quality,
                normalization will be done such that the highest value is mapped to 0 and the lowest value (close to 0)
                is mapped to 1. This can be achieved using the formula:

                                                            raw_warpq_score
                            normalized_warpq_score = 1 - ( ----------------- )
                                                            self.max_score

                Here, the default value for self.max_score is set to 3.5, based on tests conducted on the datasets
                presented in our published papers.In addition, values are clamped between 0 and 1 to ensure
                the normalized score stays within this range.

                It is possible to normalize the raw WARP-Q score to align with the Mean Opinion Score (MOS), which typically ranges from 1 to 5.
                This can be achieved using a simple linear mapping with the following formula:

                                                                    raw_warpq_score
                            normalized_warpq_score = 1 + 4 * (1 - ( ----------------- ) )
                                                                    self.max_score

                In the current implementation, we use the normalized score on a scale of 0 to 1 for simplicity. A more robust mapping to
                the MOS scale will be added in future updates.
        """

        if verbose:
            print(f"Computing WARP-Q score between two audio files:")

        # 1) Check if ref_audio and deg_audio are file paths (str) or audio arrays (np.ndarray)
        if isinstance(ref_audio, str) and isinstance(deg_audio, str):
            # Load audio from paths
            ref_signal, deg_signal, ref_sr, deg_sr = load_audio(ref_audio, deg_audio, self.sr, self.native_sr, verbose=False)

            # Check if sampling rates differ from self.sr when native_sr is True
            if self.native_sr and (ref_sr != self.sr or deg_sr != self.sr):
                raise ValueError(
                    f"Loaded sampling rates do not match the class-defined sampling rate (self.sr = {self.sr} Hz).\n"
                    f"Reference file: {ref_audio} - Sampling rate: {ref_sr} Hz\n"
                    f"Degraded file: {deg_audio} - Sampling rate: {deg_sr} Hz\n"
                    "This creates a conflict for metric design. Please ensure that you either:\n"
                    "1. Set native_sr=False to resample the audio to the class-defined sr.\n"
                    "2. Match the sampling rate of the audio files to the class-defined sr."
                )

        elif isinstance(ref_audio, np.ndarray) and isinstance(deg_audio, np.ndarray):
            # Ensure the user has provided a valid sampling rate when using arrays
            if arr_sr is None:
                raise ValueError("Sampling rate 'arr_sr' must be provided when using audio arrays.")
            if arr_sr != self.sr:
                raise ValueError(f"The provided sampling rate {arr_sr} does not match the class-defined sampling rate {self.sr}.")
            ref_signal, deg_signal = ref_audio, deg_audio
        else:
            raise ValueError("Both ref_audio and deg_audio must be either file paths (str) or audio arrays (np.ndarray).")

        # 2) Apply VAD if required
        if self.apply_vad:
            ref_signal = ref_signal[vad(ref_signal, self.sr, hop_length=self.hop_size_vad) == 1]
            deg_signal = deg_signal[vad(deg_signal, self.sr, hop_length=self.hop_size_vad) == 1]
            vad_text = "(VAD applied)"  # Note that VAD was applied
        else:
            vad_text = "(VAD not applied)"  # Indicate VAD wasn't applied

        # 3) Calculate the duration of the loaded audio signals
        ref_duration = get_audio_duration(ref_signal, self.sr)
        deg_duration = get_audio_duration(deg_signal, self.sr)

        # Handle case where the reference and degraded signals are too short for processing
        audio_files_too_short = ref_duration < self.patch_size or deg_duration < self.patch_size

        if audio_files_too_short:
            short_files = []

            # Check if the reference signal is shorter than the patch size and if a path is provided
            if ref_duration < self.patch_size:
                ref_info = f"Reference signal: {ref_duration:.2f}s"
                if isinstance(ref_audio, str):
                    ref_info += f" (file: {ref_audio})"
                short_files.append(ref_info)

            # Check if the degraded signal is shorter than the patch size and if a path is provided
            if deg_duration < self.patch_size:
                deg_info = f"Degraded signal: {deg_duration:.2f}s"
                if isinstance(deg_audio, str):
                    deg_info += f" (file: {deg_audio})"
                short_files.append(deg_info)

            # Join the information about the short files into a single message
            short_files_info = "; ".join(short_files)

            warnings.warn(
                f"\nAudio signals too short {vad_text}, shorter than the patch size {self.patch_size}s. "
                f"Short audio files: {short_files_info}. Returning a dictionary with NaN values; files skipped from calculations.",
                UserWarning,
            )
            return {
                "raw_warpq_score": np.nan,
                "normalized_warpq_score": np.nan,
                "total_patch_count": np.nan,
                "alignment_costs": np.nan,
                "aligned_ref_time_ranges": np.nan,
                "aligned_ref_frame_indices": np.nan,
                "deg_patch_time_ranges": np.nan,
                "deg_patch_frame_indices": np.nan,
            }

        # 4) Compute MFCC features
        mfcc_ref = librosa.feature.mfcc(
            y=ref_signal,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            fmax=self.fmax,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            lifter=self.lifter,
        )
        mfcc_deg = librosa.feature.mfcc(
            y=deg_signal,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            fmax=self.fmax,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            lifter=self.lifter,
        )
        # mfcc_ref = librosa.feature.delta(mfcc_ref, order=3)
        # mfcc_deg = librosa.feature.delta(mfcc_deg, order=3)

        # Get number of frames (columns) for both MFCCs
        num_frames_ref = mfcc_ref.shape[1]
        num_frames_deg = mfcc_deg.shape[1]

        # Get time stamps for each frame in both arrays
        time_ref = librosa.frames_to_time(range(num_frames_ref), sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)
        time_deg = librosa.frames_to_time(range(num_frames_deg), sr=self.sr, hop_length=self.hop_length, n_fft=self.n_fft)

        # 5) Normalize MFCC features
        mfcc_ref = cmvnw(mfcc_ref.T, win_size=self.cmvnw_win_size, variance_normalization=True).T
        mfcc_deg = cmvnw(mfcc_deg.T, win_size=self.cmvnw_win_size, variance_normalization=True).T

        # 6) Call the function to get the alignment results of each patch
        alignment_costs, deg_patch_time_ranges, deg_patch_frame_indices, aligned_ref_time_ranges, aligned_ref_frame_indices = self.align_mfcc_patches(
            mfcc_deg, mfcc_ref, time_deg, time_ref
        )

        # alignment_results = self.align_mfcc_patches(mfcc_deg, mfcc_ref, time_deg, time_ref)
        # alignment_costs = alignment_results[0]
        # deg_patch_time_ranges = alignment_results[1]
        # deg_patch_frame_indices = alignment_results[2]
        # aligned_ref_time_ranges = alignment_results[3]
        # aligned_ref_frame_indices = alignment_results[4]

        # 7) Compute the final score based on user choice (mean or median)
        if self.score_fn == "median":
            raw_warpq_score = np.median(alignment_costs)
        else:
            raw_warpq_score = np.mean(alignment_costs)

        # Round the raw score
        raw_warpq_score = round(raw_warpq_score, 3)

        # 8) Normalize the score and clamp values between 0 and 1
        normalized_warpq_score = round(1 - (raw_warpq_score / self.max_score), 3)
        normalized_warpq_score = min(1, max(0, normalized_warpq_score))  # Clamp between 0 and 1

        total_patch_count = len(alignment_costs)

        result = {
            "raw_warpq_score": raw_warpq_score,
            "normalized_warpq_score": normalized_warpq_score,
            "total_patch_count": total_patch_count,
            "alignment_costs": [round(cost, 3) for cost in alignment_costs],
            "aligned_ref_time_ranges": aligned_ref_time_ranges,
            "aligned_ref_frame_indices": aligned_ref_frame_indices,
            "deg_patch_time_ranges": deg_patch_time_ranges,
            "deg_patch_frame_indices": deg_patch_frame_indices,
        }

        # 9) If save_csv_path is provided, save the detailed results to a CSV
        if save_csv_path:
            # Determine the content for the ref_audio and deg_audio columns
            ref_audio_desc = ref_audio if isinstance(ref_audio, str) else "audio array provided by user"
            deg_audio_desc = deg_audio if isinstance(deg_audio, str) else "audio array provided by user"

            # Prepare multiple rows based on the length of alignment_costs
            rows = []

            for i in range(total_patch_count):
                row = {
                    "ref_audio": ref_audio_desc if i == 0 else "",  # Only include ref_audio on the first row
                    "deg_audio": deg_audio_desc if i == 0 else "",  # Only include deg_audio on the first row
                    "raw_warpq_score": result["raw_warpq_score"] if i == 0 else "",  # Only save raw_warpq_score on the first row
                    "normalized_warpq_score": result["normalized_warpq_score"] if i == 0 else "",  # Save normalized score
                    "total_patch_count": total_patch_count if i == 0 else "",
                    "patch_number": i + 1,  # Add a patch number for each row
                    "alignment_costs (unitless)": result["alignment_costs"][i],  # Directly access alignment_costs
                    "deg_patch_start_time (s)": result["deg_patch_time_ranges"][i][0],  # Start time of degraded patch
                    "deg_patch_end_time (s)": result["deg_patch_time_ranges"][i][1],  # End time of degraded patch
                    "aligned_ref_start_time (s)": result["aligned_ref_time_ranges"][i][0],  # Start time of reference
                    "aligned_ref_end_time (s)": result["aligned_ref_time_ranges"][i][1],  # End time of reference
                    "deg_patch_start_frame (frame)": result["deg_patch_frame_indices"][i][0],  # Start frame of degraded patch
                    "deg_patch_end_frame (frame)": result["deg_patch_frame_indices"][i][1],  # End frame of degraded patch
                    "aligned_ref_start_frame (frame)": result["aligned_ref_frame_indices"][i][0],  # Start frame of reference
                    "aligned_ref_end_frame (frame)": result["aligned_ref_frame_indices"][i][1],  # End frame of reference
                }
                rows.append(row)

            # Convert the rows to a DataFrame and append them to the CSV
            df_rows = pd.DataFrame(rows)

            # Ensure the directory exists
            save_dir = os.path.dirname(save_csv_path)
            if not os.path.exists(save_dir) and save_dir:  # Check if the directory exists
                os.makedirs(save_dir)  # Create the directory if it doesn't exist

            # Check if the file has a .csv extension, and append if necessary
            if not save_csv_path.endswith(".csv"):
                save_csv_path += ".csv"  # Append .csv if not present

            # Save DataFrame to CSV
            if not os.path.isfile(save_csv_path):
                df_rows.to_csv(save_csv_path, index=False)  # Create new file if it doesn't exist
            else:
                df_rows.to_csv(save_csv_path, mode="a", header=False, index=False)  # Append to existing file

        if verbose:
            print(f"Raw WARP-Q Score: {raw_warpq_score}")
            print(f"Normalized WARP-Q Score: {normalized_warpq_score}")
            if save_csv_path:
                print(f"Results are saved in {save_csv_path}.")
            print("Done!")

        return result

    def align_mfcc_patches(self, mfcc_deg, mfcc_ref, time_deg, time_ref):
        """
        Divide degraded MFCC features into overlapping patches, align them with reference MFCC using DTW,
        and return alignment costs, time ranges, and frame indices.

        Args:
            mfcc_deg (np.array): Degraded MFCC feature matrix.
            mfcc_ref (np.array): Reference MFCC feature matrix.
            time_deg (np.array): Timestamps (in seconds) corresponding to degraded MFCC columns.
            time_ref (np.array): Timestamps (in seconds) corresponding to reference MFCC columns.

        Returns:
            tuple: A tuple containing:
                - alignment_costs (list): DTW alignment costs for each patch.
                - deg_patch_time_ranges (list): Tuples of (start_time, end_time) in seconds for each degraded patch.
                - deg_patch_frame_indices (list): Tuples of (start_index, end_index) for the frame indices of degraded patches.
                - aligned_ref_time_ranges (list): Tuples of (start_time, end_time) in seconds for each aligned reference segment.
                - aligned_ref_frame_indices (list): Tuples of (start_index, end_index) for the frame indices of aligned reference segments.
        """
        # Lists to store information about each patch
        deg_patch_time_ranges = []
        deg_patch_frame_indices = []
        alignment_costs = []
        aligned_ref_time_ranges = []
        aligned_ref_frame_indices = []

        # Iterate over degraded MFCC columns to extract overlapping patches
        for start_col in range(0, mfcc_deg.shape[1] - self.cols_per_patch + 1, self.step):

            # Define the end column for the patch
            end_col = start_col + self.cols_per_patch

            # Extract patch of MFCC from degraded signal
            patch = mfcc_deg[:, start_col:end_col]  # Extract patch

            # Get corresponding start and end time from the degraded timestamps
            start_time = time_deg[start_col]
            end_time = time_deg[end_col - 1]

            # Store the time range and frame indices for the current patch
            deg_patch_time_ranges.append((start_time, end_time))  # Add start and end timestamps
            deg_patch_frame_indices.append((int(start_col), int(end_col - 1)))  # Add start and end frame indices

            # Perform alignment of the current patch to the reference MFCC using DTW
            alignment_results = self.dtw_alignment_cost(patch, mfcc_ref, time_ref)

            # Store the alignment cost and aligned reference time/frame data
            alignment_costs.append(alignment_results[0])
            aligned_ref_time_ranges.append(alignment_results[1])
            aligned_ref_frame_indices.append(alignment_results[2])

        return alignment_costs, deg_patch_time_ranges, deg_patch_frame_indices, aligned_ref_time_ranges, aligned_ref_frame_indices

    def evaluate_from_csv(
        self,
        input_csv,
        ref_wave_col="ref_wave",
        deg_wave_col="deg_wave",
        raw_score_col="Raw WARP-Q Score",
        output_csv=None,
        save_details=False,
    ):
        """
        Evaluate WARP-Q for multiple files listed in a CSV.

        Args:
            input_csv (str): Path to a CSV file with specified reference and degraded wave columns.
            ref_wave_col (str): Name of the reference wave column. Default is 'ref_wave'.
            deg_wave_col (str): Name of the degraded wave column. Default is 'deg_wave'.
            raw_score_col (str): Column name where raw scores will be saved. Default is 'Raw WARP-Q Score'.
            output_csv (str): Path to save results. If None, results are not saved.
            save_details (bool): If True, save detailed results (alignment costs, times) in the same DataFrame.

        Returns:
            pd.DataFrame: DataFrame with computed WARP-Q scores and detailed results if requested.

            Additional Detailed Results (saved when save_details is True):
                - total_patch_count (int): The total number of patches in the degraded signal.
                - alignment_costs (list): The alignment costs for each patch between the degraded and reference signals.
                - deg_patch_time_ranges (list): List of tuples for (start, end) times in seconds of each patch in the degraded signal.
                - aligned_ref_time_ranges (list): List of tupes for (start, end) times in seconds of the aligned segments in the reference signal.
        """

        # Try to read the CSV, raise an error if file cannot be opened
        try:
            df = pd.read_csv(input_csv)
        except Exception as e:
            raise FileNotFoundError(f"Error loading CSV file at {input_csv}: {str(e)}")

        # Check if the CSV contains the required columns for reference and degraded wave paths
        if ref_wave_col not in df.columns or deg_wave_col not in df.columns:
            raise ValueError(f"CSV file must contain '{ref_wave_col}' and '{deg_wave_col}' columns.")

        total_files = len(df)

        # Get column indices for the reference and degraded wave columns
        ref_wave_idx = df.columns.get_loc(ref_wave_col)
        deg_wave_idx = df.columns.get_loc(deg_wave_col)

        def process_row(row):
            ref_wave = row[ref_wave_idx]
            deg_wave = row[deg_wave_idx]
            # Call the evaluate function for each pair of files (ref and deg waves)
            return self.evaluate(ref_wave, deg_wave, verbose=False)

        # Use joblib's Parallel to process rows concurrently (using multiple cores)
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(process_row)(row)
            for row in tqdm(df.itertuples(index=False), total=total_files, desc=f"Computing WARP-Q scores ({self.n_jobs} jobs)")
        )

        # Unpack the results from the parallel processing and assign to the DataFrame
        df[raw_score_col] = [r["raw_warpq_score"] for r in results]
        df[f"Normalized {raw_score_col}"] = [r["normalized_warpq_score"] for r in results]

        if save_details:
            # Unpack additional metrics only if save_details is True
            # Although this could be done with a single loop and appends, I used list comprehensions for better conciseness and readability.
            df["total_patch_count"] = [r["total_patch_count"] for r in results]
            df["alignment_costs"] = [r["alignment_costs"] for r in results]
            df["deg_patch_time_ranges, (start, end) [s]"] = [r["deg_patch_time_ranges"] for r in results]
            df["aligned_ref_time_ranges, (start, end) [s]"] = [r["aligned_ref_time_ranges"] for r in results]

        # Calculate how many rows were skipped (in case of any missing results)
        skipped_count = total_files - len(df.dropna(subset=[raw_score_col]))
        if skipped_count > 0:
            print(
                f"\n{skipped_count} pairs of reference and degraded audio files out of a total of {total_files} pairs were skipped due to short audio durations."
            )

        # Save the results to a CSV file if output_csv is specified
        if output_csv:
            save_dir = os.path.dirname(output_csv)
            # Ensure the output directory exists, create it if necessary
            if not os.path.exists(save_dir) and save_dir:
                os.makedirs(save_dir)
            # Add .csv extension if it's not present
            if not output_csv.endswith(".csv"):
                output_csv += ".csv"

            # Save the DataFrame to the specified CSV file
            df.to_csv(output_csv, index=False)
            detail_msg = " including additional metrics." if save_details else "."
            print(f"\nResults are saved in {output_csv}{detail_msg}")

        print("Done!")
        return df

    # Function to compute distance based on patch index
    def compute_cost_for_patch(self, index, mfcc_deg_patches, mfcc_ref, time_ref):
        """
        Compute the alignment cost and subsequence information between a patch (identified by index)
        and the full reference MFCC using DTW.

        Args:
            index (int): The index of the patch.
            mfcc_deg_patches (np.ndarray): 3D array of MFCC patches from the degraded signal.
            mfcc_ref (np.ndarray): The full MFCC array of the reference signal.
            time_ref (np.ndarray): Array of time stamps for each frame in the reference MFCC.

        Returns:
            tuple:
                - float: DTW alignment cost (distance) between the patch and the reference MFCC.
                - tuple: (start_time, end_time) for the best matching subsequence in the reference MFCC.
                - tuple: (a_ast, b_ast), the start and end frame indices of the best matching subsequence in the reference MFCC.
        """
        # Extract the patch from the degraded signal using the index
        patch = mfcc_deg_patches[:, index, :]

        # Call the dtw_alignment_cost function to compute the alignment cost and subsequence information
        return self.dtw_alignment_cost(patch, mfcc_ref, time_ref)

    def dtw_alignment_cost(self, patch, mfcc_ref, time_ref):
        """
        Compute alignment cost between two spectral representations using Subsequence DTW.

        Args:
            patch (np.ndarray): MFCC patch from the degraded signal (n_mfcc, patch_length).
            mfcc_ref (np.ndarray): Full reference MFCC matrix (n_mfcc, ref_length).
            time_ref (np.ndarray): Array of time stamps for each frame in the reference MFCC matrix (ref_length,).

        Returns:
            tuple:
                - float: DTW alignment cost between the patch and the best matching subsequence in the reference MFCC.
                - tuple: (start_time, end_time) for the best matching subsequence in the reference MFCC.
                - tuple: (a_ast, b_ast), the start and end frame indices of the best matching subsequence in the reference MFCC.
        """

        # Compute the DTW cost matrix (D) and the optimal path (P)
        D, P = librosa.sequence.dtw(
            X=patch,
            Y=mfcc_ref,
            metric=self.dtw_metric,
            step_sizes_sigma=self.sigma,
            weights_mul=np.array([1, 1, 1]),
            band_rad=0.25,
            subseq=True,
            backtrack=True,
        )

        # Reverse P (P_librosa) to get the alignment path from start to end
        P_librosa = P[::-1, :]

        # Get the best matching subsequence indices in the reference MFCC
        b_ast = P_librosa[-1, 1]
        a_ast = P_librosa[0, 1]

        # Convert the start and end frame indices to time using the time_ref array
        start_time = time_ref[a_ast]
        end_time = time_ref[b_ast]

        # Return the DTW alignment cost, the (start_time, end_time) tuple, and the (a_ast, b_ast) frame indices tuple
        return D[-1, b_ast] / D.shape[0], (start_time, end_time), (int(a_ast), int(b_ast))
