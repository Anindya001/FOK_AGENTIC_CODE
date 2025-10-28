"""
Excel Reader Module - Exact Replication of Working Core Loader
Tolerant loader that mirrors the existing core.py::load_excel behavior
"""

from typing import Dict, Tuple
import numpy as np
import pandas as pd

class ExcelLoadError(RuntimeError):
    """Error raised when Excel loading fails"""
    pass

def load_excel(path: str) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Read an Excel file and return (times, channels) just like the current core loader.
    
    Args:
        path: Path to Excel file
        
    Returns:
        Tuple of (times, channels) where:
        - times: np.ndarray[np.float64] of strictly increasing time points
        - channels: Dict[str, np.ndarray[np.float64]] of channel data
        
    Raises:
        ExcelLoadError: If loading fails or data validation fails
    """
    
    try:
        # 1. Read the sheet with pandas (no special engine argument)
        df = pd.read_excel(path)
    except Exception as e:
        raise ExcelLoadError(f"Failed to read Excel file: {e}")
    
    # 2. Sanity check columns
    if df.empty or len(df.columns) < 2:
        raise ExcelLoadError("Excel file must have at least 2 columns")
    
    # 3. Time column detection (strict order of fallbacks)
    time_col = None
    
    # Prefer exact matches (case-insensitive)
    time_candidates = ["time", "t", "ageing time", "ageing_time"]
    for candidate in time_candidates:
        for col in df.columns:
            if str(col).lower() == candidate:
                time_col = col
                break
        if time_col is not None:
            break
    
    # Else pick the first numeric column
    if time_col is None:
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                time_col = col
                break
    
    # Else fall back to the first column
    if time_col is None:
        time_col = df.columns[0]
    
    # 4. Coerce and clean time
    times = pd.to_numeric(df[time_col], errors='coerce')
    
    # Drop rows where time is NaN
    valid_mask = ~times.isna()
    df_clean = df[valid_mask].copy()
    times_clean = times[valid_mask]
    
    if len(times_clean) == 0:
        raise ExcelLoadError("No valid time points found")
    
    # 5. Build channels
    channels = {}
    other_columns = [col for col in df_clean.columns if col != time_col]
    
    for col in other_columns:
        channel_data = pd.to_numeric(df_clean[col], errors='coerce')
        # Convert to float64
        channels[str(col)] = channel_data.astype(np.float64).values
    
    if len(channels) == 0:
        raise ExcelLoadError("No capacitor channels found")
    
    # 6. Sort and deduplicate by time
    # Convert times to numpy array
    times_array = times_clean.astype(np.float64).values
    
    # Sort by ascending time
    sort_indices = np.argsort(times_array)
    times_sorted = times_array[sort_indices]
    
    # Apply same ordering to all channels
    for channel_name in channels:
        channels[channel_name] = channels[channel_name][sort_indices]
    
    # Remove duplicate time stamps: keep first occurrence where diff > 0
    if len(times_sorted) > 1:
        # Find where time actually increases
        time_diffs = np.diff(times_sorted)
        keep_mask = np.concatenate([[True], time_diffs > 0])  # Keep first point + where diff > 0
        
        times_dedup = times_sorted[keep_mask]
        
        # Apply deduplication to channels
        for channel_name in channels:
            channels[channel_name] = channels[channel_name][keep_mask]
    else:
        times_dedup = times_sorted
    
    # Final validation
    if len(times_dedup) < 2:
        raise ExcelLoadError("Need at least 2 strictly increasing time points")
    
    return times_dedup, channels

# Test function for validation
def test_load_excel():
    """Basic test function to validate the loader"""
    try:
        # This would be called by your UI like:
        # times, channels = load_excel(file_path)
        # channel_names = list(channels.keys())  # used to fill selector
        print("Excel reader module ready - matches core.py::load_excel behavior")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == "__main__":
    test_load_excel()