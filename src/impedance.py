"""
Impedance data parsing and analysis functions.

This module provides functions to parse impedance measurement files from
neural recording systems and convert them into pandas DataFrames for analysis.
"""

import pandas as pd
import re
from io import StringIO
from pathlib import Path
from typing import List, Dict, Any, Optional, Union


def parse_impedance_file(filepath: Union[str, Path]) -> List[pd.DataFrame]:
    """
    Parse impedance measurement file containing multiple test tables.
    
    The file format typically contains multiple measurement sessions with metadata
    and tabular data for each electrode's impedance measurements.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the impedance file
    
    Returns
    -------
    List[pd.DataFrame]
        List of DataFrames, one for each measurement table found in the file.
        Each DataFrame contains columns:
        - Array: Array identifier (e.g., 'M1', 'PMd')
        - Elec: Electrode/channel identifier (e.g., 'chan1', 'chan2')
        - Pin: Physical pin mapping
        - Front_End: Front-end device information
        - Freq_Hz: Test frequency in Hz
        - Curr_nA: Test current in nA
        - Cycles: Number of test cycles
        - Mag_kOhms: Impedance magnitude in kΩ
        - Phase: Phase angle in degrees
        
        Metadata from each measurement is stored in the DataFrame's .attrs attribute.
    
    Raises
    ------
    FileNotFoundError
        If the specified file does not exist
    IOError
        If the file cannot be read
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Impedance file not found: {filepath}")
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except IOError as e:
        raise IOError(f"Error reading impedance file {filepath}: {e}")
    
    # Split content into sections by separator lines
    sections = re.split(r'#=+', content)
    dataframes = []
    
    i = 0
    while i < len(sections):
        if not sections[i].strip():
            i += 1
            continue
            
        # Look for metadata section (contains # [Test Date] etc.)
        metadata_section = sections[i].strip()
        data_section = ""
        
        # Check if next section contains data
        if i + 1 < len(sections):
            potential_data = sections[i + 1].strip()
            if potential_data and not potential_data.startswith('#'):
                data_section = potential_data
                i += 2  # Skip both sections
            else:
                i += 1  # Only skip metadata section
        else:
            i += 1
        
        # Parse metadata from this section
        metadata = {}
        data_lines = []
        
        # Process metadata section
        for line in metadata_section.split('\n'):
            line = line.strip()
            if line.startswith('# ['):
                match = re.match(r'# \[([^\]]+)\]\s+(.+)', line)
                if match:
                    key = match.group(1)
                    value = match.group(2)
                    metadata[key] = value
        
        # Process data section
        if data_section:
            for line in data_section.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    data_lines.append(line)
        
        # Create DataFrame if we have data
        if data_lines:
            df = _process_data_section(data_lines, metadata)
            if df is not None:
                dataframes.append(df)
    
    return dataframes


def _process_data_section(data_lines: List[str], metadata: Dict[str, str]) -> Optional[pd.DataFrame]:
    """
    Helper function to process a section of data lines into a DataFrame.
    
    Parameters
    ----------
    data_lines : List[str]
        Lines of text containing the measurement data
    metadata : Dict[str, str]
        Metadata dictionary to attach to the DataFrame
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame with parsed impedance data, or None if parsing fails
    """
    if not data_lines:
        return None
    
    # Create StringIO object for pandas to read
    data_text = '\n'.join(data_lines)
    
    try:
        # Use pandas to read the whitespace-separated data
        df = pd.read_csv(
            StringIO(data_text), 
            sep=r'\s+',  # Multiple whitespace as separator
            header=None,
            names=['Array', 'Elec', 'Pin', 'Front_End', 'Freq_Hz', 'Curr_nA', 'Cycles', 'Mag_kOhms', 'Phase']
        )
        
        # Convert numeric columns
        numeric_cols = ['Freq_Hz', 'Curr_nA', 'Cycles', 'Mag_kOhms', 'Phase']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Add metadata as attributes
        for key, value in metadata.items():
            df.attrs[key] = value
        
        return df
        
    except Exception as e:
        print(f"Error processing data section: {e}")
        return None


def combine_impedance_measurements(impedance_dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine multiple impedance measurement DataFrames into a single DataFrame.
    
    Parameters
    ----------
    impedance_dfs : List[pd.DataFrame]
        List of impedance DataFrames from parse_impedance_file()
    
    Returns
    -------
    pd.DataFrame
        Combined DataFrame with additional columns:
        - measurement_num: Sequential measurement number (1, 2, ...)
        - test_date: Test date from metadata
        - test_time: Test time from metadata
    """
    if not impedance_dfs:
        return pd.DataFrame()
    
    combined_impedance = []
    for i, df in enumerate(impedance_dfs):
        df_copy = df.copy()
        df_copy['measurement_num'] = i + 1
        df_copy['test_date'] = df.attrs.get('Test Date', '')
        df_copy['test_time'] = df.attrs.get('Test Time', '')
        combined_impedance.append(df_copy)
    
    # Concatenate all measurements
    return pd.concat(combined_impedance, ignore_index=True)


def get_impedance_summary(all_impedance: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate summary statistics for impedance measurements.
    
    Parameters
    ----------
    all_impedance : pd.DataFrame
        Combined impedance DataFrame from combine_impedance_measurements()
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing summary statistics:
        - shape: DataFrame shape
        - arrays: Unique array identifiers
        - impedance_range: Min and max impedance values
        - phase_range: Min and max phase values
        - array_summary: Summary statistics by array
    """
    if all_impedance.empty:
        return {
            'shape': (0, 0),
            'arrays': [],
            'impedance_range': (None, None),
            'phase_range': (None, None),
            'array_summary': pd.DataFrame()
        }
    
    return {
        'shape': all_impedance.shape,
        'arrays': all_impedance['Array'].unique().tolist(),
        'impedance_range': (all_impedance['Mag_kOhms'].min(), all_impedance['Mag_kOhms'].max()),
        'phase_range': (all_impedance['Phase'].min(), all_impedance['Phase'].max()),
        'array_summary': all_impedance.groupby('Array')['Mag_kOhms'].describe()
    }


def load_and_process_impedance(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Convenience function to load and combine impedance data in one call.
    
    Parameters
    ----------
    filepath : str or Path
        Path to the impedance file
    
    Returns
    -------
    pd.DataFrame
        Combined measurements DataFrame with additional columns:
        - measurement_num: Sequential measurement number (1, 2, ...)
        - test_date: Test date from metadata
        - test_time: Test time from metadata
    """
    impedance_dfs = parse_impedance_file(filepath)
    combined_df = combine_impedance_measurements(impedance_dfs)
    
    return combined_df
