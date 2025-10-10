"""
Unit tests for impedance parsing functions.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from io import StringIO

from src.impedance import (
    parse_impedance_file,
    _process_data_section,
    combine_impedance_measurements,
    get_impedance_summary,
    load_and_process_impedance
)


@pytest.fixture
def sample_impedance_data():
    """Sample impedance file content for testing."""
    return """#====================================================================================================
#
# [Test Date]       Tue Sep 30 2025
# [Test Time]       11:59:16
# [Test Frequency]  1000 (Hz)
# [Test Current]    10 (nA)
# [Test Cycles]     100
#
#  Array   Elec      Pin           Front End         Freq(Hz)   Curr(nA)   Cycles   Mag(kOhms)   Phase   
#====================================================================================================
   M1      chan1     Nip1:A-1-01   R02003-0492v7.1   1000       3          100      93.0         17      
   M1      chan2     Nip1:A-1-02   R02003-0492v7.1   1000       3          100      83.9         22      
   PMd     chan33    Nip1:A-2-01   R02003-0481v7.9   1000       3          100      106.1        18      

#====================================================================================================
#
# [Test Date]       Tue Sep 30 2025
# [Test Time]       12:00:46
# [Test Frequency]  1000 (Hz)
# [Test Current]    10 (nA)
# [Test Cycles]     100
#
#  Array   Elec      Pin           Front End         Freq(Hz)   Curr(nA)   Cycles   Mag(kOhms)   Phase   
#====================================================================================================
   M1      chan1     Nip1:A-1-01   R02003-0492v7.1   1000       3          100      92.4         18      
   M1      chan2     Nip1:A-1-02   R02003-0492v7.1   1000       3          100      84.0         22      
   PMd     chan33    Nip1:A-2-01   R02003-0481v7.9   1000       3          100      107.9        17      
"""


@pytest.fixture
def malformed_impedance_data():
    """Malformed impedance file content for testing error handling."""
    return """#====================================================================================================
# [Test Date]       Invalid Date
# [Test Time]       Invalid Time
#  Array   Elec      Pin           Front End         Freq(Hz)   Curr(nA)   Cycles   Mag(kOhms)   Phase   
#====================================================================================================
   M1      chan1     Nip1:A-1-01   R02003-0492v7.1   invalid    3          100      93.0         17      
   M1      chan2     Nip1:A-1-02   R02003-0492v7.1   1000       invalid    100      83.9         22      
"""


@pytest.fixture
def temp_impedance_file(sample_impedance_data):
    """Create a temporary impedance file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_impedance_data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


@pytest.fixture
def temp_malformed_file(malformed_impedance_data):
    """Create a temporary malformed impedance file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(malformed_impedance_data)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    Path(temp_path).unlink()


class TestParseImpedanceFile:
    """Tests for parse_impedance_file function."""
    
    def test_parse_valid_file(self, temp_impedance_file):
        """Test parsing a valid impedance file."""
        result = parse_impedance_file(temp_impedance_file)
        
        assert len(result) == 2  # Two measurement tables
        
        # Check first table
        df1 = result[0]
        assert df1.shape == (3, 9)  # 3 rows, 9 columns
        assert list(df1.columns) == ['Array', 'Elec', 'Pin', 'Front_End', 'Freq_Hz', 'Curr_nA', 'Cycles', 'Mag_kOhms', 'Phase']
        assert df1.attrs['Test Date'] == 'Tue Sep 30 2025'
        assert df1.attrs['Test Time'] == '11:59:16'
        assert df1.attrs['Test Frequency'] == '1000 (Hz)'
        
        # Check second table
        df2 = result[1]
        assert df2.shape == (3, 9)
        assert df2.attrs['Test Date'] == 'Tue Sep 30 2025'
        assert df2.attrs['Test Time'] == '12:00:46'
        
        # Check data types
        assert df1['Freq_Hz'].dtype in ['int64', 'float64']
        assert df1['Mag_kOhms'].dtype in ['int64', 'float64']
        assert df1['Phase'].dtype in ['int64', 'float64']
        
        # Check specific values
        assert df1.loc[0, 'Array'] == 'M1'
        assert df1.loc[0, 'Elec'] == 'chan1'
        assert df1.loc[0, 'Mag_kOhms'] == 93.0
        assert df1.loc[2, 'Array'] == 'PMd'
    
    def test_parse_nonexistent_file(self):
        """Test parsing a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            parse_impedance_file('/nonexistent/path/file.txt')
    
    def test_parse_empty_file(self):
        """Test parsing an empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            result = parse_impedance_file(temp_path)
            assert result == []
        finally:
            Path(temp_path).unlink()
    
    def test_parse_file_with_malformed_data(self, temp_malformed_file):
        """Test parsing a file with some malformed numeric data."""
        result = parse_impedance_file(temp_malformed_file)
        
        assert len(result) == 1
        df = result[0]
        
        # Should still parse but with NaN values for invalid numeric data
        assert pd.isna(df.loc[0, 'Freq_Hz'])  # 'invalid' should become NaN
        assert pd.isna(df.loc[1, 'Curr_nA'])  # 'invalid' should become NaN
        assert df.loc[0, 'Mag_kOhms'] == 93.0  # Valid numeric should parse correctly


class TestProcessDataSection:
    """Tests for _process_data_section helper function."""
    
    def test_process_valid_data_section(self):
        """Test processing valid data lines."""
        data_lines = [
            'M1      chan1     Nip1:A-1-01   R02003-0492v7.1   1000       3          100      93.0         17',
            'M1      chan2     Nip1:A-1-02   R02003-0492v7.1   1000       3          100      83.9         22'
        ]
        metadata = {'Test Date': 'Tue Sep 30 2025', 'Test Time': '11:59:16'}
        
        result = _process_data_section(data_lines, metadata)
        
        assert result is not None
        assert result.shape == (2, 9)
        assert result.attrs['Test Date'] == 'Tue Sep 30 2025'
        assert result.loc[0, 'Elec'] == 'chan1'
        assert result.loc[0, 'Mag_kOhms'] == 93.0
    
    def test_process_empty_data_section(self):
        """Test processing empty data section."""
        result = _process_data_section([], {'Test Date': 'test'})
        assert result is None
    
    def test_process_malformed_data_section(self):
        """Test processing malformed data lines."""
        data_lines = ['invalid data line with too few columns']
        metadata = {'Test Date': 'test'}
        
        # Should handle gracefully and return None or DataFrame with NaN values
        result = _process_data_section(data_lines, metadata)
        # The exact behavior depends on implementation, but should not crash


class TestCombineImpedanceMeasurements:
    """Tests for combine_impedance_measurements function."""
    
    def test_combine_valid_dataframes(self):
        """Test combining multiple valid DataFrames."""
        # Create sample DataFrames
        df1 = pd.DataFrame({
            'Array': ['M1', 'M1'],
            'Elec': ['chan1', 'chan2'],
            'Mag_kOhms': [93.0, 83.9],
            'Phase': [17, 22]
        })
        df1.attrs = {'Test Date': 'Tue Sep 30 2025', 'Test Time': '11:59:16'}
        
        df2 = pd.DataFrame({
            'Array': ['M1', 'M1'],
            'Elec': ['chan1', 'chan2'],
            'Mag_kOhms': [92.4, 84.0],
            'Phase': [18, 22]
        })
        df2.attrs = {'Test Date': 'Tue Sep 30 2025', 'Test Time': '12:00:46'}
        
        result = combine_impedance_measurements([df1, df2])
        
        assert result.shape == (4, 7)  # 4 rows, original columns + 3 new ones
        assert 'measurement_num' in result.columns
        assert 'test_date' in result.columns
        assert 'test_time' in result.columns
        
        # Check measurement numbers
        assert (result['measurement_num'] == [1, 1, 2, 2]).all()
        
        # Check metadata propagation
        assert all(result['test_date'] == 'Tue Sep 30 2025')
        assert (result['test_time'] == ['11:59:16', '11:59:16', '12:00:46', '12:00:46']).all()
    
    def test_combine_empty_list(self):
        """Test combining an empty list of DataFrames."""
        result = combine_impedance_measurements([])
        assert result.empty
        assert isinstance(result, pd.DataFrame)


class TestGetImpedanceSummary:
    """Tests for get_impedance_summary function."""
    
    def test_summary_valid_dataframe(self):
        """Test generating summary for valid DataFrame."""
        df = pd.DataFrame({
            'Array': ['M1', 'M1', 'PMd', 'PMd'],
            'Elec': ['chan1', 'chan2', 'chan33', 'chan34'],
            'Mag_kOhms': [93.0, 83.9, 106.1, 120.5],
            'Phase': [17, 22, 18, 15]
        })
        
        result = get_impedance_summary(df)
        
        assert result['shape'] == (4, 4)
        assert set(result['arrays']) == {'M1', 'PMd'}
        assert result['impedance_range'] == (83.9, 120.5)
        assert result['phase_range'] == (15, 22)
        assert isinstance(result['array_summary'], pd.DataFrame)
        assert 'M1' in result['array_summary'].index
        assert 'PMd' in result['array_summary'].index
    
    def test_summary_empty_dataframe(self):
        """Test generating summary for empty DataFrame."""
        df = pd.DataFrame()
        
        result = get_impedance_summary(df)
        
        assert result['shape'] == (0, 0)
        assert result['arrays'] == []
        assert result['impedance_range'] == (None, None)
        assert result['phase_range'] == (None, None)
        assert result['array_summary'].empty


class TestLoadAndProcessImpedance:
    """Tests for load_and_process_impedance convenience function."""
    
    def test_load_and_process_valid_file(self, temp_impedance_file):
        """Test the convenience function with a valid file."""
        combined_df = load_and_process_impedance(temp_impedance_file)
        
        # Check returned value
        assert not combined_df.empty
        assert isinstance(combined_df, pd.DataFrame)
        
        # Check that combined DataFrame has additional columns
        assert 'measurement_num' in combined_df.columns
        assert 'test_date' in combined_df.columns
        assert 'test_time' in combined_df.columns
        
        # Check that we have data from multiple measurements
        assert combined_df['measurement_num'].nunique() == 2  # Should have 2 measurements
        assert combined_df.shape[0] == 6  # 3 rows per measurement * 2 measurements
    
    def test_load_and_process_nonexistent_file(self):
        """Test the convenience function with a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_and_process_impedance('/nonexistent/path/file.txt')


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self, temp_impedance_file):
        """Test the complete workflow from file to analysis."""
        # Parse file
        impedance_dfs = parse_impedance_file(temp_impedance_file)
        
        # Combine measurements
        combined_df = combine_impedance_measurements(impedance_dfs)
        
        # Generate summary
        summary = get_impedance_summary(combined_df)
        
        # Verify the workflow worked correctly
        assert len(impedance_dfs) == 2
        assert combined_df.shape[0] == 6  # 3 rows per measurement * 2 measurements
        assert summary['shape'] == combined_df.shape  # Summary shape should match combined DataFrame
        
        # Verify data integrity through the workflow
        original_arrays = set()
        for df in impedance_dfs:
            original_arrays.update(df['Array'].unique())
        
        assert set(summary['arrays']) == original_arrays
        assert set(combined_df['Array'].unique()) == original_arrays


if __name__ == '__main__':
    pytest.main([__file__])