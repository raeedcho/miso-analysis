import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from src.process_ripple import get_trial_id, read_map


class TestGetTrialId:
    """Test suite for get_trial_id function."""
    
    def test_basic_trial_assignment(self):
        """Test basic trial ID assignment with simple timestamps."""
        # Create trial start times at 1s, 2s, 3s
        trial_starts = pd.Series([
            pd.to_timedelta(1, unit='s'),
            pd.to_timedelta(2, unit='s'),
            pd.to_timedelta(3, unit='s'),
        ])
        
        # Create timestamps before, between, and after trials
        timestamps = pd.Series([
            pd.to_timedelta(0.5, unit='s'),  # Before first trial -> 0
            pd.to_timedelta(1.5, unit='s'),  # Between trial 0 and 1 -> 1
            pd.to_timedelta(2.5, unit='s'),  # Between trial 1 and 2 -> 2
            pd.to_timedelta(3.5, unit='s'),  # After trial 2 -> 3
        ])
        
        result = get_trial_id(timestamps, trial_starts)
        expected = pd.Series([0, 1, 2, 3])
        
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_timestamps_before_first_trial(self):
        """Test that timestamps before first trial get ID 0."""
        trial_starts = pd.Series([
            pd.to_timedelta(5, unit='s'),
            pd.to_timedelta(10, unit='s'),
        ])
        
        timestamps = pd.Series([
            pd.to_timedelta(1, unit='s'),
            pd.to_timedelta(2, unit='s'),
            pd.to_timedelta(4.9, unit='s'),
        ])
        
        result = get_trial_id(timestamps, trial_starts)
        expected = pd.Series([0, 0, 0])
        
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_timestamps_at_exact_trial_start(self):
        """Test behavior when timestamp equals trial start time."""
        trial_starts = pd.Series([
            pd.to_timedelta(1, unit='s'),
            pd.to_timedelta(2, unit='s'),
            pd.to_timedelta(3, unit='s'),
        ])
        
        # Timestamps exactly at trial start times
        timestamps = pd.Series([
            pd.to_timedelta(1, unit='s'),
            pd.to_timedelta(2, unit='s'),
            pd.to_timedelta(3, unit='s'),
        ])
        
        result = get_trial_id(timestamps, trial_starts)
        # With side='right', exact matches get the next trial ID
        expected = pd.Series([1, 2, 3])
        
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_large_dataset_performance(self):
        """Test with a large dataset to verify efficiency."""
        # Create 100 trial starts
        trial_starts = pd.Series([
            pd.to_timedelta(i, unit='s') for i in range(1, 101)
        ])
        
        # Create 10000 random timestamps
        np.random.seed(42)
        random_times = np.random.uniform(0, 105, 10000)
        timestamps = pd.Series([pd.to_timedelta(t, unit='s') for t in sorted(random_times)])
        
        result = get_trial_id(timestamps, trial_starts)
        
        # Verify result has correct length
        assert len(result) == len(timestamps)
        
        # Verify all trial IDs are in valid range
        assert result.min() >= 0
        assert result.max() <= len(trial_starts)
        
        # Spot check a few values
        # Timestamp before first trial should be 0
        early_timestamps = timestamps[timestamps < trial_starts.iloc[0]]
        if len(early_timestamps) > 0:
            early_ids = result[timestamps < trial_starts.iloc[0]]
            assert (early_ids == 0).all()
    
    def test_preserve_index(self):
        """Test that the function preserves the original index."""
        trial_starts = pd.Series([
            pd.to_timedelta(1, unit='s'),
            pd.to_timedelta(2, unit='s'),
        ])
        
        timestamps = pd.Series(
            [
                pd.to_timedelta(0.5, unit='s'),
                pd.to_timedelta(1.5, unit='s'),
                pd.to_timedelta(2.5, unit='s'),
            ],
            index=[10, 20, 30]  # Custom index
        )
        
        result = get_trial_id(timestamps, trial_starts)
        
        # Check that index is preserved
        assert list(result.index) == [10, 20, 30]
        assert list(result.values) == [0, 1, 2]
    
    def test_empty_trial_starts_raises_error(self):
        """Test that empty trial_start_times raises an assertion error."""
        trial_starts = pd.Series([], dtype='timedelta64[ns]')
        timestamps = pd.Series([pd.to_timedelta(1, unit='s')])
        
        with pytest.raises(AssertionError, match="trial_start_times is empty"):
            get_trial_id(timestamps, trial_starts)
    
    def test_single_trial(self):
        """Test with only one trial start time."""
        trial_starts = pd.Series([pd.to_timedelta(5, unit='s')])
        
        timestamps = pd.Series([
            pd.to_timedelta(1, unit='s'),   # Before -> 0
            pd.to_timedelta(5, unit='s'),   # At start -> 1
            pd.to_timedelta(10, unit='s'),  # After -> 1
        ])
        
        result = get_trial_id(timestamps, trial_starts)
        expected = pd.Series([0, 1, 1])
        
        pd.testing.assert_series_equal(result, expected, check_names=False)
    
    def test_unsorted_timestamps(self):
        """Test that function works even if timestamps are not sorted."""
        trial_starts = pd.Series([
            pd.to_timedelta(1, unit='s'),
            pd.to_timedelta(2, unit='s'),
            pd.to_timedelta(3, unit='s'),
        ])
        
        # Unsorted timestamps
        timestamps = pd.Series([
            pd.to_timedelta(2.5, unit='s'),  # -> 2
            pd.to_timedelta(0.5, unit='s'),  # -> 0
            pd.to_timedelta(3.5, unit='s'),  # -> 3
            pd.to_timedelta(1.5, unit='s'),  # -> 1
        ])
        
        result = get_trial_id(timestamps, trial_starts)
        expected = pd.Series([2, 0, 3, 1])
        
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestReadMap:
    """Test suite for read_map function."""
    
    def test_read_map_basic(self):
        """Test reading a basic map file."""
        # Create a temporary map file
        map_content = """# Test map file
# Comment line
1.A.1.001; M1.chan001; 7.1
1.A.1.002; M1.chan002; 8.1
1.A.2.033; PMd.chan033; 18.5
1.B.1.097; M1.chan097; 8.5
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            f.write(map_content)
            temp_path = Path(f.name)
        
        try:
            result = read_map(temp_path)
            
            # Check that we have the right number of rows
            assert len(result) == 4
            
            # Check column names
            assert list(result.columns) == ['hw_address', 'x', 'y']
            
            # Check index name
            assert result.index.name == 'channel'
            
            # Check specific values
            assert result.loc['M1.chan001', 'hw_address'] == '1.A.1.001'
            assert result.loc['M1.chan001', 'x'] == 7
            assert result.loc['M1.chan001', 'y'] == 1
            
            assert result.loc['PMd.chan033', 'x'] == 18
            assert result.loc['PMd.chan033', 'y'] == 5
            
        finally:
            temp_path.unlink()
    
    def test_read_map_whitespace_handling(self):
        """Test that whitespace is properly handled."""
        map_content = """# Test map file
1.A.1.001;M1.chan001;7.1
  1.A.1.002  ;  M1.chan002  ;  8.1  
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            f.write(map_content)
            temp_path = Path(f.name)
        
        try:
            result = read_map(temp_path)
            
            # Check that we have the right number of rows
            assert len(result) == 2
            
            # Check that whitespace was properly stripped
            assert 'M1.chan001' in result.index
            assert 'M1.chan002' in result.index
            assert result.loc['M1.chan001', 'hw_address'] == '1.A.1.001'
            
        finally:
            temp_path.unlink()
    
    def test_read_map_coordinate_parsing(self):
        """Test that coordinates are correctly parsed into x and y."""
        map_content = """# Test map file
1.A.1.001; M1.chan001; 1.2
1.A.1.002; M1.chan002; 10.20
1.A.2.033; PMd.chan033; 5.15
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.map', delete=False) as f:
            f.write(map_content)
            temp_path = Path(f.name)
        
        try:
            result = read_map(temp_path)
            
            # Check coordinate parsing
            assert result.loc['M1.chan001', 'x'] == 1
            assert result.loc['M1.chan001', 'y'] == 2
            
            assert result.loc['M1.chan002', 'x'] == 10
            assert result.loc['M1.chan002', 'y'] == 20
            
            assert result.loc['PMd.chan033', 'x'] == 5
            assert result.loc['PMd.chan033', 'y'] == 15
            
            # Check that x and y are integers
            assert result['x'].dtype == np.int64
            assert result['y'].dtype == np.int64
            
        finally:
            temp_path.unlink()
    
    def test_read_map_real_file(self):
        """Test reading the actual map file from the repository."""
        map_path = Path(__file__).parent.parent / "data" / "sulley" / "sulley_SN_4566-004410.map"
        
        if not map_path.exists():
            pytest.skip(f"Map file not found at {map_path}")
        
        result = read_map(map_path)
        
        # Check that we have data
        assert len(result) > 0
        
        # Check column names
        assert list(result.columns) == ['hw_address', 'x', 'y']
        
        # Check that index is channel labels
        assert result.index.name == 'channel'
        
        # Check that all channels have the expected format
        assert all(result.index.str.match(r'^(M1|PMd)\.chan\d{3}$'))
        
        # Check data types
        assert result['x'].dtype == np.int64
        assert result['y'].dtype == np.int64
        
        # Check reasonable coordinate ranges (for an 8x8 or similar array)
        assert result['x'].min() >= 1
        assert result['y'].min() >= 1
        assert result['x'].max() <= 20
        assert result['y'].max() <= 20

