import pandas as pd
from pyns.nsfile import NSFile
from pyns.nsentity import EntityType, EventEntity
import numpy as np
import pandas as pd
import re
import smile_extract
import logging
from pathlib import Path
logger = logging.getLogger(__name__)

def read_map(map_path: Path) -> pd.DataFrame:
    """Read a Trellis-style .map file into a pandas DataFrame.
    
    Parameters
    ----------
    map_path : Path
        Path to the .map file
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'hw_address', 'channel', 'x', 'y'
        Indexed by channel label (e.g., 'M1.chan004')
    """
    # Read the file, skipping comment lines (those starting with #)
    df = pd.read_csv(
        map_path,
        sep=';',
        comment='#',
        names=['hw_address', 'channel', 'coords'],
        skipinitialspace=True,  # Remove leading whitespace after delimiter
        dtype=str,  # Read all columns as strings to avoid auto-conversion
    )
    
    # Strip any remaining whitespace from string columns
    df = df.apply(lambda col: col.str.strip())
    
    # Split the coords column (e.g., "18.1") into x and y
    coords_split = df['coords'].str.split('.', expand=True)
    df['x'] = coords_split[0].astype(int)
    df['y'] = coords_split[1].astype(int)
    
    # Drop the coords column and set channel as index
    df = df.drop(columns=['coords']).set_index('channel')
    
    return df

def get_event_times(event_entity: EventEntity, digline_idx: int, label: str):
    return (
        pd.Series(
            [
                pd.to_timedelta(event_entity.get_event_data(index)[0], unit='s')
                for index in range(event_entity.item_count)
                if event_entity.get_event_data(index)[1][digline_idx] > 0
            ],
        )
        .rename(label)
        .rename_axis('event id')
    )

def get_trial_starts(nsfile: NSFile) -> pd.Series:
    entities = [e for e in nsfile.get_entities(entity_type=EntityType.event)]
    trial_start_events = entities[0]
    stim_events = entities[1]

    digline = {'trial start': 1, 'stim': 2}

    trial_starts = (
        get_event_times(trial_start_events, digline['trial start'], 'trial start')
        .reset_index(drop=True)
        .rename_axis('trial_id')
        .rename(index=lambda x: x + 1)
    )
    return trial_starts

def process_neural_data(nsfile: NSFile) -> tuple[pd.DataFrame, pd.DataFrame]:
    entities = [e for e in nsfile.get_entities(EntityType.segment) if e.item_count > 0] # filter out empty entities
    neural_entities = [e for e in entities if len(e.label) < 8] # filter out non-neural entities
    stim_entities = [e for e in entities if len(e.label) >= 8] # filter out non-stimulation entities

    if len(neural_entities) == 0:
        logger.warning("No neural entities with spikes found in the provided NSFile.")
    if len(stim_entities) == 0:
        logger.warning("No stimulation entities found in the provided NSFile.")

    def get_entity_event_times(entity):
        return np.array([entity.get_time_by_index(idx) for idx in range(entity.item_count)])

    def relabel_channels(label: str) -> str | int:
        m = re.match(r'elec\s*(\d+)$', label, re.IGNORECASE)
        if not m:
            return label
        num = int(m.group(1))
        if num > 5120:
            num -= 5120
        if num > 128:
            num -= 32

        if num <= 32 or num > 96:
            array = 'M1'
        elif num > 32 and num <= 96:
            array = 'PMd'
        else:
            raise ValueError(f"Unexpected channel number {num} in label '{label}'")
        return f"{array}.chan{num:03d}"

    def compose_event_table(entity_list: list) -> pd.DataFrame:
        return pd.DataFrame(
            [(relabel_channels(e.label), 0, pd.to_timedelta(t, unit='s')) for e in entity_list for t in get_entity_event_times(e)],
            columns=['channel', 'unit', 'timestamp'],
        ).rename_axis('snippet_id',axis=0)

    spike_times = compose_event_table(neural_entities)
    stim_times = compose_event_table(stim_entities)

    return spike_times, stim_times

def get_trial_id(timestamps: pd.Series, trial_start_times: pd.Series) -> pd.Series:
    """Map timestamps to trial IDs based on trial start times.
    
    Parameters
    ----------
    timestamps : pd.Series
        Series of timestamps (timedelta)
    trial_start_times : pd.Series
        Series of trial start times (timedelta)
    
    Returns
    -------
    pd.Series
        Series of trial IDs (int) with the same index as timestamps
    """
    assert not trial_start_times.empty, "trial_start_times is empty, cannot get trial ID."
    
    # Use searchsorted for efficient O(n log m) lookup instead of O(n*m)
    # Convert to numpy arrays for searchsorted
    timestamps_array = timestamps.values.astype('int64')
    trial_starts_array = trial_start_times.values.astype('int64')
    
    # searchsorted finds the insertion point, which gives us the trial ID
    # side='right' means timestamps equal to a trial start get the next trial ID
    trial_ids = np.searchsorted(trial_starts_array, timestamps_array, side='right')
    
    return pd.Series(trial_ids, index=timestamps.index)

def trialize_timestamps(timestamps: pd.DataFrame, trial_starts: pd.Series) -> pd.DataFrame:
    """Assign trial IDs to spike times based on trial start times.
    Also subtract trial start time from spike timestamps to get offset within trial.
    
    Parameters
    ----------
    spike_times : pd.DataFrame
        DataFrame with columns ['channel', 'unit', 'timestamp'] and index 'snippet_id'
    trial_starts : pd.Series
        Series of trial start times (timedelta) indexed by trial ID (1-based)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with additional column 'trial_id' and index ['trial_id', 'snippet_id']
    """
    trialized_timestamps = (
        timestamps
        .assign(trial_id=get_trial_id(timestamps['timestamp'], trial_starts))
        .reset_index()
        .set_index(['trial_id','snippet_id'])
        .sort_index()
        .loc[1:,:]  # only keep trials with trial_id (i.e. drop pre-trial spikes)
        .assign(timestamp=lambda df: df['timestamp'] - df.index.get_level_values('trial_id').map(trial_starts))
    )
    return trialized_timestamps

def compose_ripple_smile(nsfile: NSFile, smile_data: list, bin_size: str = '1ms') -> pd.DataFrame:
    stim_trials = (
        smile_extract.get_smile_meta(smile_data)
        ['trial name']
        .apply(lambda x: 'stim' in x.lower())
        .rename('stim trial')
    )
    smile_states = smile_extract.concat_trial_func_results(
        smile_extract.get_trial_states,
        smile_data,
        bin_size=bin_size,
    )

    spike_times, stim_times = process_neural_data(nsfile)
    trial_starts = get_trial_starts(nsfile)

    stim_channels = (
        stim_times
        .pipe(trialize_timestamps, trial_starts)
        .groupby('trial_id')
        .first()
        ['channel']
        .rename('stimulated channel')
        .reindex(stim_trials.index, fill_value=-1)
    )
    
    trialframe = (
        spike_times
        .pipe(trialize_timestamps, trial_starts)
        .pipe(smile_extract.bin_spikes, bin_size=bin_size)
        .droplevel(level='unit',axis='columns')
        .join(smile_states.rename('state'), how='right')
        .reset_index(level='time')
        .assign(**{'stim trial': stim_trials, 'stimulated channel': stim_channels})
        .set_index(['time','stim trial','state','stimulated channel'], append=True)
        .rename_axis('recorded channel',axis=1)
        .sort_index(axis=1)
    )

    return trialframe

def get_channel_stats(spike_mat: pd.DataFrame, min_firing_rate: float=1.0, max_fano: float=8.0, max_coincidence: float=0.2) -> pd.DataFrame:
    percent_coincidence = (
        (spike_mat.T @ spike_mat)
        / spike_mat.sum(axis=0)
    ) # type: ignore
    coincidence_triu = pd.DataFrame(
        np.triu(percent_coincidence, k=1),
        index=percent_coincidence.index,
        columns=percent_coincidence.columns
    ) # type: ignore
    
    channel_stats = (
        spike_mat
        .groupby('trial_id')
        .resample('500ms',level='time')
        .sum()
        .stack()
        .groupby('recorded channel')
        .agg(['mean','var'])
        .assign(**{
            'mean firing rate': lambda df: df['mean'] / 0.5, # convert to Hz from 500ms bins
            'fano factor': lambda df: df['var'] / df['mean'],
            'max coincidence': lambda df: coincidence_triu.max(axis=0).reindex(df.index).fillna(0),
            'pass': lambda df: (df['mean firing rate'] > min_firing_rate) & (df['fano factor'] < max_fano) & (df['max coincidence'] < max_coincidence),
        })
    )
    
    return channel_stats