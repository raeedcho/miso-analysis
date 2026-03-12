from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from trialframe import get_epoch_data


def extract_neural_activity(trialframe: pd.DataFrame) -> pd.DataFrame:
    """Return neural activity as a channel-labeled DataFrame.

    Parameters
    ----------
    trialframe : pd.DataFrame
        Session table with a top-level signal column containing "neural activity".
    """
    return (
        pd.DataFrame(trialframe["neural activity"])
        .rename_axis("recorded channel", axis=1)
        .sort_index(axis=1)
    )


def select_nonstim_trials(
    neural_data: pd.DataFrame,
    stim_trial_level: str = "stim trial",
) -> pd.DataFrame:
    """Select non-stimulation trials from neural_data."""
    return neural_data.xs(level=stim_trial_level, key=False)


def select_passing_channels(
    neural_data: pd.DataFrame,
    channel_stats: pd.DataFrame,
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Filter neural data to channels where channel_stats['pass'] is True."""
    passing_channels = channel_stats.index[channel_stats["pass"]]
    selected = neural_data.loc[:, passing_channels]
    if fill_value is not None:
        selected = selected.fillna(fill_value)
    return selected


def compute_channel_coincidence(
    spike_mat: pd.DataFrame,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """Compute pairwise coincidence matrix used in notebook QC plots."""
    spike_mat_filled = spike_mat.fillna(fill_value)
    denominator = spike_mat_filled.sum(axis=0).replace(0, np.nan)
    coincidence = (spike_mat_filled.T @ spike_mat_filled).div(denominator, axis=1)
    return coincidence.fillna(0.0)


def default_pre_post_epochs() -> dict[str, tuple[str, slice]]:
    """Default stimulation epochs used across stimulation notebooks."""
    return {
        "pre-stim": (
            "stim",
            slice(pd.to_timedelta("-200ms"), pd.to_timedelta("-50ms")),
        ),
        "post-stim": (
            "stim",
            slice(pd.to_timedelta("200ms"), pd.to_timedelta("350ms")),
        ),
    }


def peri_stim_epoch(
    start: str = "-250ms",
    stop: str = "350ms",
    event: str = "stim",
) -> dict[str, tuple[str, slice]]:
    """Construct a single peri-stim epoch dictionary for raster extraction."""
    return {
        "peri-stim": (
            event,
            slice(pd.to_timedelta(start), pd.to_timedelta(stop)),
        ),
    }


def compute_stim_response(
    neural_data: pd.DataFrame,
    channels: Sequence[str],
    epochs: dict[str, tuple[str, slice]] | None = None,
    bin_size_seconds: float = 1e-3,
    group_levels: Sequence[str] = ("trial_id", "stimulated channel", "phase"),
    result_key: str | None = None,
    fill_value: float | None = None,
) -> pd.DataFrame:
    """Compute mean stimulation responses by epoch and condition."""
    if epochs is None:
        epochs = default_pre_post_epochs()

    selected = neural_data.loc[:, list(channels)]
    if result_key is not None:
        selected = selected.xs(level="result", key=result_key)
    if fill_value is not None:
        selected = selected.fillna(fill_value)

    return (
        selected
        .pipe(get_epoch_data, epochs=epochs)
        .div(bin_size_seconds)
        .groupby(list(group_levels), observed=True)
        .mean()
    )


def compute_stim_change(
    stim_response: pd.DataFrame,
    pre_phase: str = "pre-stim",
    post_phase: str = "post-stim",
    phase_level: str = "phase",
) -> pd.DataFrame:
    """Compute post-pre change from an epoch response table."""
    post_stim = stim_response.xs(post_phase, level=phase_level)
    pre_stim = stim_response.xs(pre_phase, level=phase_level)
    return post_stim - pre_stim


def zscore_against_nonstim(
    stim_change: pd.DataFrame,
    stim_level: str = "stimulated channel",
    nonstim_key: frozenset[str] = frozenset(),
) -> pd.DataFrame:
    """Normalize stimulation changes against non-stimulated trials."""
    nonstim = stim_change.xs(nonstim_key, level=stim_level)
    denom = nonstim.std().replace(0, np.nan)
    normalized = (
        stim_change
        .loc[stim_change.index.get_level_values(stim_level) != nonstim_key]
        .sub(nonstim.mean())
        .div(denom)
    )
    return normalized


def summarize_stim_response(
    norm_stim_response: pd.DataFrame,
    value_name: str = "stimulation response (z-score)",
) -> pd.DataFrame:
    """Stack and aggregate channel responses for plotting."""
    return (
        norm_stim_response
        .stack()
        .to_frame(value_name)
        .groupby(["stimulated channel", "recorded channel"], observed=True)
        .mean()
        .reset_index()
    )


def format_stim_channel_label(channel_set: frozenset[str]) -> str:
    """Render frozenset channel labels to the notebook-friendly display format."""
    return str(set(channel_set)).strip("{}").replace("'", "   ").replace(", ", " ").replace("M1", "  M1")


def with_formatted_stim_labels(
    df: pd.DataFrame,
    column: str = "stimulated channel",
) -> pd.DataFrame:
    """Format stimulated-channel labels for faceting and axis display."""
    return df.assign(**{column: df[column].map(format_stim_channel_label)})


def compute_variance_ratio(
    projected_data: pd.DataFrame,
    group_levels: Sequence[str] = ("stimulated channel", "phase"),
    phase_level: str = "phase",
    pre_phase: str = "pre-stim",
    post_phase: str = "post-stim",
    min_group_size: int = 10,
) -> pd.Series:
    """Compute post/pre variance ratio from latent activity tables."""
    phase_var = (
        projected_data
        .groupby(list(group_levels), observed=True)
        .filter(lambda x: x.shape[0] >= min_group_size)
        .groupby(list(group_levels), observed=True)
        .var()
        .sum(axis=1)
        .unstack(level=phase_level)
    )
    return (phase_var[post_phase] / phase_var[pre_phase]).rename("variance ratio")


def prepare_electrode_response_table(
    norm_stim_response: pd.DataFrame,
    electrode_map: pd.DataFrame,
    value_name: str = "stimulation response (z-score)",
) -> pd.DataFrame:
    """Join response tables with electrode coordinates for array heatmaps."""
    return (
        norm_stim_response
        .stack()
        .to_frame(value_name)
        .groupby(["recorded channel", "stimulated channel"], observed=True)
        .mean()
        .join(electrode_map.rename_axis("recorded channel"), how="left")
        .reset_index()
    )