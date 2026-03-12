import numpy as np
import pandas as pd

from src.stim_analysis import (
    compute_channel_coincidence,
    compute_stim_change,
    compute_stim_response,
    compute_variance_ratio,
    default_pre_post_epochs,
    extract_neural_activity,
    format_stim_channel_label,
    prepare_electrode_response_table,
    select_nonstim_trials,
    select_passing_channels,
    summarize_stim_response,
    with_formatted_stim_labels,
    zscore_against_nonstim,
)


def _make_neural_data() -> pd.DataFrame:
    trial_ids = [1, 2]
    times = [
        pd.to_timedelta("-100ms"),
        pd.to_timedelta("0ms"),
        pd.to_timedelta("250ms"),
    ]
    stim_channels_by_trial = {1: frozenset(), 2: frozenset({"PMd.chan010"})}
    states = ["pre", "stim", "post"]

    tuples = [
        (trial_id, time, False, "success", 0, state, stim_channels_by_trial[trial_id])
        for trial_id in trial_ids
        for time, state in zip(times, states)
    ]
    index = pd.MultiIndex.from_tuples(
        tuples,
        names=[
            "trial_id",
            "time",
            "stim trial",
            "result",
            "target direction",
            "state",
            "stimulated channel",
        ],
    )
    data = np.arange(len(index) * 2, dtype=float).reshape(len(index), 2)
    return pd.DataFrame(data, index=index, columns=["M1.chan001", "PMd.chan010"])


def test_extract_neural_activity_and_select_nonstim() -> None:
    neural = _make_neural_data()
    trialframe = pd.concat({"neural activity": neural}, axis=1)

    extracted = extract_neural_activity(trialframe)
    nonstim = select_nonstim_trials(extracted)

    assert extracted.columns.name == "recorded channel"
    assert "stim trial" not in nonstim.index.names


def test_select_passing_channels_and_coincidence() -> None:
    neural = _make_neural_data()
    channel_stats = pd.DataFrame({"pass": [True, False]}, index=neural.columns)

    selected = select_passing_channels(neural, channel_stats, fill_value=0.0)
    coincidence = compute_channel_coincidence(selected)

    assert list(selected.columns) == ["M1.chan001"]
    assert coincidence.shape == (1, 1)
    assert coincidence.iloc[0, 0] >= 0


def test_compute_stim_response_change_and_normalization() -> None:
    neural = _make_neural_data()
    channels = ["M1.chan001", "PMd.chan010"]

    response = compute_stim_response(
        neural,
        channels=channels,
        epochs=default_pre_post_epochs(),
        bin_size_seconds=1e-3,
        group_levels=("trial_id", "stimulated channel", "phase"),
    )
    change = compute_stim_change(response)
    normalized = zscore_against_nonstim(change)

    assert "phase" in response.index.names
    assert "stimulated channel" in normalized.index.names
    assert all(k != frozenset() for k in normalized.index.get_level_values("stimulated channel"))


def test_summarize_and_format_labels() -> None:
    idx = pd.MultiIndex.from_product(
        [[1], [frozenset({"PMd.chan010"})]],
        names=["trial_id", "stimulated channel"],
    )
    df = pd.DataFrame([[1.0, 2.0]], index=idx, columns=["M1.chan001", "PMd.chan010"])
    df.columns.name = "recorded channel"

    summary = summarize_stim_response(df)
    formatted = with_formatted_stim_labels(summary)

    assert "stimulation response (z-score)" in summary.columns
    assert formatted["stimulated channel"].iloc[0].strip() != ""
    assert "PMd" in format_stim_channel_label(frozenset({"PMd.chan010"}))


def test_variance_ratio_and_electrode_table() -> None:
    index = pd.MultiIndex.from_product(
        [[frozenset(), frozenset({"PMd.chan010"})], ["pre-stim", "post-stim"], [1, 2, 3, 4, 5]],
        names=["stimulated channel", "phase", "trial_id"],
    )
    projected = pd.DataFrame(
        {
            0: np.linspace(0.0, 1.0, len(index)),
            1: np.linspace(1.0, 2.0, len(index)),
        },
        index=index,
    )
    ratio = compute_variance_ratio(projected, min_group_size=3)

    stim_idx = pd.MultiIndex.from_product(
        [[1], [frozenset({"PMd.chan010"})]],
        names=["trial_id", "stimulated channel"],
    )
    stim_df = pd.DataFrame([[0.5, -0.1]], index=stim_idx, columns=["M1.chan001", "PMd.chan010"])
    electrode_map = pd.DataFrame(
        {"x": [1, 2], "y": [3, 4]},
        index=pd.Index(["M1.chan001", "PMd.chan010"], name="recorded channel"),
    )
    electrode_table = prepare_electrode_response_table(stim_df, electrode_map)

    assert ratio.name == "variance ratio"
    assert "x" in electrode_table.columns
    assert "y" in electrode_table.columns