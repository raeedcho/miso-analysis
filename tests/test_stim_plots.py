import pandas as pd

from src.stim_plots import (
    build_electrode_heatmap,
    build_raster_chart,
    build_response_heatmap,
    build_variance_ratio_chart,
)


def test_build_raster_chart() -> None:
    index = pd.MultiIndex.from_product(
        [[1, 2], [pd.to_timedelta("0ms"), pd.to_timedelta("1ms")], [frozenset({"PMd.chan010"})]],
        names=["trial_id", "time", "stimulated channel"],
    )
    raster_df = pd.DataFrame({"M1.chan001": [0, 1, 0, 2], "target direction": [0, 0, 180, 180]}, index=index)

    chart = build_raster_chart(raster_df, rec_channel="M1.chan001", stim_channel=frozenset({"PMd.chan010"}))
    chart_dict = chart.to_dict()

    assert chart_dict["mark"]["type"] == "tick"
    assert chart_dict["encoding"]["x"]["field"] == "time"
    assert chart_dict["encoding"]["y"]["field"] == "trial_id"


def test_build_response_heatmap() -> None:
    df = pd.DataFrame(
        {
            "stimulated channel": ["A", "B"],
            "recorded channel": ["r1", "r2"],
            "stimulation response (z-score)": [0.1, -0.3],
        }
    )
    chart = build_response_heatmap(df)
    chart_dict = chart.to_dict()

    assert chart_dict["mark"]["type"] == "rect"
    assert chart_dict["encoding"]["color"]["field"] == "stimulation response (z-score)"


def test_build_variance_ratio_chart() -> None:
    ratio = pd.Series([1.2, 0.8], index=pd.Index(["A", "B"], name="stimulated channel"), name="variance ratio")
    chart = build_variance_ratio_chart(ratio)
    chart_dict = chart.to_dict()

    assert chart_dict["mark"]["type"] == "bar"
    assert chart_dict["encoding"]["x"]["field"] == "variance ratio"


def test_build_electrode_heatmap() -> None:
    df = pd.DataFrame(
        {
            "recorded channel": ["M1.chan001", "PMd.chan010"],
            "stimulated channel": ["A", "A"],
            "x": [1, 2],
            "y": [3, 4],
            "stimulation response (z-score)": [0.2, -0.4],
        }
    )
    chart = build_electrode_heatmap(df)
    chart_dict = chart.to_dict()

    assert chart_dict["mark"]["type"] == "rect"
    assert chart_dict["encoding"]["row"]["field"] == "stimulated channel"