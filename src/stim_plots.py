from __future__ import annotations

import altair as alt
import pandas as pd


def build_raster_chart(
    stim_raster: pd.DataFrame,
    rec_channel: str,
    stim_channel: frozenset[str] = frozenset(),
    color_field: str | None = "target direction:N",
) -> alt.Chart:
    """Create a peri-stim raster chart for one recorded channel."""
    raster_df = (
        stim_raster[rec_channel]
        .xs(level="stimulated channel", key=stim_channel)
        .reset_index()
        .loc[lambda df: df[rec_channel] > 0]
        .assign(time=lambda df: df["time"].dt.total_seconds())
    )

    encodings: dict[str, alt.X | alt.Y | str] = {
        "x": alt.X("time:Q"),
        "y": alt.Y("trial_id:O"),
    }
    if color_field is not None:
        encodings["color"] = color_field

    return alt.Chart(raster_df).mark_tick().encode(**encodings)


def build_response_heatmap(
    response_table: pd.DataFrame,
    value_field: str = "stimulation response (z-score)",
    width: int = 400,
    height: int = 400,
) -> alt.Chart:
    """Create a recorded-by-stimulated response heatmap."""
    return (
        alt.Chart(response_table)
        .mark_rect()
        .encode(
            x=alt.X("recorded channel:N", title="Recorded Channel"),
            y=alt.Y("stimulated channel:N", title="Stimulated Channel"),
            color=alt.Color(
                f"{value_field}:Q",
                scale=alt.Scale(
                    scheme="redblue",
                    domainMid=0,
                    reverse=True,
                    domain=[-2.5, 1.5],
                ),
                title="firing rate change (z-score)",
            ),
            tooltip=["stimulated channel", "recorded channel", value_field],
        )
        .properties(width=width, height=height)
    )


def build_variance_ratio_chart(
    variance_ratio: pd.Series,
) -> alt.Chart:
    """Create a bar chart of post/pre variance ratio by stimulation group."""
    ratio_df = variance_ratio.reset_index()
    return (
        alt.Chart(ratio_df)
        .mark_bar()
        .encode(
            y=alt.Y("stimulated channel:N", title="Stimulated Channel"),
            x=alt.X("variance ratio:Q", title="Post-Stim / Pre-Stim Variance Ratio"),
            tooltip=["stimulated channel", "variance ratio"],
        )
    )


def build_electrode_heatmap(
    electrode_response: pd.DataFrame,
    value_field: str = "stimulation response (z-score)",
    width: int = 650,
    height: int = 400,
) -> alt.Chart:
    """Create coordinate heatmaps faceted by stimulated channel."""
    return (
        alt.Chart(electrode_response)
        .mark_rect()
        .encode(
            x=alt.X("x:O", title="X Coordinate"),
            y=alt.Y("y:O", title="Y Coordinate"),
            color=alt.Color(
                f"{value_field}:Q",
                scale=alt.Scale(
                    scheme="redblue",
                    domainMid=0,
                    reverse=True,
                    domain=[-2.5, 1.5],
                ),
                title="Z-scored Change in Firing Rate",
            ),
            row="stimulated channel:N",
            tooltip=["recorded channel", value_field, "x", "y"],
        )
        .properties(width=width, height=height)
    )