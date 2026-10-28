"""Visualization dashboard builder using Bokeh."""

import os
from typing import Dict, List
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save

from src.analyzer import ModelFit, PointMatch


class Visualizer:
    """Builds interactive HTML dashboards displaying curve fits and test mappings."""

    def __init__(self, output_file_path: str = "output/dashboard.html"):
        self.output_file_path = output_file_path
        output_dir = os.path.dirname(self.output_file_path)
        # Create output directory if a subfolder was specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def plot(
        self,
        train_df: pd.DataFrame,
        ideal_df: pd.DataFrame,
        fits: Dict[str, ModelFit],
        test_matches: List[PointMatch],
    ) -> str:
        """Render a 2x2 grid dashboard for the 4 chosen models."""
        output_file(self.output_file_path, title="Function Matching Dashboard")

        test_data = [
            {
                "x": m.x,
                "y": m.y,
                "delta_y": m.delta_y if m.delta_y is not None else 0.0,
                "assigned_model": m.ideal_col,
            }
            for m in test_matches
        ]
        # Ensure DataFrame has expected schema even if no test points matched
        test_df = pd.DataFrame(
            test_data, 
            columns=["x", "y", "delta_y", "assigned_model"]
        )

        panels = []
        palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd"]

        for idx, (t_col, fit) in enumerate(fits.items()):
            i_col = fit.ideal_col
            color = palette[idx % len(palette)]

            fig = figure(
                title=f"{t_col} -> {i_col} (Max Dev: {fit.max_dev:.3f}, Tol: {fit.threshold:.3f})",
                x_axis_label="X Coordinate",
                y_axis_label="Y Value",
                width=600,
                height=400,
                tools="pan,box_zoom,wheel_zoom,reset,save",
            )

            ideal_source = ColumnDataSource(
                data={
                    "x": ideal_df["x"],
                    "y_ideal": ideal_df[i_col],
                    "upper": ideal_df[i_col] + fit.threshold,
                    "lower": ideal_df[i_col] - fit.threshold,
                }
            )

            # Shaded acceptance tolerance band
            fig.varea(
                x="x",
                y1="lower",
                y2="upper",
                source=ideal_source,
                fill_color="gray",
                fill_alpha=0.2,
                legend_label="Acceptance Band (+/- Max Dev * sqrt(2))",
            )

            # Ideal curve
            ideal_line = fig.line(
                "x",
                "y_ideal",
                source=ideal_source,
                line_width=2.5,
                line_color="crimson",
                legend_label=f"Ideal Model ({i_col})",
            )

            # Training points
            train_source = ColumnDataSource(
                data={"x": train_df["x"], "y_train": train_df[t_col]}
            )
            train_dots = fig.circle(
                "x",
                "y_train",
                source=train_source,
                size=4,
                color=color,
                alpha=0.6,
                legend_label=f"Training Points ({t_col})",
            )

            # Mapped test points for this model
            matched_points = test_df[test_df["assigned_model"] == i_col]
            if not matched_points.empty:
                test_source = ColumnDataSource(matched_points)
                test_dots = fig.square(
                    "x",
                    "y",
                    source=test_source,
                    size=8,
                    color="limegreen",
                    line_color="black",
                    legend_label=f"Mapped Test Points ({i_col})",
                )
                test_hover = HoverTool(
                    renderers=[test_dots],
                    tooltips=[
                        ("Type", "Mapped Test Point"),
                        ("(x, y)", "(@x{0.00}, @y{0.00})"),
                        ("Delta Y", "@delta_y{0.0000}"),
                        ("Model", "@assigned_model"),
                    ],
                )
                fig.add_tools(test_hover)

            general_hover = HoverTool(
                renderers=[ideal_line, train_dots],
                tooltips=[("X", "@x{0.00}"), ("Y", "$y{0.00}")],
            )
            fig.add_tools(general_hover)

            fig.legend.location = "top_left"
            fig.legend.click_policy = "hide"
            panels.append(fig)

        dashboard_grid = gridplot(panels, ncols=2)
        save(dashboard_grid)
        return self.output_file_path