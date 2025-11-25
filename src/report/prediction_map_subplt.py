import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import contextily as ctx
from matplotlib.lines import Line2D
from shapely.geometry import box
from src.report.funs._plot_style import PlotStyle
from src.report.funs._load_setup import load_setup
import ast


# -- CONFIG --
save = True


# -- LOAD DATA --
# Predictions and model
data, predictions, _, info, _ = load_setup(task="clf")
conformal = pd.read_csv("models/clf_conformal_predictions.csv")

# Geo data
gdf = gpd.read_file("data/interim/urau.geojson")


# -- PREPARE DATA --
# Convert to Web Mercator for basemap compatibility
gdf = gdf.to_crs(epsg=3857)

# Define tighter bounding box for mainland Europe + UK only
minx, miny, maxx, maxy = (-10, 36, 35, 75)
bbox = gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs="EPSG:4326").to_crs(
    epsg=3857
)

# Keep only points within the bounding box
gdf = gdf[gdf.geometry.within(bbox.geometry.iloc[0])]

# Convert polygons to centroids
gdf["geometry"] = gdf.centroid

# Select relevant columns from conformal
conformal = conformal[
    [
        "eu_city_code",
        "eu_cycling_level",
        "predicted",
        "prediction_set",
        "prediction_set_size",
    ]
]

# Merge predictions with geographic data
gdf = gdf.merge(conformal, on="eu_city_code", how="left")

# Map cycling levels to numeric values for visualization
cycling_level_map = {
    "low": 0,
    "moderate": 1,
    "high": 2,
}
gdf["cycling_level_numeric"] = gdf["eu_cycling_level"].map(cycling_level_map)


# Download country borders and ensure compatibility
countries_url = (
    "https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip"
)
countries = gpd.read_file(countries_url)
countries = countries.to_crs(epsg=3857)
countries_clipped = gpd.clip(countries, bbox.geometry.iloc[0])


# -- PLOT --
# Initialize plot style
style = PlotStyle().apply()
figsize = PlotStyle.figsize_from_pt(fraction=2, ratio=0.4)  # Wider for side-by-side


# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="white")

# Split the data into true values and predicted values
true_values = gdf[~gdf["predicted"]]
predicted_values = gdf[gdf["predicted"]]


# Function to map numerical values to colors
def get_color(value):
    if value == 0:
        return style.colors["low-m"]
    elif value == 1:
        return style.colors["mid-m"]
    else:
        return style.colors["high-m"]


# Function to get marker size based on class (low=biggest, high=smallest)
def get_marker_size_by_order(order_position):
    """Get marker size based on position in prediction set (0=main, 1=second, 2=third)"""
    if order_position == 0:  # Main prediction (smallest, on top)
        return 35  # Smallest
    elif order_position == 1:  # Second prediction (medium, middle layer)
        return 80  # Medium
    else:  # Third+ prediction (largest, bottom layer)
        return 120  # Biggest


# Function to plot map for given axis and data
def plot_map(ax, data, marker_style="o", alpha=0.9, size=60, use_prediction_sets=False):
    if not use_prediction_sets:
        # Original plotting logic for true values
        for level in [0, 1, 2]:  # Low, medium, high
            subset = data[data["cycling_level_numeric"] == level]
            ax.scatter(
                subset.geometry.x,
                subset.geometry.y,
                color=get_color(level),
                s=size,
                marker=marker_style,
                edgecolor="white",
                linewidth=0.2,
                alpha=alpha,
                zorder=1,
            )
    else:
        # Modified plotting logic for predicted values with prediction sets
        for _, row in data.iterrows():
            x, y = row.geometry.x, row.geometry.y

            # Determine alpha based on prediction_set_size
            if pd.notna(row["prediction_set_size"]) and row["prediction_set_size"] == 1:
                current_alpha = 0.4  # More opaque for confident single predictions
            else:
                current_alpha = 0.6  # Less opaque for uncertain multiple predictions

            # Check if prediction_set is not null and has multiple predictions
            if (
                pd.notna(row["prediction_set"])
                and row["prediction_set"] != ""
                and row["prediction_set_size"] > 1
            ):
                # Parse the prediction set string to list
                if isinstance(row["prediction_set"], str):
                    prediction_classes = ast.literal_eval(row["prediction_set"])
                else:
                    prediction_classes = row["prediction_set"]

                # Plot in normal order: main prediction first (largest, bottom layer)
                for i, class_name in enumerate(prediction_classes):
                    class_numeric = cycling_level_map.get(class_name, 0)
                    marker_size = get_marker_size_by_order(i)

                    actual_order_position = len(prediction_classes) - 1 - i
                    marker_size = get_marker_size_by_order(actual_order_position)

                    # Use higher z-order for smaller circles (later in the reversed loop)
                    z_order = 1 + i

                    ax.scatter(
                        x,
                        y,
                        color=get_color(class_numeric),
                        s=marker_size,
                        marker="o",
                        edgecolor="white",
                        linewidth=0.2,
                        alpha=current_alpha,
                        zorder=z_order,
                    )

            else:
                # Single prediction point
                level = row["cycling_level_numeric"]
                ax.scatter(
                    x,
                    y,
                    color=get_color(level),
                    s=size,
                    marker="o",
                    edgecolor="white",
                    linewidth=0.2,
                    alpha=current_alpha,
                    zorder=3,
                )

    # Expand display area to include more of Europe
    display_bbox = gpd.GeoDataFrame(
        geometry=[
            box(
                -15,
                30,
                35,
                68,
            )
        ],
        crs="EPSG:4326",
    ).to_crs(epsg=3857)

    # Set axis limits to this expanded area
    display_bounds = display_bbox.total_bounds
    ax.set_xlim(display_bounds[0], display_bounds[2])
    ax.set_ylim(display_bounds[1], display_bounds[3])

    # Add a basemap
    ctx.add_basemap(
        ax,
        source=ctx.providers.CartoDB.VoyagerNoLabels,
        zoom=6,
        alpha=0.6,
    )

    # Add country borders
    countries_clipped.boundary.plot(
        ax=ax,
        color="gray",
        linewidth=0.8,
        alpha=0.1,
        zorder=0.5,  # Bottom position
    )

    # Remove axes for cleaner look
    ax.set_axis_off()
    ax.set_aspect(0.7)


# Plot both maps
plot_map(
    ax1, true_values, marker_style="o", alpha=0.8, size=40, use_prediction_sets=False
)
plot_map(
    ax2, predicted_values, marker_style="h", alpha=0.6, size=33, use_prediction_sets=True
)

# Add A and B labels in upper right corner
ax1.text(
    0.95,
    0.95,
    "A",
    transform=ax1.transAxes,
    fontsize=16,
    fontweight="bold",
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    color=style.colors["text"],
)

ax2.text(
    0.95,
    0.95,
    "B",
    transform=ax2.transAxes,
    fontsize=16,
    fontweight="bold",
    verticalalignment="top",
    horizontalalignment="right",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.8),
    color=style.colors["text"],
)


# Create color legend (for both subplots)
legend_colors = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=style.colors["low-m"],
        markeredgecolor="white",
        markersize=10,
        linewidth=0.5,
        label="Low",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=style.colors["mid-m"],
        markeredgecolor="white",
        markersize=10,
        linewidth=0.5,
        label="Moderate",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=style.colors["high-m"],
        markeredgecolor="white",
        markersize=10,
        linewidth=0.5,
        label="High",
    ),
]

# Create prediction legend (for right subplot)
legend_prediction = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="lightgrey",
        markeredgecolor="white",
        markersize=10,
        linewidth=0.5,
        label="Predicted Class",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="whitesmoke",  # Light grey fill
        markeredgecolor="lightgrey",  # Dark outline color
        markersize=8,
        markeredgewidth=2,
        label="Alternative Classes",
    ),
]

# Add divider line
divider_line = Line2D(
    [0], [0], color="lightgrey", alpha=0.8, linewidth=0.9, linestyle="-"
)

# Combine handles with divider
combined_handles = legend_colors + [divider_line] + legend_prediction

# Add combined legend to subplot A
ax1.legend(
    handles=combined_handles,
    labels=[h.get_label() for h in legend_colors]
    + [""]
    + [h.get_label() for h in legend_prediction],
    loc="upper left",
    frameon=True,
    fancybox=True,
    facecolor="white",
    edgecolor="none",
    framealpha=1.0,
    fontsize=8,
    labelcolor=style.colors["text"],
    ncol=1,
    borderpad=1.0,
    handletextpad=0.8,
)


# Reduce spacing between subplots
plt.subplots_adjust(wspace=0, left=0, right=1, top=1, bottom=0)

plt.tight_layout()

# Save
if save:
    plt.savefig("output/figs/mode_share_map_subplt.pdf", bbox_inches="tight", dpi=600)
    plt.savefig(
        "output/figs/ppt/mode_share_map_comparison_subplt.svg",
        bbox_inches="tight",
        dpi=600,
    )

plt.show()
