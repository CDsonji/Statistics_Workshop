from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_histogram_intuition(
    df: pd.DataFrame,
    column: str,
    bins_list: List[int] = [5, 20, 50],
    show_kde: bool = True,
) -> None:
    """Demonstrates how bin selection affects histogram interpretation, displays

    distribution modality (peaks), and highlights skewness using Mean vs.
    Median lines.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    column : str
        The continuous numerical column to visualize.
    bins_list : List[int]
        List of 3 bin sizes to compare (e.g. underfitting, optimal,
        overfitting).
    show_kde : bool
        Whether to overlay the Kernel Density Estimate (KDE) smooth curve.
    """
    fig, axes = plt.subplots(1, len(bins_list), figsize=(16, 5), sharey=True)

    data = df[column].dropna()
    mean_val = data.mean()
    median_val = data.median()

    for idx, bins in enumerate(bins_list):
        ax = axes[idx]

        # Draw histogram
        sns.histplot(
            data, bins=bins, kde=show_kde, ax=ax, color="teal", alpha=0.6
        )

        # Draw Mean vs Median lines
        ax.axvline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.1f}",
        )
        ax.axvline(
            median_val,
            color="green",
            linestyle="-.",
            linewidth=2,
            label=f"Median: {median_val:.1f}",
        )

        # Label bin effect
        if idx == 0:
            label_suffix = " (Too Few Bins / Oversmoothed)"
        elif idx == len(bins_list) - 1:
            label_suffix = " (Too Many Bins / Noisy)"
        else:
            label_suffix = " (Balanced Resolution)"

        ax.set_title(f"Bins = {bins}{label_suffix}", fontsize=11)
        ax.set_xlabel(column, fontsize=10)
        if idx == 0:
            ax.set_ylabel("Frequency (Count)", fontsize=10)

        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        f"Histogram Sensitivity & Distribution Intuition ({column})",
        fontsize=14,
        y=1.03,
    )
    plt.tight_layout()
    plt.show()

def plot_hourly_boxplot(
    df: pd.DataFrame,
    group_col: str = "hour",
    target_col: str = "total_vehicles",
) -> None:
    """Plots a 24-hour boxplot distribution showing hourly traffic spread,

    medians, means, and individual statistical outliers.
    """
    plt.figure(figsize=(15, 6))

    # Boxplot with red dots indicating hourly means
    sns.boxplot(
        data=df,
        x=group_col,
        y=target_col,
        palette="mako",
        showmeans=True,
        meanprops={
            "marker": "o",
            "markerfacecolor": "red",
            "markeredgecolor": "red",
            "markersize": "6",
        },
    )

    plt.title(
        f"Hourly Traffic Distribution & Outliers ({target_col})",
        fontsize=14,
        pad=12,
    )
    plt.xlabel("Hour of Day (0 to 23)", fontsize=11)
    plt.ylabel("Total Vehicle Count", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


def detect_iqr_outliers(
    df: pd.DataFrame,
    column: str,
    factor: float = 1.5,
    group_by: Optional[str] = None,
) -> pd.DataFrame:
    """Detects statistical outliers using the IQR rule.

    Parameters:
    -----------
    df : pd.DataFrame
        The input traffic DataFrame.
    column : str
        The numerical column to evaluate for outliers.
    factor : float, default 1.5
        The IQR multiplier threshold.
    group_by : Optional[str], default None
        If provided (e.g., 'hour'), calculates Q1, Q3, and IQR independently
        per group. If None, computes global IQR bounds across the whole dataset.

    Returns:
    --------
    pd.DataFrame
        Subsampled DataFrame containing only the outlier rows.
    """
    if group_by is None:
        # Global IQR Detection
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
        return df[outlier_mask].copy()

    # Group-wise IQR Detection (e.g., by Hour)
    def _is_group_outlier(group: pd.DataFrame) -> pd.Series:
        q1 = group[column].quantile(0.25)
        q3 = group[column].quantile(0.75)
        iqr = q3 - q1
        lb = q1 - factor * iqr
        ub = q3 + factor * iqr
        return (group[column] < lb) | (group[column] > ub)

    outlier_mask = df.groupby(group_by, group_keys=False).apply(
        _is_group_outlier
    )
    return df[outlier_mask].copy()


def apply_outlier_treatment(
    df: pd.DataFrame,
    column: str,
    strategy: str = "cap",
    factor: float = 1.5,
    group_by: Optional[str] = None,
) -> pd.Series:
    """Applies outlier treatment strategies globally or group-wise across a column.

    Strategies:
    -----------
    'cap' : Winsorizes (clips) values to IQR bounds (Q1 - factor*IQR, Q3 + factor*IQR).
    'log' : Applies log1p transformation log(1 + x) to compress right-hand skewness.
    'nan' : Replaces extreme outliers with NaN for conditional imputation.
    """
    series = df[column].copy()

    if strategy == "log":
        return np.log1p(series)

    # Global Treatment (group_by is None)
    if group_by is None:
        if strategy == "cap":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            return series.clip(lower=lower_bound, upper=upper_bound)

        elif strategy == "nan":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            series[(series < lower_bound) | (series > upper_bound)] = np.nan
            return series

    # Group-wise Treatment (e.g., group_by='hour')
    else:

        def _treat_group(group_series: pd.Series) -> pd.Series:
            Q1 = group_series.quantile(0.25)
            Q3 = group_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR

            if strategy == "cap":
                return group_series.clip(
                    lower=lower_bound, upper=upper_bound
                )
            elif strategy == "nan":
                group_series[
                    (group_series < lower_bound) | (group_series > upper_bound)
                ] = np.nan
                return group_series
            return group_series

        return df.groupby(group_by, group_keys=False)[column].transform(
            _treat_group
        )

    raise ValueError(
        f"Unknown strategy '{strategy}'. Choose 'cap', 'log', or 'nan'."
    )

def plot_scatter_intuition(
    df: pd.DataFrame,
    x_col: str = "total_vehicles",
    y_col: str = "average_speed",
    hue_col: Optional[str] = "hour",
) -> None:
    """Visualizes traffic volume vs.

    average speed color-coded by hour of day.
    """
    plt.figure(figsize=(9, 5))

    sns.scatterplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        palette="viridis",
        alpha=0.8,
    )

    plt.title("Traffic Volume vs. Average Speed (Color-Coded by Hour of Day)")
    plt.xlabel("Total Vehicles per Hour")
    plt.ylabel("Average Speed (km/h)")
    plt.tight_layout()
    plt.show()


def plot_hourly_traffic_profile(
    df: pd.DataFrame,
    hour_col: str = "hour",
    volume_col: str = "total_vehicles",
    peak_threshold_quantile: float = 0.75,
) -> Tuple[List[int], List[int]]:
    """Plots average hourly traffic volume as a continuous line trend and

    classifies peak vs. off-peak hours based on a quantile threshold.
    """
    # 1. Compute mean volume per hour
    hourly_avg = (
        df.groupby(hour_col)[volume_col].mean().reset_index()
    )

    # 2. Calculate peak cutoff threshold
    threshold_val = hourly_avg[volume_col].quantile(
        peak_threshold_quantile
    )

    peak_hours = hourly_avg[hourly_avg[volume_col] >= threshold_val][
        hour_col
    ].tolist()
    off_peak_hours = hourly_avg[hourly_avg[volume_col] < threshold_val][
        hour_col
    ].tolist()

    # 3. Render Line Plot
    plt.figure(figsize=(10, 5))

    # Base line plot with markers
    sns.lineplot(
        data=hourly_avg,
        x=hour_col,
        y=volume_col,
        color="slategray",
        linewidth=2.5,
        marker="o",
        markersize=6,
    )

    # Highlight Peak Hours with Red Dots
    peak_data = hourly_avg[hourly_avg[hour_col].isin(peak_hours)]
    plt.scatter(
        peak_data[hour_col],
        peak_data[volume_col],
        color="crimson",
        s=80,
        zorder=5,
        label=f"Peak Hours (Top {int((1-peak_threshold_quantile)*100)}%)",
    )

    # Threshold Line
    plt.axhline(
        threshold_val,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Peak Threshold ({threshold_val:.0f} veh/h)",
    )

    plt.title(
        "Average Hourly Traffic Volume Line Trend (Peak vs. Off-Peak Definition)"
    )
    plt.xlabel("Hour of Day (0 to 23)")
    plt.ylabel("Mean Total Vehicles")
    plt.xticks(range(0, 24))
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return peak_hours, off_peak_hours