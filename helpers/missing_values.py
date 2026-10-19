from typing import Any, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def inject_missing_values(
    df: pd.DataFrame,
    column: str,
    missing_ratio: float = 0.1,
    strategy: str = "mcar",
    condition_col: Optional[str] = None,
    condition_val: Optional[Any] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Injects missing values (NaN) into a target DataFrame column.

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    column : str
        Target column name to inject NaN values into.
    missing_ratio : float
        Fraction of values to replace with NaN (e.g., 0.15 for 15%).
    strategy : str
        'mcar' (Missing Completely At Random) or 'conditional' (e.g. night
        time sensor failure).
    condition_col : str, optional
        Column to evaluate for conditional injection (e.g., 'hour').
    condition_val : Any, optional
        Value to match for conditional injection (e.g., 17 for 5 PM).
    random_state : int
        Seed for reproducibility.

    Returns:
    --------
    pd.DataFrame
        A copy of the DataFrame containing injected missing values.
    """
    df_injected = df.copy()
    np.random.seed(random_state)

    if strategy == "mcar":
        mask = np.random.rand(len(df_injected)) < missing_ratio
        df_injected.loc[mask, column] = np.nan

    elif strategy == "conditional":
        if condition_col is None or condition_val is None:
            raise ValueError(
                "Both 'condition_col' and 'condition_val' must be provided for conditional strategy."
            )

        cond_mask = df_injected[condition_col] == condition_val
        matching_indices = df_injected[cond_mask].index

        num_to_mask = int(len(matching_indices) * missing_ratio)
        selected_indices = np.random.choice(
            matching_indices, size=num_to_mask, replace=False
        )
        df_injected.loc[selected_indices, column] = np.nan

    else:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Use 'mcar' or 'conditional'."
        )

    return df_injected


def plot_imputation_comparison(
    df_orig: pd.DataFrame,
    df_mean: pd.DataFrame,
    df_median: pd.DataFrame,
    column: str,
    title_suffix: str = "",
) -> None:
    """Plots a side-by-side comparative visualization showing the effects of Mean

    vs Median imputation against original data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Density Plot Comparison
    sns.kdeplot(
        df_orig[column].dropna(),
        ax=axes[0],
        label="Original Data",
        color="black",
        linewidth=2,
    )
    sns.kdeplot(
        df_mean[column],
        ax=axes[0],
        label="Mean Imputed",
        color="red",
        linestyle="--",
    )
    sns.kdeplot(
        df_median[column],
        ax=axes[0],
        label="Median Imputed",
        color="green",
        linestyle="-.",
    )

    axes[0].set_title(
        f"KDE Distribution Comparison ({column}) {title_suffix}", fontsize=12
    )
    axes[0].set_xlabel(column)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Boxplot Comparison (Variance & Outlier Distortion)
    comp_df = pd.DataFrame(
        {
            "Original": df_orig[column],
            "Mean Imputed": df_mean[column],
            "Median Imputed": df_median[column],
        }
    )

    sns.boxplot(data=comp_df, ax=axes[1], palette="Set2")
    axes[1].set_title(
        f"Boxplot & Outlier Comparison ({column})", fontsize=12
    )
    axes[1].set_ylabel(column)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()

def impute_missing_groupwise(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    strategy: str = "median",
) -> pd.DataFrame:
    """Imputes missing values in a column using group-specific central measures

    (e.g., imputing missing speeds using the median speed of that specific
    hour).

    Parameters:
    -----------
    df : pd.DataFrame
        The source DataFrame.
    target_col : str
        Column with missing values to fill.
    group_col : str
        Column to group by before computing statistics (e.g., 'hour').
    strategy : str
        'median' or 'mean'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with group-wise imputed values.
    """
    df_imputed = df.copy()

    if strategy == "median":
        group_stats = df_imputed.groupby(group_col)[target_col].transform(
            "median"
        )
    elif strategy == "mean":
        group_stats = df_imputed.groupby(group_col)[target_col].transform(
            "mean"
        )
    else:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Choose 'median' or 'mean'."
        )

    df_imputed[target_col] = df_imputed[target_col].fillna(group_stats)
    return df_imputed


def impute_missing_groupwise(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    strategy: str = "median",
) -> pd.DataFrame:
    """Imputes missing values using group-specific central measures."""
    df_imputed = df.copy()

    if strategy == "median":
        group_stats = df_imputed.groupby(group_col)[target_col].transform(
            "median"
        )
    elif strategy == "mean":
        group_stats = df_imputed.groupby(group_col)[target_col].transform(
            "mean"
        )
    else:
        raise ValueError(
            f"Unsupported strategy '{strategy}'. Choose 'median' or 'mean'."
        )

    df_imputed[target_col] = df_imputed[target_col].fillna(group_stats)
    return df_imputed


def plot_all_imputations_comparison(
    df_orig: pd.DataFrame,
    df_global_median: pd.DataFrame,
    df_global_mean: pd.DataFrame,
    df_group_median: pd.DataFrame,
    df_group_mean: pd.DataFrame,
    target_col: str,
    group_col: str,
) -> None:
    """Visualizes Global Median, Global Mean, Group-Wise Median, and Group-Wise Mean

    imputations against original data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))

    # 1. Overall Distribution Density (KDE)
    sns.kdeplot(
        df_orig[target_col].dropna(),
        ax=axes[0],
        label="Original Data",
        color="black",
        linewidth=2,
    )
    sns.kdeplot(
        df_global_median[target_col],
        ax=axes[0],
        label="Global Median Imputed",
        color="red",
        linestyle="--",
    )
    sns.kdeplot(
        df_global_mean[target_col],
        ax=axes[0],
        label="Global Mean Imputed",
        color="orange",
        linestyle=":",
    )
    sns.kdeplot(
        df_group_median[target_col],
        ax=axes[0],
        label="Group-Wise Median Imputed",
        color="green",
        linestyle="-.",
    )
    sns.kdeplot(
        df_group_mean[target_col],
        ax=axes[0],
        label="Group-Wise Mean Imputed",
        color="blue",
        linestyle="--",
    )

    axes[0].set_title(
        f"Overall Distribution Comparison ({target_col})", fontsize=12
    )
    axes[0].set_xlabel(target_col)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Hourly Profile Preservation
    sns.lineplot(
        data=df_orig,
        x=group_col,
        y=target_col,
        ax=axes[1],
        label="Original",
        color="black",
        linewidth=2,
        errorbar=None,
    )
    sns.lineplot(
        data=df_global_median,
        x=group_col,
        y=target_col,
        ax=axes[1],
        label="Global Median",
        color="red",
        linestyle="--",
        errorbar=None,
    )
    sns.lineplot(
        data=df_global_mean,
        x=group_col,
        y=target_col,
        ax=axes[1],
        label="Global Mean",
        color="orange",
        linestyle=":",
        errorbar=None,
    )
    sns.lineplot(
        data=df_group_median,
        x=group_col,
        y=target_col,
        ax=axes[1],
        label="Group Median (Hourly)",
        color="green",
        linestyle="-.",
        errorbar=None,
    )
    sns.lineplot(
        data=df_group_mean,
        x=group_col,
        y=target_col,
        ax=axes[1],
        label="Group Mean (Hourly)",
        color="blue",
        linestyle="--",
        errorbar=None,
    )

    axes[1].set_title(
        f"Hourly Traffic Profile Preservation ({target_col})", fontsize=12
    )
    axes[1].set_xlabel("Hour of Day (0 to 23)", fontsize=10)
    axes[1].set_ylabel(f"Mean {target_col}", fontsize=10)
    axes[1].legend()
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()