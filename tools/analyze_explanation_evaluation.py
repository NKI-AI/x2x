import argparse
import os

# Add x2x to path without installing it
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

from x2x.utils.logger import get_logger

logger = get_logger()


def create_log_dir(log_dir: str) -> Path:
    """Create timestamped log directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(log_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """Load and process the results CSV file.

    The output of the LLM is "VERY HIGH" / "HIGH NEAR BOUNDARY" / "LOW NEAR BOUNDARY" / "VERY HIGH"
    The analyses we wish to do:
        - All HIGH vs all LOW
        - VERY HIGH vs VERY LOW
        - Generally look at VERY HIGH vs NEAR BOUNDARY vs VERY LOW

    Parameters
    ----------
    csv_path: str
        Path to the CSV file containing the results

    Returns
    -------
    pd.DataFrame: Processed dataframe with the following columns:
        - y_hat: Binary predictions (0 or 1) for each sample.
        - y: Binary predictions (0 or 1) for each sample, based on the logit.
        - y_continuous: Continuous ground truth values or probabilities for each sample.
        - threshold_deviation: Absolute deviation from decision threshold.
    """
    df = pd.read_csv(csv_path)
    df["y_hat"] = df["conclusion"].apply(
        lambda x: 0 if "LOW" in x else 1 if "HIGH" in x else -1
    )
    df["y"] = (df["logit"] > -0.135).astype(int)  # binary for metrics
    df["y_continuous"] = df["logit"]  # continuous for visualization

    # Used to create subsets of the dataset of samples with very high or low predictions
    # by the prognostics model under test.
    df["threshold_deviation"] = np.abs(df["logit"] - (-0.135))

    return df


def bootstrap_auroc(
    y_hat: pd.Series,
    y_continuous: pd.Series,
    n_iterations: int = 1000,
    sample_size: int | None = None,
    min_successful_iterations: int = 100,
) -> tuple[float, float]:
    """Calculate AUROC with bootstrapped confidence intervals.

    This function calculates the Area Under the Receiver Operating Characteristic (AUROC)
    curve using bootstrap resampling to estimate confidence intervals. It performs multiple
    iterations of sampling with replacement to generate bootstrap samples and computes
    AUROC statistics across these samples.

    Parameters
    ----------
    y_hat : pd.Series
        Binary predictions (0 or 1) for each sample.
    y_continuous : pd.Series
        Continuous ground truth values or probabilities for each sample.
    n_iterations : int, default=1000
        Number of bootstrap iterations to perform.
    sample_size : int or None, default=None
        Number of samples to draw in each bootstrap iteration.
        If None, uses the same size as the input data (standard bootstrap).
        If int, uses that specific number of samples.
    min_successful_iterations : int, default=100
        Minimum number of successful iterations required.
        If fewer iterations succeed, returns np.nan.

    Returns
    -------
    mean_auroc : float
        Mean AUROC across successful bootstrap samples.
    std_auroc : float
        Standard deviation of AUROC across successful bootstrap samples.

    Notes
    -----
    - Uses sampling with replacement (standard bootstrap procedure)
    - Each bootstrap sample can contain duplicates of original samples
    - Some original samples may not appear in a given bootstrap sample
    - Default behavior uses same sample size as input data
    - Iterations may fail if a bootstrap sample contains only one class
    - Returns statistics only if enough iterations succeed
    """
    if len(y_hat) == 0:
        return np.nan, np.nan

    # If sample_size not specified, use the size of input data
    if sample_size is None:
        sample_size = len(y_hat)

    auroc_scores = []
    failed_iterations = 0

    for _ in range(n_iterations):
        # Generate bootstrap sample indices
        indices = resample(
            np.arange(len(y_hat)),
            replace=True,
            n_samples=sample_size,
        )

        # Get the bootstrap sample
        y_hat_sample = y_hat.iloc[indices]
        y_continuous_sample = y_continuous.iloc[indices]

        # Check if we have both classes in the sample
        if len(np.unique(y_hat_sample)) < 2:
            failed_iterations += 1
            continue

        # Calculate AUROC for this bootstrap sample
        try:
            auroc = roc_auc_score(y_hat_sample, y_continuous_sample)
            auroc_scores.append(auroc)
        except Exception as e:
            failed_iterations += 1
            continue

    # Return statistics only if we have enough successful iterations
    if len(auroc_scores) >= min_successful_iterations:
        success_rate = (n_iterations - failed_iterations) / n_iterations
        logger.info(
            f"Bootstrap success rate: {success_rate:.2%} ({len(auroc_scores)} successful iterations)"
        )
        return np.mean(auroc_scores), np.std(auroc_scores)
    else:
        logger.warning(
            f"Only {len(auroc_scores)} successful bootstrap iterations (minimum required: {min_successful_iterations})"
        )
        return np.nan, np.nan


def create_auroc_comparison_table(explanation_results_dfs: dict[str, pd.DataFrame]):
    """Create formatted tables comparing AUROC scores across different data subsets.

    This function generates both a plain text and a LaTeX table comparing AUROC scores
    for different subsets of the data based on the threshold deviation.

    Parameters
    ----------
    explanation_results_dfs : dict[str, pd.DataFrame]
        Dictionary mapping dataset names to their corresponding DataFrames.
        Each DataFrame must contain the following columns:
            - y_hat : binary predictions (0 or 1)
            - y_continuous : continuous ground truth values
            - threshold_deviation : absolute deviation from decision threshold

    Returns
    -------
    table_str : str
        Formatted plain text table showing AUROC scores (mean ± std) for each subset.
        The table includes the following columns:
            - X: All valid data points
            - X1: Points with threshold deviation > 1 (very high)
            - X3: Points with threshold deviation > 3 (very high)

    latex_str : str
        LaTeX-formatted version of the same table, suitable for academic papers.

    Notes
    -----
    - Invalid predictions (y_hat == -1) are filtered out before analysis
    - AUROC scores are calculated using bootstrap resampling for confidence intervals
    """
    results = {}

    for dataset_name, df in explanation_results_dfs.items():
        # Filter out invalid predictions
        df = df[df["y_hat"] != -1]

        # Create all subsets
        X = df
        X1 = df[df["threshold_deviation"] > 1]
        X3 = df[df["threshold_deviation"] > 3]

        # Calculate AUROC for each subset
        logger.info(f"Computing AUROCS for {dataset_name}")

        results[dataset_name] = {
            "All Data": {
                "X": bootstrap_auroc(X["y_hat"], X["y_continuous"]),
                "X1": bootstrap_auroc(X1["y_hat"], X1["y_continuous"]),
                "X3": bootstrap_auroc(X3["y_hat"], X3["y_continuous"]),
            },
        }

    # Create a nicely formatted string table
    table_str = "AUROC Scores for Different Data Subsets (mean ± std)\n\n"
    table_str += "=" * 80 + "\n"
    table_str += f"{'Dataset':<20} | {'All Data':-^56}|\n"
    table_str += f"{'':20} | {'X':^16} | {'X1':^16} | {'X3':^16} |\n"
    table_str += "-" * 80 + "\n"

    for dataset_name, scores in results.items():
        row = f"{dataset_name:<20} | "
        row += f"{scores['All Data']['X'][0]:.2f}±{scores['All Data']['X'][1]:.2f} | "
        row += f"{scores['All Data']['X1'][0]:.2f}±{scores['All Data']['X1'][1]:.2f} | "
        row += f"{scores['All Data']['X3'][0]:.2f}±{scores['All Data']['X3'][1]:.2f} | "
        table_str += row

    table_str += "\n" + "=" * 80 + "\n"

    # Create LaTeX table
    latex_str = "\\begin{table}[h]\n\\centering\n\\caption{AUROC Scores for Different Data Subsets}\n"
    latex_str += "\\begin{tabular}{l|ccc}\n\\hline\n"
    latex_str += "& \\multicolumn{3}{c|}{All Data} \\\\\n"
    latex_str += "Dataset & X & X1 & X3 \\\\\n\\hline\n"

    for dataset_name, scores in results.items():
        row = f"{dataset_name} & "
        row += (
            f"{scores['All Data']['X'][0]:.2f}$\\pm${scores['All Data']['X'][1]:.2f} & "
        )
        row += f"{scores['All Data']['X1'][0]:.2f}$\\pm${scores['All Data']['X1'][1]:.2f} & "
        row += f"{scores['All Data']['X3'][0]:.2f}$\\pm${scores['All Data']['X3'][1]:.2f} & "
        row += "\\\\\n"
        latex_str += row

    latex_str += "\\hline\n\\end{tabular}\n\\end{table}"

    return table_str, latex_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default=None)
    args = parser.parse_args()

    args.log_dir = create_log_dir(args.log_dir)

    # Dictionary to store dataframes for combined table
    explanation_results_dfs = {}

    for csv in Path(args.input_dir).glob("**/results.csv"):
        # Use the parent directory name as the dataset name for combined table
        dataset_name = csv.parent.parent.name

        # Load and process data
        df = load_and_process_data(csv)

        # Check how many incorrect responses we got, we set y to -1 if the response is not clear
        incorrect_responses = df[df["y_hat"] == -1]
        logger.info(f"Number of incorrect responses: {len(incorrect_responses)}")

        # Filter these, otherwise metrics wont work
        df = df[df["y_hat"] != -1]

        # Store filtered df for combined plot
        explanation_results_dfs[dataset_name] = df

    # Save results table to log dir
    if explanation_results_dfs:
        # Create AUROC comparison table
        table_str, latex_str = create_auroc_comparison_table(explanation_results_dfs)

        auroc_comparison_table_path = args.log_dir / "auroc_comparison_table.txt"
        auroc_comparison_latex_table_path = (
            args.log_dir / "auroc_comparison_latex_table.txt"
        )

        with open(auroc_comparison_table_path, "w") as f:
            logger.info(
                f"Saving AUROC comparison table to {auroc_comparison_table_path}"
            )
            logger.info(f"AUROC comparison table: \n{table_str}")
            f.write(table_str)
        with open(auroc_comparison_latex_table_path, "w") as f:
            logger.info(
                f"Saving AUROC comparison latex table to {auroc_comparison_latex_table_path}"
            )
            f.write(latex_str)
