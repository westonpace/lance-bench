#!/usr/bin/env python3
"""Generate regression analysis report for benchmark results.

This script analyzes benchmark results to detect potential performance regressions
using statistical testing (t-test) and visualizes trends over time.
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# Add packages to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "packages"))

from lance_bench_db.dataset import connect, get_database_uri
from lance_bench_db.models import Result


def fetch_all_results() -> list[dict]:
    """Fetch all benchmark results from the database.

    Returns:
        List of result dictionaries
    """
    print(f"Connecting to database at <{get_database_uri()}>...")
    db = connect()
    results_table = Result.open_table(db)

    print("Fetching all results...")
    # Fetch all results
    results = results_table.to_pandas()

    print(f"Fetched {len(results)} total results")
    return results.to_dict("records")


def group_and_sort_results(results: list[dict]) -> dict[str, list[dict]]:
    """Group results by benchmark name and sort by timestamp.

    Args:
        results: List of result dictionaries

    Returns:
        Dictionary mapping benchmark_name to sorted list of results
    """
    grouped = defaultdict(list)

    for result in results:
        benchmark_name = result["benchmark_name"]
        grouped[benchmark_name].append(result)

    # Sort each group by dut.timestamp
    for benchmark_name in grouped:
        grouped[benchmark_name].sort(key=lambda r: r["dut"]["timestamp"])

    print(f"Grouped into {len(grouped)} unique benchmarks")
    return dict(grouped)


def calculate_regression_pvalue(results: list[dict], recent_count: int = 4) -> float | None:
    """Calculate p-value for potential regression using t-test.

    Compares the most recent N results against all older results.

    Args:
        results: List of results sorted by timestamp (oldest to newest)
        recent_count: Number of recent results to compare

    Returns:
        P-value from t-test, or None if insufficient data
    """
    if len(results) < recent_count + 2:
        # Need at least recent_count + 2 results for meaningful comparison
        return None

    # Extract mean values from summary
    all_means = [r["summary"]["mean"] for r in results]

    # Split into recent and older results
    older_means = all_means[:-recent_count]
    recent_means = all_means[-recent_count:]

    # Perform two-sample t-test (two-tailed)
    # Null hypothesis: recent and older results have same mean
    # Lower p-value suggests they're different (potential regression)
    result = stats.ttest_ind(recent_means, older_means)
    # Result is tuple-like with (statistic, pvalue)
    pvalue: float = result[1]  # type: ignore[index]

    return pvalue


def analyze_benchmarks(grouped_results: dict[str, list[dict]], recent_count: int = 4) -> list[tuple[str, float, list[dict]]]:
    """Analyze all benchmarks and calculate p-values.

    Args:
        grouped_results: Dictionary of benchmark_name -> sorted results
        recent_count: Number of recent results to compare

    Returns:
        List of (benchmark_name, p_value, results) tuples sorted by p-value (descending)
    """
    analyzed = []

    for benchmark_name, results in grouped_results.items():
        pvalue = calculate_regression_pvalue(results, recent_count)

        if pvalue is not None:
            analyzed.append((benchmark_name, pvalue, results))
        else:
            print(f"Skipping {benchmark_name}: insufficient data ({len(results)} results)")

    # Sort by p-value descending (highest p-value = least likely regression)
    analyzed.sort(key=lambda x: x[1], reverse=True)

    print(f"\nAnalyzed {len(analyzed)} benchmarks with sufficient data")
    return analyzed


def create_regression_chart(analyzed_benchmarks: list[tuple[str, float, list[dict]]], output_path: Path) -> None:
    """Create a multi-plot chart showing benchmark trends.

    Args:
        analyzed_benchmarks: List of (benchmark_name, p_value, results) tuples
        output_path: Path to save the chart
    """
    n_benchmarks = len(analyzed_benchmarks)
    if n_benchmarks == 0:
        print("No benchmarks to plot")
        return

    # Create figure with subplots (one per benchmark)
    fig, axes = plt.subplots(
        nrows=n_benchmarks,
        ncols=1,
        figsize=(12, max(n_benchmarks * 2, 10)),
        sharex=False,
    )

    # Handle single benchmark case (axes is not an array)
    if n_benchmarks == 1:
        axes = [axes]

    print(f"\nCreating chart with {n_benchmarks} subplots...")

    for idx, (benchmark_name, pvalue, results) in enumerate(analyzed_benchmarks):
        ax = axes[idx]

        # Extract data for plotting
        timestamps = [r["dut"]["timestamp"] for r in results]
        means = [r["summary"]["mean"] for r in results]

        # Convert timestamps to datetime for better x-axis labels
        dates = [datetime.fromtimestamp(ts) for ts in timestamps]

        # Plot the data
        ax.plot(dates, means, marker="o", linestyle="-", linewidth=1, markersize=4)

        # Add vertical line separating recent results
        if len(results) >= 4:
            split_date = dates[-4]
            ax.axvline(x=split_date, color="red", linestyle="--", alpha=0.5, linewidth=1)

        # Set title with p-value
        color = "green" if pvalue > 0.05 else "orange" if pvalue > 0.01 else "red"
        ax.set_title(f"{benchmark_name} (p={pvalue:.4f})", fontsize=10, color=color)

        # Labels
        ax.set_ylabel("Time (ns)", fontsize=8)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.tick_params(axis="x", rotation=45)

    # Set x-label on bottom plot
    axes[-1].set_xlabel("Date", fontsize=8)

    # Overall title
    fig.suptitle("Benchmark Regression Analysis\n(Sorted by p-value: High→Low)", fontsize=14, fontweight="bold")

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=(0, 0, 1, 0.98))

    # Save figure
    print(f"Saving chart to {output_path}...")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print("✓ Chart saved successfully")


def print_summary(analyzed_benchmarks: list[tuple[str, float, list[dict]]], threshold: float = 0.05) -> None:
    """Print summary of regression analysis.

    Args:
        analyzed_benchmarks: List of (benchmark_name, p_value, results) tuples
        threshold: P-value threshold for flagging regressions
    """
    print("\n" + "=" * 80)
    print("REGRESSION ANALYSIS SUMMARY")
    print("=" * 80)

    # Count potential regressions
    regressions = [b for b in analyzed_benchmarks if b[1] < threshold]
    warnings = [b for b in analyzed_benchmarks if 0.01 <= b[1] < threshold]
    likely_regressions = [b for b in analyzed_benchmarks if b[1] < 0.01]

    print(f"\nTotal benchmarks analyzed: {len(analyzed_benchmarks)}")
    print(f"Potential regressions (p < {threshold}): {len(regressions)}")
    print(f"  - High concern (p < 0.01): {len(likely_regressions)}")
    print(f"  - Medium concern (0.01 ≤ p < 0.05): {len(warnings)}")
    print(f"Likely stable (p ≥ {threshold}): {len(analyzed_benchmarks) - len(regressions)}")

    if likely_regressions:
        print("\n⚠️  HIGH CONCERN BENCHMARKS (p < 0.01):")
        for benchmark_name, pvalue, results in likely_regressions:
            recent_mean = np.mean([r["summary"]["mean"] for r in results[-4:]])
            older_mean = np.mean([r["summary"]["mean"] for r in results[:-4]])
            change_pct = ((recent_mean - older_mean) / older_mean) * 100
            print(f"  - {benchmark_name}")
            print(f"    p-value: {pvalue:.6f}")
            print(f"    Change: {change_pct:+.2f}% ({'slower' if change_pct > 0 else 'faster'})")

    if warnings:
        print("\n⚠️  MEDIUM CONCERN BENCHMARKS (0.01 ≤ p < 0.05):")
        for benchmark_name, pvalue, _ in warnings:
            print(f"  - {benchmark_name} (p={pvalue:.4f})")

    print("\n" + "=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate regression analysis report for benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate default report
  python regression_report.py

  # Specify output location and recent count
  python regression_report.py -o report.png --recent-count 5

  # Use custom p-value threshold
  python regression_report.py --threshold 0.01
        """,
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("regression_report.png"),
        help="Output path for the chart (default: regression_report.png)",
    )
    parser.add_argument(
        "--recent-count",
        type=int,
        default=4,
        help="Number of recent results to compare against older results (default: 4)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="P-value threshold for flagging regressions (default: 0.05)",
    )

    args = parser.parse_args()

    try:
        # Fetch all results
        results = fetch_all_results()

        if not results:
            print("No results found in database")
            sys.exit(1)

        # Group and sort by benchmark
        grouped_results = group_and_sort_results(results)

        # Analyze for regressions
        analyzed_benchmarks = analyze_benchmarks(grouped_results, args.recent_count)

        if not analyzed_benchmarks:
            print("No benchmarks with sufficient data for analysis")
            sys.exit(1)

        # Print summary
        print_summary(analyzed_benchmarks, args.threshold)

        # Create chart
        create_regression_chart(analyzed_benchmarks, args.output)

        print(f"\n✅ Report generated successfully: {args.output}")

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
