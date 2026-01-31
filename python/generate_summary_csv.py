#!/usr/bin/env python3
"""
Generate a summary CSV with benchmarks as rows and targets as columns.
"""

import csv
from pathlib import Path


def discover_all_polybench_benchmarks(script_dir):
    """Discover all benchmarks from multiple directories including skipped ones."""
    benchmark_dirs = ["polybench", "cavity_flow", "go_fast", "spmv", "weather_stencils"]

    all_benchmarks = {}

    for dir_name in benchmark_dirs:
        bench_dir = script_dir / "benchmarks" / "npbench" / dir_name

        if not bench_dir.exists():
            print(f"Warning: Directory {bench_dir} does not exist, skipping...")
            continue

        print(f"Scanning directory: {dir_name}")
        test_files = sorted(bench_dir.glob("test_*.py"))

        for test_file in test_files:
            benchmark_name = test_file.stem.replace("test_", "")

            # Check if the test function is marked with @pytest.mark.skip()
            with open(test_file, "r") as f:
                content = f.read()
                is_skipped = "@pytest.mark.skip(" in content

            all_benchmarks[benchmark_name] = {
                "skipped": is_skipped,
                "file": test_file,
                "directory": dir_name,
            }

    return all_benchmarks


def main():
    script_dir = Path(__file__).parent
    input_file = script_dir / "polybench_results.csv"
    output_file = script_dir / "polybench_summary.csv"

    # Discover all benchmarks including skipped ones
    all_benchmarks = discover_all_polybench_benchmarks(script_dir)

    # Read the input CSV
    data = {}
    with open(input_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            benchmark = row["benchmark"]
            target = row["target"]

            if benchmark not in data:
                data[benchmark] = {}

            # Use "skipped" if the benchmark failed, otherwise use avg_cached_time
            if row["success"] == "False" or row["avg_cached_time"] == "":
                data[benchmark][target] = "skipped"
            else:
                data[benchmark][target] = row["avg_cached_time"]

    # Define the target columns in the desired order
    targets = ["none", "sequential", "openmp", "cuda", "numpy"]

    # Write the output CSV
    with open(output_file, "w", newline="") as f:
        fieldnames = ["benchmark", ""] + [f"target_{t}" for t in targets]
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()

        # Include all discovered benchmarks
        for benchmark_name in sorted(all_benchmarks.keys()):
            row = {"benchmark": benchmark_name, "": ""}

            # If benchmark was skipped, mark all targets as skipped
            if all_benchmarks[benchmark_name]["skipped"]:
                for target in targets:
                    column_name = f"target_{target}"
                    row[column_name] = "skipped"
            else:
                # Use data from CSV if available
                for target in targets:
                    column_name = f"target_{target}"
                    if benchmark_name in data:
                        value = data[benchmark_name].get(target, "skipped")
                        # Replace dot with comma for decimals
                        if value != "skipped":
                            value = value.replace(".", ",")
                        row[column_name] = value
                    else:
                        row[column_name] = "skipped"

            writer.writerow(row)

    total_benchmarks = len(all_benchmarks)
    skipped_count = sum(1 for b in all_benchmarks.values() if b["skipped"])
    ran_count = total_benchmarks - skipped_count

    print(f"Summary CSV saved to: {output_file}")
    print(f"Total benchmarks: {total_benchmarks}")
    print(f"Ran: {ran_count}")
    print(f"Skipped: {skipped_count}")


if __name__ == "__main__":
    main()
