import argparse
import time
import numpy as np
import pytest
import docc


class SDFGVerification:
    def __init__(self, verification: dict):
        self._verification = verification

    def verify(self, stats: dict) -> None:
        print(stats)
        for key, val in self._verification.items():
            if key not in stats:
                assert val == 0, f"Key {key} not found in stats"
            else:
                assert (
                    stats[key] == val
                ), f"Key {key} has value {stats[key]} but expected {val}"
        print("All verifications passed.")


def run_benchmark(initialize_func, kernel_func, parameters, name, args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", type=str, default="paper")
        parser.add_argument("--docc", action="store_true")
        parser.add_argument("--numpy", action="store_true")
        parser.add_argument("--target", type=str, default="none")
        parser.add_argument("--n_runs", type=int, default=10)
        args = parser.parse_args()

    if args.size not in parameters:
        print(f"Unknown size: {args.size}. Available sizes: {list(parameters.keys())}")
        return

    params = parameters[args.size]
    inputs = initialize_func(**params)

    # Unpack inputs for the kernel
    # We assume initialize_func returns a tuple of arguments
    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    if args.numpy:
        # Create a copy of inputs for numpy execution to avoid modification
        inputs_ref = [x.copy() if isinstance(x, np.ndarray) else x for x in inputs]
        for _ in range(args.n_runs):
            start = time.time()
            # Execute the original python function (undecorated)
            kernel_func(*inputs_ref)
            end = time.time()
            print(f"Numpy execution time: {end - start:.6f} seconds")

    if args.docc:
        # Create a copy of inputs for docc execution
        inputs_docc = [x.copy() if isinstance(x, np.ndarray) else x for x in inputs]
        # Execute the decorated function
        kernel_with_target = docc.program(
            kernel_func,
            target=args.target,
        )

        times = []
        start = time.time()
        kernel_with_target(*inputs_docc)
        end = time.time()
        times.append(end - start)
        print(f"Docc execution time: {end - start:.6f} seconds")

        for _ in range(args.n_runs):
            start = time.time()
            kernel_with_target(*inputs_docc)
            end = time.time()
            times.append(end - start)
            print(f"Docc execution time (cached): {end - start:.6f} seconds")

        # print(f"Average Docc execution time over {N+1} runs: {np.mean(times):.6f} seconds")
        # print(f"Average Docc execution time (cached) over {N} runs: {np.mean(times[1:]):.6f} seconds")


def run_pytest(
    initialize_func,
    kernel_func,
    parameters,
    target="none",
    verifier: SDFGVerification = None,
):
    # Use the smallest size for testing
    size = "S"
    if "S" not in parameters:
        size = list(parameters.keys())[0]

    params = parameters[size]
    inputs = initialize_func(**params)

    if not isinstance(inputs, tuple):
        inputs = (inputs,)

    # Run Python version
    inputs_ref = [x.copy() if isinstance(x, np.ndarray) else x for x in inputs]
    res_ref = kernel_func(*inputs_ref)

    # Run Docc version
    inputs_docc = [x.copy() if isinstance(x, np.ndarray) else x for x in inputs]

    kernel_with_target = docc.program(
        kernel_func,
        target=target,
    )
    res_docc = kernel_with_target(*inputs_docc)

    sdfg = kernel_with_target.last_sdfg
    stats = sdfg.loop_report()
    print(stats)  # {'FOR': 5, 'MAP': 2, 'CPU': 2, ...}
    assert stats is not None, "No stats found in SDFG."
    verifier.verify(stats)

    # Validate return values if they exist
    if res_ref is not None:
        if isinstance(res_ref, tuple):
            for i in range(len(res_ref)):
                np.testing.assert_allclose(
                    res_docc[i], res_ref[i], rtol=1e-5, atol=1e-8
                )
        else:
            np.testing.assert_allclose(res_docc, res_ref, rtol=1e-5, atol=1e-8)

    # Validate arguments (in-place modifications)
    for i in range(len(inputs)):
        if isinstance(inputs[i], np.ndarray):
            np.testing.assert_allclose(
                inputs_docc[i], inputs_ref[i], rtol=1e-5, atol=1e-8
            )
