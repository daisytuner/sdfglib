import argparse
import time
import numpy as np
import pytest
import docc


def run_benchmark(initialize_func, kernel_func, parameters, name, args=None):
    if args is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--size", type=str, default="paper")
        parser.add_argument("--docc", action="store_true")
        parser.add_argument("--numpy", action="store_true")
        parser.add_argument("--target", type=str, default="none")
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
        kernel_func(*inputs_ref)
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
        kernel_with_target(*inputs_docc)
        start = time.time()
        kernel_with_target(*inputs_docc)
        end = time.time()
        print(f"Docc execution time: {end - start:.6f} seconds")


def run_pytest(initialize_func, kernel_func, parameters, target="none"):
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
