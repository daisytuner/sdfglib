import os
import subprocess
import pytest
import json

from pathlib import Path

@pytest.mark.parametrize(
    "event",
    [
        pytest.param("DURATION_TIME"),
        pytest.param("perf::BRANCHES,perf::CYCLES"),
    ],
)
def test_instrumentation(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_test.c"
    output_path = workdir / "instrumentation_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    assert process.returncode == 0

    (workdir / "data.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    print(stdout)
    print(stderr)
    assert process.returncode == 0

    event_names = event.split(",")

    result = json.load(open(workdir / "data.json"))
    events = result["traceEvents"]
    for i in range(len(events)):
        assert events[i]["name"] == "instrumentation_test_main"
        assert events[i]["cat"] == "DAISY"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["file"] == "instrumentation_test.c"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["line_begin"] == 18
        assert events[i]["args"]["line_end"] == 31
        assert events[i]["args"]["column_begin"] == 4
        assert events[i]["args"]["column_end"] == 5
        for event_name in event_names:
            assert event_name in events[i]["args"]

@pytest.mark.parametrize(
    "event",
    [
        pytest.param("DURATION_TIME"),
        pytest.param("cuda:::dram__bytes,cuda:::sm__inst_executed_pipe_fma"),
    ],
)
def test_instrumentation_cuda(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_cuda_test.cu"
    output_path = workdir / "instrumentation_cuda_test.out"
    cmd = [
        "nvcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    (workdir / "data.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    assert process.returncode == 0

    event_names = event.split(",")

    result = json.load(open(workdir / "data.json"))
    events = result["traceEvents"]
    for i in range(len(events)):
        assert events[i]["name"] == "instrumentation_cuda_test_main"
        assert events[i]["cat"] == "DAISY"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["file"] == "instrumentation_cuda_test.cu"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["line_begin"] == 18
        assert events[i]["args"]["line_end"] == 31
        assert events[i]["args"]["column_begin"] == 4
        assert events[i]["args"]["column_end"] == 5
        for event_name in event_names:
            assert event_name in events[i]["args"]
