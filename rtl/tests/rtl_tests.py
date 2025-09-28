import os
import subprocess
import pytest
import json

from pathlib import Path

@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
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
        "-larg_capture_io",
        "-lstdc++"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cpu.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cpu.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    event_names = []
    if event:
        event_names = event.split(",")

    result = json.load(open(workdir / "data_cpu.json"))
    events = result["traceEvents"]
    assert(len(events) > 0)

    print(events)
    for i in range(len(events)):
        assert events[i]["name"].startswith("main")
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["module"] == "instrumentation_test.c"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] == 18
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] == 31
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] == 4
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] == 5
        assert events[i]["args"]["loopnest_index"] == 0
        for event_name in event_names:
            assert event_name in events[i]["args"]["metrics"]
            assert events[i]["args"]["metrics"][event_name] > 0

@pytest.mark.parametrize(
    "event",
    [
        pytest.param("perf::BRANCHES,perf::CYCLES"),
    ],
)
def test_instrumentation_aggregate(event):
    workdir = Path(__file__).parent / "applications"

    benchmark_path = workdir / "instrumentation_test.c"
    output_path = workdir / "instrumentation_test.out"
    cmd = [
        "gcc",
        str(benchmark_path),
        "-o",
        str(output_path),
        "-ldaisy_rtl",
        "-larg_capture_io",
        "-lstdc++"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cpu.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cpu.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = "aggregate"

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    event_names = []
    if event:
        event_names = event.split(",")

    result = json.load(open(workdir / "data_cpu.json"))
    events = result["traceEvents"]
    assert(len(events) > 0)

    print(events)
    for i in range(len(events)):
        assert events[i]["name"].startswith("main")
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["module"] == "instrumentation_test.c"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] == 18
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] == 31
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] == 4
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] == 5
        assert events[i]["args"]["loopnest_index"] == 0
        for event_name in event_names:
            assert event_name in events[i]["args"]["metrics"]
            assert events[i]["args"]["metrics"][event_name]["mean"] > 0
            assert events[i]["args"]["metrics"][event_name]["min"] > 0
            assert events[i]["args"]["metrics"][event_name]["max"] > 0
            assert events[i]["args"]["metrics"][event_name]["variance"] >= 0

@pytest.mark.parametrize(
    "event",
    [
        pytest.param(""),
        pytest.param("nvml:::NVIDIA_GeForce_RTX_5060_Ti:device_0:gpu_utilization,nvml:::NVIDIA_GeForce_RTX_5060_Ti:device_0:memory_utilization"),
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
        "-larg_capture_io",
        "-lstdc++"
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    (workdir / "data_cuda.json").unlink(missing_ok=True)

    ## THIS VERSION IS RUNNER-SPECIFIC and points to CI version of PAPI
    os.environ["__DAISY_PAPI_VERSION"] = "0x07020000"
    os.environ["__DAISY_INSTRUMENTATION_FILE"] = str(workdir / "data_cuda.json")
    os.environ["__DAISY_INSTRUMENTATION_EVENTS_CUDA"] = event
    os.environ["__DAISY_INSTRUMENTATION_MODE"] = ""

    # Run benchmark
    process = subprocess.Popen(
        [str(output_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stdout)
        print(stderr)
    assert process.returncode == 0

    event_names = []
    if event:
        event_names = event.split(",")

    result = json.load(open(workdir / "data_cuda.json"))
    events = result["traceEvents"]
    assert(len(events) > 0)

    print(events)
    for i in range(len(events)):
        assert events[i]["name"].startswith("main")
        assert events[i]["cat"] == "region,daisy"
        assert events[i]["ph"] == "X"
        assert events[i]["args"]["module"] == "instrumentation_cuda_test.cu"
        assert events[i]["args"]["function"] == "main"
        assert events[i]["args"]["source_ranges"][0]["from"]["line"] == 18
        assert events[i]["args"]["source_ranges"][0]["to"]["line"] == 31
        assert events[i]["args"]["source_ranges"][0]["from"]["col"] == 4
        assert events[i]["args"]["source_ranges"][0]["to"]["col"] == 5
        assert events[i]["args"]["loopnest_index"] == 0
        for event_name in event_names:
            assert event_name in events[i]["args"]["metrics"]
