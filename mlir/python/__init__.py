from torch_mlir import fx
import subprocess

def optimize_model(model, example_input):
    m = fx.export_and_import(model, example_input, output_type=fx.OutputType.LINALG_ON_TENSORS)
    result_opt = subprocess.run(["../../build/sdfglib/mlir/tools/sdfg-opt/sdfg-opt", "--convert-to-sdfg"], input=str(m).encode(), capture_output=True)
    result_translate = subprocess.run(["../../build/sdfglib/mlir/tools/sdfg-translate/sdfg-translate", "--mlir-to-sdfg"], input=result_opt.stdout, capture_output=True)
    print(result_translate.stdout.decode())