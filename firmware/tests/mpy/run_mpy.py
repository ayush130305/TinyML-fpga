# run_mpy.py - under MicroPython: run a .tmf through the FPGA driver on the register model ("sim":
# HwEngine + SimBus, the full hardware code path) and on the software engine ("cpu"), and compare
# with expected outputs computed on the PC.
# usage: micropython run_mpy.py <repo root> <model.tmf> <inputs.bin> <expected.bin> <n> <in_len> <out_len>
import sys
root, model, inp, exp, n, il, ol = sys.argv[1:8]
n, il, ol = int(n), int(il), int(ol)
sys.path.insert(0, root)
import tinyml_fpga
with open(inp, "rb") as f:
    X = f.read()
with open(exp, "rb") as f:
    E = f.read()
ok = True
for be in ("sim", "cpu"):
    m = tinyml_fpga.load(model, backend=be)
    for i in range(n):
        y = m.invoke(bytearray(X[i * il:(i + 1) * il]))
        if isinstance(y, list):
            print("unexpected float output")
            ok = False
            break
        if bytes(y) != E[i * ol:(i + 1) * ol]:
            print("MISMATCH", be, "sample", i)
            ok = False
    m.close()
print("MPY PASS" if ok else "MPY FAIL")
