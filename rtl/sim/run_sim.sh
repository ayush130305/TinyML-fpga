#!/usr/bin/env bash
# run_sim.sh - compile and run the testbenches with Icarus Verilog.
#   bash run_sim.sh            both (generates fresh vectors first)
#   bash run_sim.sh engine     tb_engine only   (uses engine_vectors.hex)
#   bash run_sim.sh uart       tb_uart_top only (uses system_vectors.hex)
# Vectors: python3 gen_engine_vectors.py [seed] [jobs]; python3 gen_system_vectors.py [model.tflite] [runs]
set -e
cd "$(dirname "$0")"
R=..
want=("$@")
# No arguments: regenerate the vectors and run everything.
if [ ${#want[@]} -eq 0 ]; then
    python3 gen_engine_vectors.py 1 80
    python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 2
    want=(engine uart)
fi
for t in "${want[@]}"; do
    case "$t" in
        engine)
            iverilog -g2012 -o tb_engine.vvp tb_engine.v $R/tmf_engine.v $R/tmf_ram.v $R/tmf_requant.v
            vvp -n tb_engine.vvp | grep -v readmemh ;;
        uart)
            iverilog -g2012 -o tb_uart_top.vvp tb_uart_top.v $R/tinyml_uart_top.v \
                $R/uart_bus.v $R/uart_rx.v $R/uart_tx.v $R/tmf_engine.v $R/tmf_ram.v $R/tmf_requant.v
            vvp -n tb_uart_top.vvp | grep -v readmemh ;;
        *) echo "unknown bench $t (engine | uart)"; exit 1 ;;
    esac
done
