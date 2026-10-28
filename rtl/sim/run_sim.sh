#!/usr/bin/env bash
# run_sim.sh - compile and run the testbenches with Icarus Verilog.
#   bash run_sim.sh               everything: requantisers, engine (4 builds), whole UART path (2 builds)
#   bash run_sim.sh requant       tb_requant  (serial and pipelined requantiser vs the Python reference)
#   bash run_sim.sh engine        tb_engine   (LANES=${LANES:-1} RQ_MODE=${RQ_MODE:-0} QUEUE=${QUEUE:-0} PERF=${PERF:-0}, uses engine_vectors.hex)
#   bash run_sim.sh bus          tb_bus (BUS=1 SPI, 2 I2C, 3 AXI4-Lite; same options, uses system_vectors.hex)
#   bash run_sim.sh uart          tb_uart_top (same options, uses system_vectors.hex)
# Vectors: python3 gen_requant_vectors.py; python3 gen_engine_vectors.py [seed] [jobs] [lanes] [rq_mode] [queue] [perf];
#          python3 gen_system_vectors.py [model.tflite] [runs] [lanes] [rq_mode] [queue] [perf]
set -e
cd "$(dirname "$0")"
R=..
# Engine sources (shared by both benches).
ENG="$R/tmf_engine.v $R/tmf_lane.v $R/tmf_ram.v $R/tmf_rq.v $R/tmf_requant.v $R/tmf_requant_pipe.v"
# Build options for the single-bench runs.
LANES=${LANES:-1}
RQ_MODE=${RQ_MODE:-0}
QUEUE=${QUEUE:-0}
PERF=${PERF:-0}
WIN=${WIN:-0}
FUSE=${FUSE:-0}
W4=${W4:-0}
CB=${CB:-0}

# Run one bench: name, lanes, rq_mode, queue, perf.
run_bench() {
    local t=$1 l=$2 q=$3 u=$4 f=$5 w=${6:-0} x=${7:-0} z=${8:-0} c=${9:-0}
    case "$t" in
        requant)
            python3 gen_requant_vectors.py 1 20000 > /dev/null
            iverilog -g2012 -o tb_requant.vvp tb_requant.v $R/tmf_rq.v $R/tmf_requant.v $R/tmf_requant_pipe.v
            vvp -n tb_requant.vvp | grep -v readmemh ;;
        engine)
            iverilog -g2012 -P tb_engine.LANES=$l -P tb_engine.RQ_MODE=$q -P tb_engine.HAS_QUEUE=$u -P tb_engine.HAS_PERF=$f -P tb_engine.HAS_WIN=$w -P tb_engine.HAS_FUSE=$x -P tb_engine.HAS_W4=$z -P tb_engine.HAS_CB=$c -o tb_engine.vvp tb_engine.v $ENG
            vvp -n tb_engine.vvp | grep -v readmemh ;;
        uart)
            iverilog -g2012 -P tb_uart_top.LANES=$l -P tb_uart_top.RQ_MODE=$q -P tb_uart_top.HAS_QUEUE=$u -P tb_uart_top.HAS_PERF=$f -P tb_uart_top.HAS_WIN=$w -P tb_uart_top.HAS_FUSE=$x -P tb_uart_top.HAS_CB=$c -P tb_uart_top.HAS_W4=$z -o tb_uart_top.vvp tb_uart_top.v \
                $R/tinyml_uart_top.v $R/uart_bus.v $R/uart_rx.v $R/uart_tx.v $ENG
            vvp -n tb_uart_top.vvp | grep -v readmemh ;;
        bus)
            iverilog -g2012 -P tb_bus.BUS=${BUS:-1} -P tb_bus.LANES=$l -P tb_bus.RQ_MODE=$q -P tb_bus.HAS_QUEUE=$u -P tb_bus.HAS_PERF=$f -P tb_bus.HAS_WIN=$w -P tb_bus.HAS_FUSE=$x -P tb_bus.HAS_W4=$z -P tb_bus.HAS_CB=$c -o tb_bus.vvp tb_bus.v \
                $R/tinyml_spi_top.v $R/tinyml_i2c_top.v $R/tinyml_axil_top.v $R/spi_bus.v $R/i2c_bus.v $R/axil_bus.v $ENG
            vvp -n tb_bus.vvp | grep -v readmemh ;;
        *) echo "unknown bench $t (requant | engine | uart | bus)"; exit 1 ;;
    esac
}

if [ $# -eq 0 ]; then
    # Everything, with fresh vectors.
    run_bench requant 1 0 0 0
    # lanes rq_mode queue perf win fuse w4 cb
    for cfg in "1 0 0 0 0 0 0 0" "1 1 1 1 1 1 1 1" "4 1 1 1 1 1 1 1" "8 0 0 1 1 0 1 1" "2 1 1 0 0 1 0 1" "1 0 0 0 0 0 1 0" "2 0 0 0 0 0 0 1"; do
        set -- $cfg
        python3 gen_engine_vectors.py 1 80 $1 $2 $3 $4 $5 $6 $7 $8 > /dev/null
        run_bench engine $1 $2 $3 $4 $5 $6 $7 $8
    done
    for cfg in "1 0 0 0 0 0 0" "2 1 1 1 1 1 0"; do
        set -- $cfg
        python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 2 $1 $2 $3 $4 $5 $6 $7 > /dev/null
        run_bench uart $1 $2 $3 $4 $5 $6 $7
    done
    # A 4-bit-weight model through the UART bridge.
    python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 2 2 1 1 0 1 1 1 1 > /dev/null
    run_bench uart 2 1 1 0 1 1 1
    # A codebook model through the UART bridge.
    python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 2 2 1 1 0 1 1 0 0 1 1 > /dev/null
    run_bench uart 2 1 1 0 1 1 0 1
    # SPI, I2C and AXI4-Lite bridges: a whole inference through each (plain and fully featured engines).
    python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 1 1 0 0 0 0 0 0 > /dev/null
    for b in 1 2 3; do BUS=$b run_bench bus 1 0 0 0 0 0 0 0; done
    python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 1 2 1 1 0 1 1 0 > /dev/null
    for b in 1 2 3; do BUS=$b run_bench bus 2 1 1 0 1 1 0 0; done
else
    for t in "$@"; do run_bench "$t" "$LANES" "$RQ_MODE" "$QUEUE" "$PERF" "$WIN" "$FUSE" "$W4" "$CB"; done
fi
