#!/usr/bin/env bash
# run_sim.sh - compile and run the testbenches with Icarus Verilog.
#   bash run_sim.sh               everything: requantisers, engine (4 builds), whole UART path (2 builds)
#   bash run_sim.sh requant       tb_requant  (serial and pipelined requantiser vs the Python reference)
#   bash run_sim.sh engine        tb_engine   (LANES=${LANES:-1} RQ_MODE=${RQ_MODE:-0} QUEUE=${QUEUE:-0} PERF=${PERF:-0}, uses engine_vectors.hex)
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

# Run one bench: name, lanes, rq_mode, queue, perf.
run_bench() {
    local t=$1 l=$2 q=$3 u=$4 f=$5
    case "$t" in
        requant)
            python3 gen_requant_vectors.py 1 20000 > /dev/null
            iverilog -g2012 -o tb_requant.vvp tb_requant.v $R/tmf_rq.v $R/tmf_requant.v $R/tmf_requant_pipe.v
            vvp -n tb_requant.vvp | grep -v readmemh ;;
        engine)
            iverilog -g2012 -P tb_engine.LANES=$l -P tb_engine.RQ_MODE=$q -P tb_engine.HAS_QUEUE=$u -P tb_engine.HAS_PERF=$f -o tb_engine.vvp tb_engine.v $ENG
            vvp -n tb_engine.vvp | grep -v readmemh ;;
        uart)
            iverilog -g2012 -P tb_uart_top.LANES=$l -P tb_uart_top.RQ_MODE=$q -P tb_uart_top.HAS_QUEUE=$u -P tb_uart_top.HAS_PERF=$f -o tb_uart_top.vvp tb_uart_top.v \
                $R/tinyml_uart_top.v $R/uart_bus.v $R/uart_rx.v $R/uart_tx.v $ENG
            vvp -n tb_uart_top.vvp | grep -v readmemh ;;
        *) echo "unknown bench $t (requant | engine | uart)"; exit 1 ;;
    esac
}

if [ $# -eq 0 ]; then
    # Everything, with fresh vectors.
    run_bench requant 1 0 0 0
    for cfg in "1 0 0 0" "1 1 1 1" "4 1 1 1" "8 0 0 1"; do
        set -- $cfg
        python3 gen_engine_vectors.py 1 80 $1 $2 $3 $4 > /dev/null
        run_bench engine $1 $2 $3 $4
    done
    for cfg in "1 0 0 0" "2 1 1 1"; do
        set -- $cfg
        python3 gen_system_vectors.py ../../firmware/tests/models/m_cnn.tflite 2 $1 $2 $3 $4 > /dev/null
        run_bench uart $1 $2 $3 $4
    done
else
    for t in "$@"; do run_bench "$t" "$LANES" "$RQ_MODE" "$QUEUE" "$PERF"; done
fi
