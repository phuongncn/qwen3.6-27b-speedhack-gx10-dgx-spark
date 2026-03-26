#!/bin/bash
# TurboQuant quality + speed gate
# Tests PPL and decode speed for turbo3/turbo4 vs q8_0 baseline
#
# Usage: bash scripts/turbo-quality-gate.sh
# Override paths: LLAMA=~/path/to/bin MODEL=~/model.gguf bash scripts/turbo-quality-gate.sh
#
# Requires: llama-perplexity (build with: cmake --build build -t llama-perplexity)

set -euo pipefail

LLAMA=${LLAMA:-~/llama-cuda-turbo/build/bin}
MODEL=${MODEL:-~/Qwen3.5-27B-heretic.Q6_K.gguf}
WIKI=${WIKI:-wikitext-2-raw/wiki.test.raw}
THREADS=${THREADS:-10}
NGL=${NGL:-99}
PPL_THRESHOLD=${PPL_THRESHOLD:-1.05}    # turbo PPL must be < q8_0 * this
SPEED_THRESHOLD=${SPEED_THRESHOLD:-0.90} # turbo tok/s must be > q8_0 * this

if [ ! -f "$WIKI" ]; then
    echo "Downloading wikitext-2..."
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    bash "$SCRIPT_DIR/get-wikitext-2.sh"
fi

if [ ! -f "$LLAMA/llama-perplexity" ]; then
    echo "ERROR: $LLAMA/llama-perplexity not found"
    echo "Build it with: cmake --build build -t llama-perplexity"
    exit 1
fi

FAIL=0

run_ppl() {
    local label=$1 ctk=$2 ctv=$3 ctx=$4 chunks=$5
    $LLAMA/llama-perplexity -m "$MODEL" -f "$WIKI" \
        -c "$ctx" -ctk "$ctk" -ctv "$ctv" -fa -t "$THREADS" \
        --chunks "$chunks" -ngl "$NGL" 2>&1
}

extract_ppl() {
    grep -oE 'PPL = [0-9.]+' | tail -1 | grep -oE '[0-9.]+'
}

extract_speed() {
    grep "eval time" | head -1 | grep -oE '[0-9.]+ tokens per second' | grep -oE '[0-9.]+'
}

echo "========================================"
echo "  TurboQuant Quality + Speed Gate"
echo "========================================"
echo "  Model: $(basename "$MODEL")"
echo "  Threads: $THREADS"
echo ""

# --- Test 1: Perplexity (2K context, 8 chunks) ---
echo "[1/3] Perplexity — q8_0 baseline (2K ctx, 8 chunks)..."
PPL_Q8=$(run_ppl "q8_0" q8_0 q8_0 2048 8 | extract_ppl)
echo "  q8_0 PPL = ${PPL_Q8:-FAILED}"

echo ""
echo "[2/3] Perplexity — turbo3 + turbo4 (2K ctx, 8 chunks)..."
PPL_T3=$(run_ppl "turbo3" turbo3 turbo3 2048 8 | extract_ppl)
PPL_T4=$(run_ppl "turbo4" turbo4 turbo4 2048 8 | extract_ppl)
echo "  turbo3 PPL = ${PPL_T3:-FAILED}"
echo "  turbo4 PPL = ${PPL_T4:-FAILED}"

if [ -n "$PPL_Q8" ]; then
    MAX_PPL=$(echo "$PPL_Q8 * $PPL_THRESHOLD" | bc)
    for name_ppl in "turbo3:$PPL_T3" "turbo4:$PPL_T4"; do
        name=${name_ppl%%:*}
        ppl=${name_ppl##*:}
        if [ -z "$ppl" ]; then
            echo "  FAIL: $name — could not measure PPL"
            FAIL=1
        elif [ "$(echo "$ppl < $MAX_PPL" | bc)" -eq 1 ]; then
            echo "  PASS: $name PPL $ppl < $MAX_PPL (within ${PPL_THRESHOLD}x of q8_0)"
        else
            echo "  FAIL: $name PPL $ppl > $MAX_PPL (exceeds ${PPL_THRESHOLD}x threshold)"
            FAIL=1
        fi
    done
else
    echo "  FAIL: q8_0 baseline failed — cannot compare"
    FAIL=1
fi

# --- Test 2: Context Scaling (decode speed at 4K, 16K, 32K) ---
echo ""
echo "[3/3] Context scaling — decode speed at multiple context lengths..."
echo ""
printf "  %-8s %10s %10s %10s %10s %10s\n" "CTX" "q8_0" "turbo3" "turbo4" "t3/q8" "t4/q8"
printf "  %-8s %10s %10s %10s %10s %10s\n" "--------" "----------" "----------" "----------" "----------" "----------"

for CTX in 4096 16384 32768; do
    CHUNKS=$((CTX / 1024))
    Q8_TPS=$(run_ppl "q8" q8_0 q8_0 "$CTX" "$CHUNKS" | extract_speed)
    T3_TPS=$(run_ppl "t3" turbo3 turbo3 "$CTX" "$CHUNKS" | extract_speed)
    T4_TPS=$(run_ppl "t4" turbo4 turbo4 "$CTX" "$CHUNKS" | extract_speed)

    if [ -n "$Q8_TPS" ] && [ -n "$T3_TPS" ]; then
        R3=$(echo "scale=4; $T3_TPS / $Q8_TPS" | bc)
    else
        R3="N/A"
    fi
    if [ -n "$Q8_TPS" ] && [ -n "$T4_TPS" ]; then
        R4=$(echo "scale=4; $T4_TPS / $Q8_TPS" | bc)
    else
        R4="N/A"
    fi

    printf "  %-8s %8s %8s %8s %10s %10s\n" \
        "${CTX}" "${Q8_TPS:-N/A}" "${T3_TPS:-N/A}" "${T4_TPS:-N/A}" "$R3" "$R4"

    # Check speed threshold at 32K (the hardest test)
    if [ "$CTX" -eq 32768 ]; then
        for name_ratio in "turbo3:$R3" "turbo4:$R4"; do
            name=${name_ratio%%:*}
            ratio=${name_ratio##*:}
            if [ "$ratio" = "N/A" ]; then
                echo "  FAIL: $name — could not measure speed at 32K"
                FAIL=1
            elif [ "$(echo "$ratio > $SPEED_THRESHOLD" | bc)" -eq 1 ]; then
                echo "  PASS: $name ${ratio}x at 32K (> ${SPEED_THRESHOLD} threshold)"
            else
                echo "  FAIL: $name ${ratio}x at 32K (< ${SPEED_THRESHOLD} threshold)"
                FAIL=1
            fi
        done
    fi
done

# --- Summary ---
echo ""
echo "========================================"
if [ "$FAIL" -eq 0 ]; then
    echo "  ALL CHECKS PASSED"
else
    echo "  SOME CHECKS FAILED"
fi
echo "========================================"
exit $FAIL
