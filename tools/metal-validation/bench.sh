#!/bin/zsh
# Solo benchmark sweep: CPU (28 threads) vs Metal GPU.
set -u
export OPENMC_CROSS_SECTIONS=/Users/davidallonby/Pooper/openmc-data/endfb-viii.0-hdf5/cross_sections.xml
VENV=/Users/davidallonby/Pooper/openmc-venv/bin
BIN=/Users/davidallonby/Pooper/openmc-metal/build/bin/openmc
MODELS=/Users/davidallonby/Pooper/metal-work/models
OUT=/Users/davidallonby/Pooper/metal-work/bench_results.txt
: > $OUT

run_case() {
  local tag=$1 dir=$2 gpu=$3
  cd $dir
  if [[ $gpu == 1 ]]; then
    OPENMC_GPU=1 $BIN > bench.log 2>&1
  else
    $BIN > bench.log 2>&1
  fi
  local rate=$(grep "Calculation Rate (active)" bench.log | awk '{print $5}')
  local k=$(grep "Combined k-effective" bench.log | awk '{print $4, $6}')
  local lost=$(grep -c "lost" bench.log)
  echo "$tag rate=$rate k=$k lost_warns=$lost" >> $OUT
  echo "$tag rate=$rate"
}

B=/Users/davidallonby/Pooper/metal-work/bench
mkdir -p $B/mg $B/pin $B/god

# --- MG lattice ---
for np in 10000 100000 1000000; do
  cd $B/mg && $VENV/python $MODELS/mg_pincell.py --particles $np --batches 40 --inactive 10 --tallies > /dev/null 2>&1
  run_case "mg-cpu-$np" $B/mg 0
  run_case "mg-gpu-$np" $B/mg 1
done

# --- CE pincell (no S(a,b)) ---
cd $B/pin && $VENV/python $MODELS/pincell.py --particles 20000 --batches 40 --inactive 10 --no-sab --tallies > /dev/null 2>&1
run_case "pin-cpu-20k" $B/pin 0
run_case "pin-gpu-20k" $B/pin 1
for np in 100000 500000; do
  cd $B/pin && $VENV/python $MODELS/pincell.py --particles $np --batches 40 --inactive 10 --no-sab --tallies > /dev/null 2>&1
  run_case "pin-gpu-$np" $B/pin 1
done

# --- CE Godiva ---
cd $B/god && $VENV/python $MODELS/godiva.py --particles 100000 --batches 40 --inactive 10 --tallies > /dev/null 2>&1
run_case "god-cpu-100k" $B/god 0
run_case "god-gpu-100k" $B/god 1
cd $B/god && $VENV/python $MODELS/godiva.py --particles 1000000 --batches 40 --inactive 10 --tallies > /dev/null 2>&1
run_case "god-gpu-1M" $B/god 1

echo BENCH_DONE >> $OUT
