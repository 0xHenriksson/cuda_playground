#!/usr/bin/env bash

set -u

# Define the range of values for each parameter
BK_VALUES=(8 16 32 64)
TM_VALUES=(4 8 16 32)
TN_VALUES=(4 8 16 32)
BM_VALUES=(64 128 256)
BN_VALUES=(64 128 256)
NUM_THREADS_VALUES=(256)
MATRIX_SIZE=(128 256 512 1024 2048 4096)

cd "$(dirname "$0")"
cd "../build"

DRIVER="../src/driver.cu"
KERNEL="../src/kernels/sgemm_v09.cuh"
OUTPUT="../benchmark_results/kernel_9_autotune_results.csv"



# Clear the output file
echo "" > $OUTPUT
# add headers
echo "BK,TM,TN,BM,BN,NUM_THREADS,SIZE,AVG_TIME_ELAPSE,GFLOPS" > $OUTPUT
# set GPU to use
# should be 0 on the jetson
export DEVICE="0"

TOTAL_CONFIGS="$(( ${#NUM_THREADS_VALUES[@]} * ${#BK_VALUES[@]} * ${#TM_VALUES[@]} * ${#TN_VALUES[@]} * ${#BM_VALUES[@]} * ${#BN_VALUES[@]} * ${#MATRIX_SIZE[@]} ))"
CONFIG_NUM=0

# Loop through all combinations of parameters
for bk in ${BK_VALUES[@]}; do
  for tm in ${TM_VALUES[@]}; do
    for tn in ${TN_VALUES[@]}; do
      for bm in ${BM_VALUES[@]}; do
        for bn in ${BN_VALUES[@]}; do
          for nt in ${NUM_THREADS_VALUES[@]}; do
            for size in ${MATRIX_SIZE[@]}; do
              echo ""
              CONFIG_NUM=$(( $CONFIG_NUM + 1 ))

              # skip configurations that don't fullfil preconditions
              config="BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NT=$nt"
              if [[ $(( ($nt * 4) % bk )) -ne 0 ]]; then
                echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BK = $(( ($nt * 4) % bk )) != 0))"
                continue
              fi
              if [[ $(( ($nt * 4) % bn )) -ne 0 ]]; then
                echo "VECTORIZE: Skipping $config because (NUM_THREADS * 4) % BN = $(( ($nt * 4) % bn )) != 0))"
                continue
              fi
              if [[ $(( $bn % (16 * $tn ) )) -ne 0 ]]; then
                echo "QUANTIZATION: Skipping $config because BN % (16 * TN) = $(( $bn % (16 * $tn ) )) != 0))"
                continue
              fi
              if [[ $(( $bm % (16 * $tm ) )) -ne 0 ]]; then
                echo "QUANTIZATION: Skipping $config because BM % (16 * TM) = $(( $bm % (16 * $tm ) )) != 0))"
                continue
              fi
              if [[ $(( ($bm * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
                echo "VECTORIZE: Skipping $config because (BM * BK) % (4 * NUM_THREADS) = $(( ($bm * $bk) % ( 4 * 256 ) )) != 0))"
                continue
              fi
              if [[ $(( ($bn * $bk) % ( 4 * $nt ) )) -ne 0 ]]; then
                echo "VECTORIZE: Skipping $config because (BN * BK) % (4 * NUM_THREADS) = $(( ($bn * $bk) % ( 4 * 256 ) )) != 0))"
                continue
              fi

              # Update the parameters in the source code
              sed -i "s/const uint K9_BK = .*/const uint K9_BK = $bk;/" $DRIVER
              sed -i "s/const uint K9_TM = .*/const uint K9_TM = $tm;/" $DRIVER
              sed -i "s/const uint K9_TN = .*/const uint K9_TN = $tn;/" $DRIVER
              sed -i "s/const uint K9_BM = .*/const uint K9_BM = $bm;/" $DRIVER
              sed -i "s/const uint K9_BN = .*/const uint K9_BN = $bn;/" $DRIVER
              sed -i "s/const int K9_NUM_THREADS = .*/const int K9_NUM_THREADS = $nt;/" $KERNEL
              
              # Rebuild the program
              make 

              echo "($CONFIG_NUM/$TOTAL_CONFIGS): BK=$bk TM=$tm TN=$tn BM=$bm BN=$bn NUM_THREADS=$nt" # |& tee -a $OUTPUT
              # Output the current configuration parameters to the CSV file
              echo -n "$bk,$tm,$tn,$bm,$bn,$nt,$size," >> $OUTPUT
              # Run the benchmark and get the result
              # Kill the program after 15 seconds if it doesn't finish
              # timeout -v 15 ./sgemm 9 | tee -a $OUTPUT
              timeout -v 30 ./sgemm 9 $size $OUTPUT
              if [ $? -eq 124 ]; then
                echo "TIMEOUT, TIMEOUT," >> $OUTPUT
              fi
            done
          done
        done
      done
    done
  done
done