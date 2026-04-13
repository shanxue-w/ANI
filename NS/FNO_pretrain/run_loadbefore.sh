#!/bin/bash

# Define the full training set size
TOTAL_TRAIN=4000

# Define the percentages to test
PERCENTAGES=(1 5 10 20 50 100 200 300)
# PERCENTAGES=(200 300)

echo "Starting batch experiments..."

for PCT in "${PERCENTAGES[@]}"
do
    # Calculate limits (using bash arithmetic)
    # train_limit = 4000 * PCT / 100
    TRAIN_LIMIT=$(( TOTAL_TRAIN * PCT / 100 ))
    # val_limit = train_limit / 4
    VAL_LIMIT=$(( TRAIN_LIMIT / 4 ))

    LOG_FILE="log_${TRAIN_LIMIT}_loadbefore.txt"

    echo "------------------------------------------------"
    echo "Running Experiment: ${PCT}% Data"
    echo "Train Limit: ${TRAIN_LIMIT}, Val Limit: ${VAL_LIMIT}"
    echo "Logging to: ${LOG_FILE}"

    # Execution using 'begin/end' block for fish-like redirection or standard bash
    # Capturing stdout, stderr, and 'time' results into the log file
    {
        time python fine_loadbefore.py --train_limit $TRAIN_LIMIT --val_limit $VAL_LIMIT
    } &> "$LOG_FILE"

    echo "Done. Result saved in $LOG_FILE"
done

echo "All experiments completed!"