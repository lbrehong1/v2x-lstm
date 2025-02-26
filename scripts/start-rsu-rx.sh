#!/bin/bash

# Check if exactly two arguments are provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <LOG_LLC> <LOG_PC5>"
    exit 1
fi

# Assign arguments to variables
LOG_LLC=$1
LOG_PC5=$2

# Function to clean up background processes on exit
cleanup() {
    echo "Stopping processes..."
    pkill -P $$  # Kill all child processes
    exit 0
}

# Trap SIGINT (Ctrl-C) and call cleanup
trap cleanup SIGINT

# Launch commands in the background with provided parameters
llc -i1 test-rx -c 174 -l -p 1 -f "$LOG_LLC" &
acme -R -V -m "$LOG_PC5" &

# Wait to keep the script running until interrupted
wait

