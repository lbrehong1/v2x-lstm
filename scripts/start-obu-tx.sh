#!/bin/bash

# Function to clean up background processes on exit
cleanup() {
    echo "Stopping processes..."
    pkill -P $$  # Kill all child processes
    exit 0
}

# Trap SIGINT (Ctrl-C) and call cleanup
trap cleanup SIGINT

# Launch commands in the background
llc -i1 test-tx -c 174 -l 1000 -n 100000 -r 50 -g time &
acme -g -I 100 -P10 -U 1 &

# Wait to keep the script running until interrupted
wait
