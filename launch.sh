#!/bin/bash
script_name="server"
PID_FILE="${script_name}.pid"
LOG_FILE="${script_name}.log"
CMD="python -m funsound.web.server 8800"
> $LOG_FILE

start() {
    # Check if the process is already running
    if [ -f "$PID_FILE" ] && kill -0 $(cat "$PID_FILE") 2>/dev/null; then
        echo "Process is already running."
        exit 1
    fi
    
    # Start the process
    echo "Starting process..."
    nohup $CMD >> "$LOG_FILE" 2>&1 &
    
    # Save the PID
    echo $! > "$PID_FILE"
    echo "Process started with PID $(cat "$PID_FILE")"
}

stop() {
    # Check if PID file exists
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        # Kill the process
        if kill -0 $PID 2>/dev/null; then
            echo "Stopping process with PID $PID..."
            kill $PID
            rm "$PID_FILE"
            echo "Process stopped."
        else
            echo "Process with PID $PID not found."
            rm "$PID_FILE"
        fi
    else
        echo "PID file not found. Process may not be running."
    fi
}

case "$1" in
    -k)
        stop
        ;;
    *)
        start
        ;;
esac