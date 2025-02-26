#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <obu|rsu> <last 4 chars of MAC address>"
    exit 1
fi

# Assign arguments
TYPE=$1
HEX=$2

# Validate first argument
if [ "$TYPE" == "obu" ]; then
    KK="50"
elif [ "$TYPE" == "rsu" ]; then
    KK="30"
else
    echo "Error: First parameter must be 'obu' or 'rsu'."
    exit 1
fi

# Validate second argument (must be four hex digits)
if ! [[ "$HEX" =~ ^[0-9A-Fa-f]{4}$ ]]; then
    echo "Error: Second argument must be the last 4 chars of MAC address"
    exit 1
fi

# Extract WX and YZ from WXYZ
WX=${HEX:0:2}
YZ=${HEX:2:2}

# Execute commands
llc -i0 chconfig -s -c 172 -f 04:e5:48:$KK:$WX:$YZ
llc -i1 chconfig -s -c 174 -f 04:e5:48:$KK:$WX:$YZ
