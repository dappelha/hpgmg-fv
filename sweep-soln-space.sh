#!/bin/bash -e

# host level threshold: number of grid elements
export THRESHOLD=100

for (( i = 1; i < 9; i++ )); do
    let thresh=8**$i+1;
    #echo $thresh
    export THRESHOLD=$thresh
    echo "Running threshold ${thresh}"
    ./sweeper.sh > "${thresh}.out"
    sleep 5
done

#./sweeper.sh
