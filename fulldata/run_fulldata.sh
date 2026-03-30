#!/bin/bash

for mol in pdb_folder/*.pdb
do
    echo "Running with $mol"
    python fulldata_2L7B.py --mol1 "$mol"
done