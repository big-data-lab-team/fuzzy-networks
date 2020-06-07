#!/bin/sh

# Command arguments:
# $1 Name of experiment folder in results containing the pre-trained neural network
# $2 tag of the fuzzy docker image to use (python, python-numpy, etc.). It corresponds to the use of MCA in different parts of the stack
# $3 verificarlo backend configuration:  containing the precision to use
# $4 type of neural network: mlp or cnn 
docker run --rm -e "VFC_BACKENDS=$3" --mount "type=bind,source=$(pwd),target=/workdir" -i gkiar/fuzzy:$2 python3.6 main.py $4 evaluate $1 "$2-$3"
