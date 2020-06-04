#!/bin/sh
docker run --rm -e "VFC_BACKENDS=$3" --mount "type=bind,source=$(pwd),target=/workdir" -i gkiar/fuzzy:$2 python3.6 evaluate.py $1 "$2-$3"
