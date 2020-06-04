#!/bin/sh
docker run --rm -e "VFC_BACKENDS=$3" --mount "type=bind,source=$(pwd),target=/workdir" -i gkiar/fuzzy:$2 python3.6 main.py evaluate $1 "$2-$3"
