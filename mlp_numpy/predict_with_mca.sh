#!/bin/sh
docker run --rm --mount "type=bind,source=$(pwd),target=/workdir" -i gkiar/fuzzy:$2 python3.6 evaluate.py $1 $2
