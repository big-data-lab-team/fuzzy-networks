#!/bin/sh
docker run --rm --mount "type=bind,source=$(pwd),target=/workdir" -ti gkiar/fuzzy:python-numpy python3.6 evaluate.py $1
