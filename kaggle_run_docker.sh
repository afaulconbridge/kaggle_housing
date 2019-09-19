#!/bin/bash
set -e

#run this script inside docker
docker-compose build

time docker-compose run --rm kaggle bash -c "time nice python3 kaggle.py"

tput bel