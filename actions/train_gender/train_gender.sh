#!/usr/bin/env bash

./upload_files.py

python ../training/__main__.py parameters_all.json
