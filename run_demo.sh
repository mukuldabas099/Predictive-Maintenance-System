#!/bin/sh
cd "$(dirname "$0")/src" && TF_CPP_MIN_LOG_LEVEL=2 python3 demo.py "$@"
