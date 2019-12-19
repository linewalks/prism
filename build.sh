#!/usr/bin/env bash

dir=$(dirname "$0")/docker
docker build -t prism-tensorflow-1.12.0-devel-gpu-py3 -f $dir/Dockerfile $dir
