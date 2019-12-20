#!/usr/bin/env bash

dir=$(dirname "$0")/docker
docker build -t prism-tensorflow-1.14.0-gpu-py3 -f $dir/Dockerfile $dir
