#! /bin/bash

CODE="$(tr -dc '[:alnum:]' </dev/urandom | head -c 6)"
DOCKERZIP="prism_${CODE}.tar.gz"
echo $DOCKERZIP
docker save prism-tensorflow-1.14.0-gpu-py3 | gzip -c -v > $DOCKERZIP
