#! /bin/bash
FASTER=${1:-0}
CODE="$(tr -dc '[:alnum:]' </dev/urandom | head -c 6)"
DOCKERZIP="prism_${CODE}.tar.gz"
echo $DOCKERZIP
if [ $FASTER = "-f" ] 
then
   echo "faster compression using multi-core"
   docker save prism-tensorflow-1.14.0-gpu-py3 | pigz -c -v > $DOCKERZIP
else
   echo "normal compression"
   docker save prism-tensorflow-1.14.0-gpu-py3 | gzip -c -v > $DOCKERZIP
fi
