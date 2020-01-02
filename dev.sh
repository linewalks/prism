#! /bin/bash

function abs_path {
  (cd $(dirname $1) &>/dev/null && printf "%s/%s" "$PWD" "$(basename $1)")
}

dir=$(dirname "$0")
data_dir=$(abs_path $dir/data)
src_dir=$(abs_path $dir/docker/src)

docker run -e ID=local_test \
  --name prism-dev \
  -v $data_dir:/data \
  -v $src_dir:/src_dev \
  -it --rm \
  prism-tensorflow-1.14.0-gpu-py3 \
  sh -c './train_dev.sh; ./inference_dev.sh'
