#!/bin/bash

run_type=$1

if [ "$run_type" == "train" ] ; then
  args=`cat /opt/ml/input/config/hyperparameters.json \
    | jq -r 'keys[] as $k | "--\($k) \(.[$k])"' | tr '\n' ' '`
  python entrypoint.py $args
else
  echo "The argument '$run_type' is not supported"
fi
