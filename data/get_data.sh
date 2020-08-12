#!/bin/bash

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    get_cmd=wget
elif [[ "$OSTYPE" == "darwin"* ]]; then
    get_cmd="curl -O"
fi

## check if file exists
FILE=w8a
if [ ! -f "$FILE" ]; then
	$get_cmd https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a
else
	echo "$FILE exist, no need to re-download"
fi

## check if file exists
FILE=w8a.t
if [ ! -f "$FILE" ]; then
    $get_cmd https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t
else
	echo "$FILE exist, no need to re-download"
fi

## check if file exists
FILE=covtype
if [ ! -f "$FILE" ]; then
    $get_cmd https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.bz2
	bzip2 -d covtype.libsvm.binary.bz2
	mv covtype.libsvm.binary covtype
else
	echo "$FILE exist, no need to re-download"
fi
