#!/bin/bash
CUR_DIR=$(dirname $0)
cd ${CUR_DIR}

BUILD_DIR=build

if [ ! -d ${BUILD_DIR} ]; then
    mkdir ${BUILD_DIR}
fi

cd ${BUILD_DIR}
cmake ../

make -j3
if [ ! $? ]; then
    echo "Build TVM Failed!"
    exit 0
fi
