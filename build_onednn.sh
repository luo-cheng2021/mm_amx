#!/usr/bin/bash

mkdir -p thirdparty/oneDNN/build
cd thirdparty/oneDNN/build
cmake -DDNNL_BUILD_TESTS=ON -DCMAKE_INSTALL_PREFIX=./install ..
cmake --build . --target install -- -j 24
