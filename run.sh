#!/bin/bash
cd build
cmake .. && make -j$(nproc)
cd ..
./build/BeaverLabeler $@
