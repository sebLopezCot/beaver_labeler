BeaverLabeler
##############

Build:
```
sudo apt update
sudo apt install build-essential cmake \
     libpcl-dev libvtk9-dev libvtk9-qt-dev qt6-base-dev qt6-base-dev-tools

mkdir build && cd build
cmake ..
make -j$(nproc)
./BeaverLabeler
```
