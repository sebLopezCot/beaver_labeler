# BeaverLabeler

### Data
```
mkdir data && cd data
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip
unzip 2011_09_26_drive_0020_sync.zip
```

### Build
```
sudo apt update
sudo apt install build-essential cmake \
     libpcl-dev libvtk9-dev libvtk9-qt-dev qt6-base-dev qt6-base-dev-tools

mkdir build && cd build
cmake ..
make -j$(nproc)

```

### Run
```
./run.sh data/2011_09_26/2011_09_26_drive_0020_sync/velodyne_points/data/0000000008.bin 
```

### Controls
 - Left click + drag: rotate
 - Right click + drag: pan
 - Scroll: zoom
 - Hold V: Label mode
 - Click (in Label Mode): put a corner down
 - Backspace + rotate: remove last label


When labeling, just pick 2 opposite corners somewhere on the ground plane and a label will be populated with a fixed height.
