#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>            // for saving if you want
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/voxel_grid.h>


int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <path_to_kitti_bin>\n";
        return 1;
    }

    // Load KITTI .bin
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    std::ifstream input(argv[1], std::ios::binary);
    if (!input) {
        std::cerr << "Failed to open " << argv[1] << "\n";
        return 1;
    }
    struct Point { float x,y,z,intensity; };
    Point p;
    while (input.read(reinterpret_cast<char*>(&p), sizeof(Point))) {
        pcl::PointXYZI pt;
        pt.x = p.x; pt.y = p.y; pt.z = p.z; pt.intensity = p.intensity;
        cloud->push_back(pt);
    }
    cloud->width  = static_cast<uint32_t>(cloud->size());
    cloud->height = 1;
    cloud->is_dense = false;

    std::cout << "Loaded " << cloud->size() << " points.\n";

    // (Optional) Simple VoxelGrid downsampling on CPU
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZI>);
    {
      pcl::VoxelGrid<pcl::PointXYZI> vg;
      vg.setInputCloud(cloud);
      vg.setLeafSize(0.2f, 0.2f, 0.2f);
      vg.filter(*cloud_filtered);
      std::cout << "Downsampled to " << cloud_filtered->size() << " points.\n";
    }

    // Visualize
    pcl::visualization::PCLVisualizer viewer("KITTI Viewer");
    viewer.addPointCloud<pcl::PointXYZI>(cloud_filtered, "kitti_cloud");
    viewer.setBackgroundColor(0,0,0);
    viewer.initCameraParameters();

    while (!viewer.wasStopped()) {
      viewer.spinOnce(10);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}

