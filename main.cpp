#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>            // for saving if you want
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common.h>   // for getMinMax3D
#include <pcl/filters/voxel_grid.h>

#include <vtkSmartPointer.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRendererCollection.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>

class OrbitInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
  static OrbitInteractorStyle* New();
  vtkTypeMacro(OrbitInteractorStyle, vtkInteractorStyleTrackballCamera);

  // Override but forward everything to the base class:
  void OnLeftButtonDown() override {
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
  }
  void OnLeftButtonUp() override {
    vtkInteractorStyleTrackballCamera::OnLeftButtonUp();
  }
  void OnMiddleButtonDown() override {
    vtkInteractorStyleTrackballCamera::OnMiddleButtonDown();
  }
  void OnMiddleButtonUp() override {
    vtkInteractorStyleTrackballCamera::OnMiddleButtonUp();
  }
  void OnRightButtonDown() override {
    vtkInteractorStyleTrackballCamera::OnRightButtonDown();
  }
  void OnRightButtonUp() override {
    vtkInteractorStyleTrackballCamera::OnRightButtonUp();
  }
  void OnMouseMove() override {
    vtkInteractorStyleTrackballCamera::OnMouseMove();
  }
  void OnMouseWheelForward() override {
    vtkInteractorStyleTrackballCamera::OnMouseWheelForward();
  }
  void OnMouseWheelBackward() override {
    vtkInteractorStyleTrackballCamera::OnMouseWheelBackward();
  }
};
vtkStandardNewMacro(OrbitInteractorStyle);


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
    pcl::visualization::PCLVisualizer viewer("BeaverLabeler");
    viewer.addPointCloud<pcl::PointXYZI>(cloud_filtered, "kitti_cloud");
    viewer.setBackgroundColor(0,0,0);
    viewer.initCameraParameters();

    // Grab the underlying VTK interactor and replace its style:
    auto iren = viewer.getRenderWindow()->GetInteractor();
    vtkSmartPointer<OrbitInteractorStyle> style = vtkSmartPointer<OrbitInteractorStyle>::New();
    style->SetCurrentRenderer(viewer.getRendererCollection()->GetFirstRenderer());
    iren->SetInteractorStyle(style);


    // ---- bird's-eye initial view ----
    // compute axis‐aligned bounding box to find center
    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D(*cloud_filtered, min_pt, max_pt);
    float cx = (min_pt.x + max_pt.x) * 0.5f;
    float cy = (min_pt.y + max_pt.y) * 0.5f;
    float cz = (min_pt.z + max_pt.z) * 0.5f;
    // compute XY span and back off far enough 
    float dx = max_pt.x - min_pt.x;
    float dy = max_pt.y - min_pt.y;
    float span = std::max(dx, dy);
    float height = span * 1.25f;
    // camera_x, camera_y, camera_z, view_x, view_y, view_z, up_x, up_y, up_z
    viewer.setCameraPosition(
      cx, cy, cz + height,
      cx, cy, cz,
      0.0, 1.0, 0.0   // Y‐axis as the “up” direction on screen
    );

    // make sure near/far clipping planes encompass the whole cloud
    {
      auto rc = viewer.getRendererCollection();
      if (rc->GetNumberOfItems() > 0) {
        vtkRenderer* ren = rc->GetFirstRenderer();
        ren->ResetCameraClippingRange();
      }
    }

    while (!viewer.wasStopped()) {
      viewer.spinOnce(10);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}

