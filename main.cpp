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
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>

#include <vtkSmartPointer.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRendererCollection.h>
#include <vtkRenderer.h>
#include <vtkCamera.h>
#include <cmath>

class HorizonInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
  static HorizonInteractorStyle* New();
  vtkTypeMacro(HorizonInteractorStyle, vtkInteractorStyleTrackballCamera);

  /// Call this once with your RANSAC plane normal (a_plane,b_plane,c_plane)
  void SetGroundNormal(double nx, double ny, double nz) {
    double len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 0.0) {
      groundNormal[0] = nx/len;
      groundNormal[1] = ny/len;
      groundNormal[2] = nz/len;
    }
  }

  void OnLeftButtonDown() override {
    // remember focal point so we always orbit around it
    this->GetCurrentRenderer()
        ->GetActiveCamera()
        ->GetFocalPoint(focalPoint);
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
  }

  void OnMouseMove() override {
    if (this->State == VTKIS_ROTATE) {
      // perform the default yaw/pitch/roll
      vtkInteractorStyleTrackballCamera::OnMouseMove();

      // then immediately enforce our “no roll” horizon
      vtkCamera* cam = this->GetCurrentRenderer()->GetActiveCamera();
      cam->SetViewUp(groundNormal);          // horizon = ground plane
      cam->OrthogonalizeViewUp();            // clean up any drift
      cam->SetFocalPoint(focalPoint);        // keep look‐at fixed
      this->GetInteractor()->GetRenderWindow()->Render();
    }
    else {
      vtkInteractorStyleTrackballCamera::OnMouseMove();
    }
  }

private:
  double groundNormal[3] = {0,0,1};         // default to world Z
  double focalPoint[3]    = {0,0,0};
};

vtkStandardNewMacro(HorizonInteractorStyle);


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
    pcl::PointXYZI min_pt, max_pt;
    pcl::getMinMax3D(*cloud_filtered, min_pt, max_pt);
    
    // 1) RANSAC‐based ground plane estimation
    pcl::SACSegmentation<pcl::PointXYZI> seg_plane;
    seg_plane.setOptimizeCoefficients(true);
    seg_plane.setModelType(pcl::SACMODEL_PLANE);
    seg_plane.setMethodType(pcl::SAC_RANSAC);
    seg_plane.setDistanceThreshold(0.2);
    seg_plane.setMaxIterations(1000);
    seg_plane.setInputCloud(cloud_filtered);
    
    pcl::ModelCoefficients::Ptr coef_plane(new pcl::ModelCoefficients());
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices());
    seg_plane.segment(*inliers_plane, *coef_plane);
    
    // fallback to simple min‐z
    float ground_z_plane = min_pt.z;
    if (!inliers_plane->indices.empty()) {
      double a_plane = coef_plane->values[0],
             b_plane = coef_plane->values[1],
             c_plane = coef_plane->values[2],
             d_plane = coef_plane->values[3];
      if (c_plane < 0) { a_plane = -a_plane; b_plane = -b_plane; c_plane = -c_plane; d_plane = -d_plane; }
      double tilt_plane = std::abs(c_plane / std::sqrt(a_plane*a_plane + b_plane*b_plane + c_plane*c_plane));
      if (tilt_plane > 0.9) {
        double cx_plane = (min_pt.x + max_pt.x) * 0.5;
        double cy_plane = (min_pt.y + max_pt.y) * 0.5;
        ground_z_plane = -(a_plane*cx_plane + b_plane*cy_plane + d_plane) / c_plane;
      }
    }
    
    // 2) Compute XY bounds + margin
    float dx_plane     = max_pt.x - min_pt.x;
    float dy_plane     = max_pt.y - min_pt.y;
    float span_plane   = std::max(dx_plane, dy_plane);
    float margin_plane = span_plane * 0.5f;
    
    double xmin_plane = min_pt.x - margin_plane;
    double xmax_plane = max_pt.x + margin_plane;
    double ymin_plane = min_pt.y - margin_plane;
    double ymax_plane = max_pt.y + margin_plane;
    
    // 3) Build a large rectangle at z = ground_z_plane
    pcl::PointCloud<pcl::PointXYZ>::Ptr ground_rect_plane(new pcl::PointCloud<pcl::PointXYZ>);
    ground_rect_plane->push_back({ 
        static_cast<float>(xmin_plane), 
        static_cast<float>(ymin_plane), 
        static_cast<float>(ground_z_plane) 
    });
    ground_rect_plane->push_back({ 
        static_cast<float>(xmax_plane), 
        static_cast<float>(ymin_plane), 
        static_cast<float>(ground_z_plane) 
    });
    ground_rect_plane->push_back({ 
        static_cast<float>(xmax_plane), 
        static_cast<float>(ymax_plane), 
        static_cast<float>(ground_z_plane) 
    });
    ground_rect_plane->push_back({ 
        static_cast<float>(xmin_plane), 
        static_cast<float>(ymax_plane), 
        static_cast<float>(ground_z_plane) 
    });
    
    // 4) Add and style the polygon
    viewer.addPolygon<pcl::PointXYZ>(ground_rect_plane, "ground_rect_plane", 0);
    viewer.setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_SURFACE,
        "ground_rect_plane");
    viewer.setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_COLOR,
        0.7, 0.7, 0.7,
        "ground_rect_plane");
    viewer.setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY,
        0.4,
        "ground_rect_plane");

    // Add point cloud
    viewer.addPointCloud<pcl::PointXYZI>(cloud_filtered, "kitti_cloud");
    viewer.setBackgroundColor(0,0,0);
    viewer.initCameraParameters();

    // Grab the underlying VTK interactor and replace its style:
    // assume coef_plane holds your RANSAC plane coefficients [a_plane,b_plane,c_plane,d_plane]
    double nx_plane = coef_plane->values[0];
    double ny_plane = coef_plane->values[1];
    double nz_plane = coef_plane->values[2];
    
    // set up the style on your PCLVisualizer’s interactor
    auto iren = viewer.getRenderWindow()->GetInteractor();
    vtkSmartPointer<HorizonInteractorStyle> horizonStyle =
        vtkSmartPointer<HorizonInteractorStyle>::New();
    horizonStyle->SetCurrentRenderer(
        viewer.getRendererCollection()->GetFirstRenderer());
    horizonStyle->SetGroundNormal(nx_plane, ny_plane, nz_plane);
    iren->SetInteractorStyle(horizonStyle);


    // ---- bird's-eye initial view ----
    // compute axis‐aligned bounding box to find center
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

