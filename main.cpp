#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>

#include <Eigen/Core>

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
#include <vtkMath.h>

#include <pcl/visualization/point_picking_event.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/extract_indices.h>

#include <vtkCellPicker.h>

// define a little struct to carry data into the callback
struct PickerData {
  pcl::visualization::PCLVisualizer* viewer;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree;
  double ground_z_plane;
  int box_count = 0;
};

// the callback itself
void pickPointEvent(const pcl::visualization::PointPickingEvent& event, void* vd)
{
  std::cout << "Pick point event triggered!" << std::endl;
  auto data = static_cast<PickerData*>(vd);

  if (!event.getPointIndex())  // no valid pick
    return;

  // 1) get the clicked point in world coords
  float x,y,z;
  event.getPoint(x,y,z);
  pcl::PointXYZI clicked; clicked.x=x; clicked.y=y; clicked.z=z;

  // 2) find the nearest actual cloud point to that click
  std::vector<int> nn_idx(1);
  std::vector<float> nn_dist(1);
  if (data->tree->nearestKSearch(clicked, 1, nn_idx, nn_dist) == 0)
    return;
  pcl::PointXYZI seed = data->cloud->points[nn_idx[0]];

  // 3) radius‐search around seed, excluding “ground” points
  const float cluster_radius = 2.5f;     // meters
  const float ground_thresh   = 0.5f;    // meters above ground
  std::vector<int> r_inds;
  std::vector<float> r_dist;
  data->tree->radiusSearch(seed, cluster_radius, r_inds, r_dist);

  pcl::PointIndices::Ptr cluster_inds(new pcl::PointIndices);
  for (int i : r_inds) {
    auto &pt = data->cloud->points[i];
    if ((pt.z - data->ground_z_plane) > ground_thresh)
      cluster_inds->indices.push_back(i);
  }
  if (cluster_inds->indices.empty()) return;

  // 4) abort if cluster’s bounding‐box too big for a car
  //    but first compute min/max using the Eigen overload:
  Eigen::Vector4f min_eig, max_eig;
  pcl::getMinMax3D(
      *data->cloud,
      *cluster_inds,
      min_eig,
      max_eig
  );
  
  // now copy into PointXYZI
  pcl::PointXYZI min_pt, max_pt;
  min_pt.x = min_eig[0];  min_pt.y = min_eig[1];  min_pt.z = min_eig[2];
  max_pt.x = max_eig[0];  max_pt.y = max_eig[1];  max_pt.z = max_eig[2];
  
  // compute extents
  double dx = max_pt.x - min_pt.x;
  double dy = max_pt.y - min_pt.y;
  double dz = max_pt.z - min_pt.z;
  if (dx > 5.0 || dy > 2.5 || dz > 2.5) {
    std::cout << "[Picker] cluster too large (" 
              << dx << "," << dy << "," << dz << ") – abort\n";
    return;
  } 

  // 5) draw an axis‐aligned box
  std::string id = "car_box_" + std::to_string(data->box_count++);
  data->viewer->addCube(
    min_pt.x, max_pt.x,
    min_pt.y, max_pt.y,
    min_pt.z, max_pt.z,
    1.0, 0.0, 0.0,
    id, 0
  );
  data->viewer->setShapeRenderingProperties(
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
    pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
    id
  );
}

class HorizonInteractorStyle : public vtkInteractorStyleTrackballCamera {
public:
  static HorizonInteractorStyle* New();
  vtkTypeMacro(HorizonInteractorStyle, vtkInteractorStyleTrackballCamera);

  HorizonInteractorStyle() {
    // default ground normal = world Z
    groundNormal[0]=0; groundNormal[1]=0; groundNormal[2]=1;
  }

  /// Call this once with your RANSAC plane normal (a_plane,b_plane,c_plane)
  void SetGroundNormal(double nx, double ny, double nz) {
    double len = std::sqrt(nx*nx + ny*ny + nz*nz);
    if (len > 1e-6) {
      groundNormal[0] = nx/len;
      groundNormal[1] = ny/len;
      groundNormal[2] = nz/len;
    }
  }

  
  // Treat right-click as middle-click for pan
  void OnRightButtonDown() override {
    // Forward to middle-button down (starts pan)
    vtkInteractorStyleTrackballCamera::OnMiddleButtonDown();
  }
  
  void OnRightButtonUp() override {
    // Forward to middle-button up (ends pan)
    vtkInteractorStyleTrackballCamera::OnMiddleButtonUp();
  }


  void OnLeftButtonDown() override {
    // record focal point to orbit around
    this->GetCurrentRenderer()->GetActiveCamera()->GetFocalPoint(focalPoint);
    vtkInteractorStyleTrackballCamera::OnLeftButtonDown();
  }

  void OnMouseMove() override {
    if (this->State == VTKIS_ROTATE) {
      // let the base style apply yaw/pitch/roll
      vtkInteractorStyleTrackballCamera::OnMouseMove();

      vtkCamera* cam = this->GetCurrentRenderer()->GetActiveCamera();
      double pos[3], fp[3], viewDir[3];
      cam->GetPosition(pos);
      cam->GetFocalPoint(fp);
      // compute view direction vector (from camera to focal)
      viewDir[0] = fp[0] - pos[0];
      viewDir[1] = fp[1] - pos[1];
      viewDir[2] = fp[2] - pos[2];
      vtkMath::Normalize(viewDir);

      // project groundNormal onto plane ⟂ viewDir: newUp = groundNormal − (groundNormal·viewDir)*viewDir
      double dot = vtkMath::Dot(groundNormal, viewDir);
      double newUp[3] = {
        groundNormal[0] - dot*viewDir[0],
        groundNormal[1] - dot*viewDir[1],
        groundNormal[2] - dot*viewDir[2]
      };
      vtkMath::Normalize(newUp);

      // enforce no‐roll: up = newUp, look at focalPoint
      cam->SetViewUp(newUp);
      cam->SetFocalPoint(focalPoint);
      // ensure orthonormal basis
      cam->OrthogonalizeViewUp();

      this->GetInteractor()->GetRenderWindow()->Render();
    }
    else {
      vtkInteractorStyleTrackballCamera::OnMouseMove();
    }
  }

private:
  double groundNormal[3];
  double focalPoint[3];
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
    // --- build an RGB cloud with blue/white coloring ---
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color(new pcl::PointCloud<pcl::PointXYZRGB>);
    cloud_color->reserve(cloud_filtered->size());
    
    const float thresh = 0.5f;  // 0.5m above ground
    for (auto &pt : cloud_filtered->points) {
        pcl::PointXYZRGB p;
        p.x = pt.x; 
        p.y = pt.y; 
        p.z = pt.z;
    
        if ( (pt.z - ground_z_plane) <= thresh ) {
          // “near ground” → blue
          p.r =   0; p.g =   0; p.b = 255;
        } else {
          // above threshold → white
          p.r = 255; p.g = 255; p.b = 255;
        }
        cloud_color->push_back(p);
    }
    cloud_color->width  = cloud_color->size();
    cloud_color->height = 1;
    cloud_color->is_dense = cloud_filtered->is_dense;
    
    // --- visualize the colored cloud instead ---
    viewer.addPointCloud<pcl::PointXYZRGB>(cloud_color, "colored_cloud");
    viewer.setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 5, "colored_cloud");

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

    // build a KdTree for fast picks & clustering:
    auto tree = pcl::make_shared<pcl::search::KdTree<pcl::PointXYZI>>();
    tree->setInputCloud(cloud_filtered);
    
    // set up the callback data
    PickerData pd;
    pd.viewer         = &viewer;
    pd.cloud          = cloud_filtered;
    pd.tree           = tree;
    pd.ground_z_plane = ground_z_plane;
   
    viewer.registerPointPickingCallback(pickPointEvent, (void*)&pd);

    while (!viewer.wasStopped()) {
      viewer.spinOnce(10);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}

