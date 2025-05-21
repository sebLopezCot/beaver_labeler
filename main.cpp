#include <chrono>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <string>
#include <algorithm>

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

#include <vtkCommand.h>
#include <vtkRenderWindowInteractor.h>

class GroundBoxPicker : public vtkCommand {
public:
  static GroundBoxPicker* New() { return new GroundBoxPicker; }

  vtkTypeMacro(GroundBoxPicker, vtkCommand);

  GroundBoxPicker()
    : state(0), box_height(2.0), viewer(nullptr)
  {}

  /// Set the PCLVisualizer so we can call addCube()
  void setViewer(pcl::visualization::PCLVisualizer* v) {
    viewer = v;
  }

  /// Provide your fitted plane normal (nx,ny,nz) and d so you solve n·x + d = 0
  void setPlane(double nx, double ny, double nz, double d_) {
    // normalize
    double len = std::sqrt(nx*nx + ny*ny + nz*nz);
    ground_n[0] = nx/len; ground_n[1] = ny/len; ground_n[2] = nz/len;
    d = d_;
  }

  /// Fixed height above ground
  void setBoxHeight(double h) { box_height = h; }

  void Execute(vtkObject* caller, unsigned long eventId, void* /*callData*/) override {
    if (eventId != vtkCommand::LeftButtonPressEvent || !viewer) return;

    auto iren = static_cast<vtkRenderWindowInteractor*>(caller);
    int xy[2];
    iren->GetEventPosition(xy);

    // 1) Unproject click into a world‐space ray (near & far points)
    auto renderer = viewer->getRenderWindow()
                         ->GetRenderers()
                         ->GetFirstRenderer();
    // get near point
    renderer->SetDisplayPoint(xy[0], xy[1], 0.0);
    renderer->DisplayToWorld();
    double p0[4]; renderer->GetWorldPoint(p0);
    for(int i=0;i<3;++i) p0[i] /= p0[3];
    // get far point
    renderer->SetDisplayPoint(xy[0], xy[1], 1.0);
    renderer->DisplayToWorld();
    double p1[4]; renderer->GetWorldPoint(p1);
    for(int i=0;i<3;++i) p1[i] /= p1[3];

    // 2) intersect ray with plane:  n·(P0 + t*(P1−P0)) + d = 0
    double dir[3] = { p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2] };
    double denom = vtkMath::Dot(ground_n, dir);
    if (std::abs(denom) < 1e-6) return;  // ray parallel
    double numer = -(vtkMath::Dot(ground_n, p0) + d);
    double t = numer/denom;
    double xi = p0[0] + t*dir[0];
    double yi = p0[1] + t*dir[1];
    double zi = p0[2] + t*dir[2];

    // 3) record the two corners
    if (state == 0) {
      c1[0]=xi; c1[1]=yi; c1[2]=zi;
      state = 1;
      std::cout<<"Corner 1: ("<<xi<<","<<yi<<")\n";
    }
    else {
      double c2[3] = {xi, yi, zi};
      // compute axis‐aligned box from c1 & c2
      double xmin = std::min(c1[0], c2[0]);
      double xmax = std::max(c1[0], c2[0]);
      double ymin = std::min(c1[1], c2[1]);
      double ymax = std::max(c1[1], c2[1]);
      double zmin = zi;               // ground plane height
      double zmax = zmin + box_height;

      // 4) draw the cube
      auto id = std::string("ground_box_") + std::to_string(box_id++);
      viewer->addCube(xmin, xmax, ymin, ymax, zmin, zmax, 
                      0.0, 1.0, 0.0,  // green wireframe
                      id, 0);
      viewer->setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
        pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
        id);

      state = 0;
    }
  }

private:
  int    state;            // 0 = waiting for first click, 1 = waiting for second
  double c1[3];            // first corner
  double ground_n[3], d;   // plane
  double box_height;       
  int    box_id = 0;       
  pcl::visualization::PCLVisualizer* viewer;
};


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

    // 1) Create and configure the picker
    auto picker = vtkSmartPointer<GroundBoxPicker>::New();
    picker->setViewer(&viewer);
    picker->setPlane(
      coef_plane->values[0],
      coef_plane->values[1],
      coef_plane->values[2],
      coef_plane->values[3]
    );
    picker->setBoxHeight(1.8);  // e.g. standard car height
    
    // 2) Install on the interactor
    auto iren2 = viewer.getRenderWindow()->GetInteractor();
    iren2->AddObserver(vtkCommand::LeftButtonPressEvent, picker);
    
    while (!viewer.wasStopped()) {
      viewer.spinOnce(10);
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return 0;
}

