#include <opencv2/opencv.hpp>
#include <cmw_app/front_vision.h>
#include <memory>
#include <yaml-cpp/yaml.h>
#include<vector>
#include <opencv2/viz.hpp>
#include <common/impl/cxxopts.hpp>
#include <mutex>
#include <thread>
#include <chrono>
#include <ctime>
#include <iostream>   
#include <unistd.h>
#include <queue>

std::shared_ptr<adc::StereoSgmApp> stereo_app_ptr;

typedef std::vector<cv::Vec3b> PointColorType;
typedef std::vector<cv::Point3f> PointCloudType;
typedef  std::chrono::system_clock::time_point  TimeType;
typedef cv::Mat PointCloudMat;
typedef  std::shared_ptr<std::list<vector<Tn::Bbox>>> OutputDataType;
typedef  std::chrono::system_clock::time_point  TimeType;
typedef  cv::Mat ImageType;
typedef std::vector<std::string> ClassNameType;


cv::viz::Viz3d plot3d("3d Frame");
cv::Mat left_img, right_img;
std::mutex  mu_left, mu_right;
cv::VideoCapture  left_cap;
cv::VideoCapture  right_cap;
cv::Mat point_cloud;
int crop_top, crop_bottom;

const unsigned int kCountBuff = 10;


bool fetch_left = true;
bool fetch_right = true;

struct DepthTime
{
  PointCloudMat point_mat;
  TimeType time_stamp;
};

struct BboxTime
{
  OutputDataType bboxs;
  ClassNameType names;
  TimeType time_stamp;

};
std::queue<DepthTime> depth_time_cahce;
std::queue<BboxTime> bbox_time_cahce;



std::string gstreamer_pipeline( int capture_width,
                                int capture_height,
                                int display_width,
                                int display_height,
                                int frame_rate,
                                int filp_method,
                                int sensor_id = 0,
                                int sensor_mode = 3
                             )
{
        return "nvarguscamerasrc sensor-id=" + std::to_string(sensor_id) +
                " sensor-mode=" + std::to_string(sensor_mode) +
                " ! video/x-raw(memory:NVMM), " +
                "width=(int)" + std::to_string(capture_width) +
                ", height=(int)" + std::to_string(capture_height) +
                ", format=(string)NV12, " +
                "framerate=(fraction)" + std::to_string(frame_rate) +
                "/1 ! nvvidconv flip-method=" + std::to_string(filp_method) +
               " ! video/x-raw, width=(int)" + std::to_string(display_width) +  
               ", height=(int)" + std::to_string(display_height) +
               ", format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";    
} 
  
void ThreadFetchImageLeft()
{
    while(fetch_left)
{

     mu_left.lock();
     left_cap.read(left_img);
     mu_left.unlock();
     usleep(20000);
}
}

void ThreadFetchImageRight()
{
   int count = 0;
   std::string foder ="/media/nvidia/新加卷/DataSet/ADC/stereo/corp_dataset/";	
     while(fetch_right)
{
    if(count>143) count = 0;

     mu_right.lock();
    // right_img = cv::imread(foder + "right/" + std::to_string(count)+".png");
   //  left_img = cv::imread(foder + "left/" + std::to_string(count)+".png");
     //std::cout<<foder + "left/" + std::to_string(count)+".png"<<std::endl;
    // count++;
     right_cap.read(right_img);
     mu_right.unlock();
     usleep(20000);
}
}

void callback_deal_points_mat(PointCloudMat& point_cloud_float, TimeType& tm)
{
  DepthTime dt;
  dt.point_mat = point_cloud_float.clone();
  dt.time_stamp = tm;
  depth_time_cahce.push(dt);
  if(depth_time_cahce.size() > kCountBuff) depth_time_cahce.pop();
}

void  callback_deal_point(PointCloudType& pt_cloud, PointColorType& pt_color, TimeType &time)
{

    if(pt_cloud.empty()|| pt_color.empty()) return;
    static bool o = false;
    cv::viz::WCloud cloud_widget = cv::viz::WCloud(pt_cloud, pt_color);
    cloud_widget.setRenderingProperty(cv::viz::POINT_SIZE, 2);
    plot3d.showWidget("ref cloud", cloud_widget);
   std::cout<<"callback"<<std::endl;
   plot3d.wasStopped();
   while(!plot3d.wasStopped() && o)
  {
   plot3d.spinOnce(1, true);
  }
  o = false;
  plot3d.spinOnce(1, true);

  
}

void callback_bbox(OutputDataType& outputs, TimeType& time, ImageType& img, ClassNameType& class_name )
{
   auto output = *(outputs->begin());
   for(vector<Tn::Bbox>::iterator bbox = output.begin(); bbox != output.end(); ++bbox)
    {
      auto box = *bbox;

      cv::rectangle(img,cv::Point(box.left,box.top),cv::Point(box.right,box.bot),cv::Scalar(0,0,255),3,8,0);
      std::cout << "class=" << box.classId << " prob=" << box.score*100 << std::endl;
      cout << "left=" << box.left << " right=" << box.right << " top=" << box.top << " bot=" << box.bot << endl;
      string str1=class_name[box.classId];
      cv::putText(img,str1, cv::Point(box.left, box.top-15), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 255, 0));     
   
    }
  cv::imshow("detetion", img);
  cv::waitKey(10);
  BboxTime bt;
  bt.bboxs = outputs;
  bt.time_stamp = time;
  bt.names = class_name;
  bbox_time_cahce.push(bt);
  if(bbox_time_cahce.size() > kCountBuff) bbox_time_cahce.pop();
}

void MergeData(DepthTime& depth_data, BboxTime& bbox_data)
{

}

void ThreadSyncData()
{
  while (!bbox_time_cahce.empty() && !depth_time_cahce.empty())
  {
    BboxTime bt = bbox_time_cahce.front();
    DepthTime dt =  depth_time_cahce.front();
    TimeType bbox_time = bt.time_stamp;
    TimeType depth_time = dt.time_stamp;
    auto duration =  std::chrono::duration_cast<std::chrono::microseconds>(bbox_time - depth_time).count();
    if(duration/1.0 <= 1e-1 && duration/1.0 >= -1e-1)
    {
      MergeData(dt, bt);
      bbox_time_cahce.pop();
      depth_time_cahce.pop();
    }
    else if (duration/1.0 > 1e-1)
    {
      depth_time_cahce.pop();
    }
    else if (duration/1.0 < -1e-1)
    {
      bbox_time_cahce.pop();
    }
  }
}

void InitCamrea()
{
    //std::string pipline = gstreamer_pipeline(capture_width, capture_height, display_width,
						// display_height, frame_rate, flip_method,
						// camera_id, sensor_mode)
   std::string pipline_left = gstreamer_pipeline(1152, 768, 1152,768, 25, 0, 0);
   std::string pipline_right = gstreamer_pipeline(1152, 768, 1152,768, 25, 0, 1);

   left_cap.open(pipline_left, cv::CAP_GSTREAMER);
   right_cap.open(pipline_right, cv::CAP_GSTREAMER);
}

int main(int argc, char **argv)
{
  cxxopts::Options options(argv[0], "double_camera_node");
  options.add_options()
      ("h,help", "this app is used for double_camera_node")
      ("s,stereo_config_file", "configuration file path",
      cxxopts::value<std::string>()->default_value("./stereo.yaml"))
      ("d,detection_config_file", "configuration file path",
      cxxopts::value<std::string>()->default_value("./detection.yaml"))
      ("m,data_method", "fetch dada merthod is loop or thead, true:loop, false: thread",
      cxxopts::value<bool>()->default_value("true"))
      ("t,crop_top", "crop_top",
      cxxopts::value<int>()->default_value("200"))
      ("b,crop_bottom", "crop_bottom",
      cxxopts::value<int>()->default_value("570"))
      ;
  cxxopts::ParseResult opts =  options.parse(argc, argv);
  std::string stereo_config_file = opts["stereo_config_file"].as<std::string>();
  std::string detection_config_file = opts["detection_config_file"].as<std::string>();
  bool used_loop_fetch = opts["data_method"].as<bool>();
  crop_top = opts["crop_top"].as<int>();
  crop_bottom = opts["crop_bottom"].as<int>();
  // std::shared_ptr<adc::FrontVisionApp> front_vision_app_ptr = std::make_shared<adc::FrontVisionApp>(stereo_config_file, detection_config_file);
  adc::FrontVisionApp front_vision_app(stereo_config_file,detection_config_file);
  front_vision_app.SetStereoPointAndColorCallback(callback_deal_point);
  //txs
  front_vision_app.SetDetectionObjectCallback(callback_bbox);
  //txs
  front_vision_app.Ready();
  InitCamrea();
  plot3d.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem());
  std::thread thread_sync_res(ThreadSyncData);
  thread_sync_res.detach();
  if(!used_loop_fetch)
  {
    std::thread thread_left(ThreadFetchImageLeft);
    std::thread thread_right(ThreadFetchImageRight);
    thread_left.detach(); 
    thread_right.detach();
    if(!left_cap.isOpened())
    {
      std::cout<<"Can not open left camera"<<std::endl;
      return 0;
    }
    if(!right_cap.isOpened())
    {
      std::cout<<"Can not open right camera"<<std::endl;
      return 0;
    }
  }

  while(true)
  {

    if(used_loop_fetch)
    {
      left_cap.read(left_img);
      usleep(10000);
      right_cap.read(right_img);
      usleep(10000);
      if(!left_img.empty() && !right_img.empty() )
      {
        left_img = left_img(cv::Range(crop_top, crop_bottom),cv::Range(0, left_img.cols));
        right_img = right_img(cv::Range(crop_top, crop_bottom),cv::Range(0, right_img.cols));
       // cv::imshow("left",left_img);
        //cv::imshow("right",right_img);
        //cv::waitKey(10);
        std::chrono::system_clock::time_point tm = std::chrono::system_clock::now();
        front_vision_app.SetImages(left_img,  right_img, tm);
        left_img = cv::Mat();
        right_img = cv::Mat();
      }
    }
    else
    {
      mu_left.lock();
      mu_right.lock();
      if(!left_img.empty() && !right_img.empty() )
      {
        left_img = left_img(cv::Range(crop_top, crop_bottom),cv::Range(0, left_img.cols));
        right_img = right_img(cv::Range(crop_top, crop_bottom),cv::Range(0, right_img.cols));
        //cv::imshow("left",left_img);
       // cv::imshow("right",right_img);
        //cv::waitKey(10);
        std::chrono::system_clock::time_point tm = std::chrono::system_clock::now();
        front_vision_app.SetImages(left_img,  right_img, tm);
        left_img = cv::Mat();
        right_img = cv::Mat();
      }
      mu_left.unlock();
      mu_right.unlock();
    }
  }
  fetch_left = false;
  fetch_right = false;
  thread_sync_res.join();
  return 0;
}
