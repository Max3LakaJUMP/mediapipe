#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/modules/face_geometry/protos/face_geometry.pb.h"

#include "mediapipe/modules/face_geometry/geometry_pipeline_calculator.pb.h"

#include "mediapipe/modules/face_geometry/libs/validation_utils.h"
#include "mediapipe/modules/face_geometry/protos/environment.pb.h"
#include "mediapipe/modules/face_geometry/protos/geometry_pipeline_metadata.pb.h"
#include "mediapipe/util/resource_util.h"
#include "meta.h"
#include "facegenlib.h"

using namespace ::mediapipe;
using namespace ::face_geometry;

#define START if(!start()) {printf("Graph is not started"); return false;}; 
#define STARTED if(!_is_started) return false; 
#define ERR(check) if(check){return false;}
#ifdef _DEBUG_MSG_
#define ERR_MSG(check, msg) if(check){printf(msg); return false;}
#else 
#define ERR_MSG(check, msg) if(check){return false;}
#endif

constexpr char kInputStream[] = "input_video";
constexpr char kEyeOffsetsL[] = "right_eye_offsets";
constexpr char kEyeOffsetsR[] = "left_eye_offsets";
constexpr char kFaceOffsets[] = "face_offsets";

constexpr char kOutputStream[] = "output_video";
constexpr char kOutputStreamGeometry[] = "multi_face_geometry";
constexpr char kOutputStreamLandmarks[] = "updated_face_landmarks";
constexpr char kOutputStreamIrisLandmarks[] = "iris_landmarks";
constexpr char kWindowName[] = "Face Gen";

DEFINE_string(
    calculator_graph_config_file, "",
    "Name of file containing text format CalculatorGraphConfig proto.");
DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");


Status init_camera(cv::VideoCapture &capture, std::string window_name = "Face gen", int width=1280, int height=720) {
  LOG(INFO) << "Initialize the camera.";
  capture.open(0);
  RET_CHECK(capture.isOpened());
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
  capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
  capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
  capture.set(cv::CAP_PROP_FPS, 33);
#endif
  return mediapipe::OkStatus();
}

Status graph_init(CalculatorGraph **graph, std::string graph_path) {
  int argc = 2;
  char temp_str[80];
  strcpy(temp_str, (std::string("--calculator_graph_config_file=") + graph_path ).c_str());
  char* argv_array[] = { "godot.windows.tools.64.exe", temp_str};
  char** argv = argv_array;

  // google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (*graph != nullptr)
    delete *graph;
  *graph = new CalculatorGraph;
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(file::GetContents(FLAGS_calculator_graph_config_file, &calculator_graph_config_contents));
  // LOG(INFO) << "Get calculator graph config contents: " << calculator_graph_config_contents;
  CalculatorGraphConfig config = ParseTextProtoOrDie<CalculatorGraphConfig>(calculator_graph_config_contents);
  // LOG(INFO) << "Initialize the calculator graph.";
  MP_RETURN_IF_ERROR((*graph)->Initialize(config));
  return OkStatus();
}

Status poller_init(CalculatorGraph *graph, OutputStreamPoller **poller, const std::string &stream_name){
  StatusOrPoller status_or_poller = graph->AddOutputStreamPoller(stream_name);
  if (status_or_poller.status().ok())
    printf((std::string("Initialized ") + stream_name + std::string(" poller\n")).c_str());
  else{
    printf((std::string("Can't initialize ") + stream_name + std::string(" poller\n")).c_str());
  }
  MP_RETURN_IF_ERROR(status_or_poller.status());
  if (*poller != nullptr)
    delete *poller;
  *poller = new OutputStreamPoller(std::move(status_or_poller.ValueOrDie()));
  return OkStatus();
}

::mediapipe::Status save_video(const mediapipe::ImageFrame &output_frame, cv::VideoWriter &writer, cv::VideoCapture &capture) {
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
  if (!writer.isOpened()) {
    // LOG(INFO) << "Prepare video writer.";
    writer.open(FLAGS_output_video_path,
                mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
    RET_CHECK(writer.isOpened());
  }
  writer.write(output_frame_mat);
  return mediapipe::OkStatus();
}

void display_frame(const mediapipe::ImageFrame &output_frame, std::string window_name = "Face gen") {
  cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
  cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
  cv::imshow(window_name, output_frame_mat);
}

void GraphLib::set_graph_path(const char *p_graph_path) {
  graph_path = std::string(p_graph_path);
}

const char *GraphLib::get_graph_path() {
  return graph_path.c_str();
}

void GraphLib::set_window_name(const char *p_window_name) {
  window_name = std::string(p_window_name);
  _is_window_name_changed = true;
}

const char *GraphLib::get_window_name() {
  return window_name.c_str();
}


bool GraphLib::start() {
  if (!_is_started && graph_init(&graph, graph_path).ok()){
    if (poller_init(graph, &video_poller, kOutputStream).ok()){ 
      poller_init(graph, &landmarks_poller, kOutputStreamLandmarks);
      poller_init(graph, &iris_landmarks_poller, kOutputStreamIrisLandmarks);
      //poller_init(graph, &geometry_poller, kOutputStreamGeometry);
      if (graph->StartRun({}).ok()){
        _is_started = true;
        _finished = false;
      }
    }
  }
  return _is_started;
}

bool GraphLib::start_capture() {
  if (!_is_capture_started){
    _is_capture_started = init_camera(capture, window_name, camera_width, camera_height).ok();
  }
  return _is_capture_started;
}

bool GraphLib::finish(){
  if (_is_started && !_finished){
    // LOG(INFO) << "Shutting down.";
    if (graph->CloseInputStream(kInputStream).ok()){// && graph->CloseInputStream(kEyeOffsetsL).ok() && graph->CloseInputStream(kEyeOffsetsR).ok() && graph->CloseInputStream(kFaceOffsets).ok()){
      _finished = true;
      return graph->WaitUntilDone().ok();
    }
    return false;
  }
  return false;
}

void GraphLib::free(){
  delete this;
}


bool GraphLib::in_camera() {
  START
  cv::Mat camera_frame_raw;
  start_capture();
  capture >> camera_frame_raw;
  if (camera_frame_raw.empty()) return false;  // End of video.
  cv::Mat camera_frame;
  cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
  cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
  
  input_frame.reset();
  input_frame = absl::make_unique<ImageFrame>(
      mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
      mediapipe::ImageFrame::kDefaultAlignmentBoundary);
  cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
  camera_frame.copyTo(input_frame_mat);
  if (_in_exec()){
    input_height = camera_height;
    input_width = camera_width;
    return true;
  }else{
    return false;
  }
}

bool GraphLib::in_texture(int width, int height, uint8_t* pixel_data) {
  START
  input_frame.reset();
  input_frame = absl::make_unique<mediapipe::ImageFrame>(
      mediapipe::ImageFormat::SRGBA, width, height, width * sizeof(uint8) * 4, pixel_data);
  input_width = width;
  input_height = height;
  if (_in_exec()){
    input_width = width;
    input_height = height;
    // Wait for result
    while (landmarks_poller->QueueSize() == 0 && iris_landmarks_poller->QueueSize() == 0) {}
    return true;
  }else{
    return false;
  }
}

#define POOLLER_CHECK(poller, packet, name) \
ERR_MSG(poller == nullptr, (std::string("No ") + std::string(name) + std::string(" poller\n")).c_str()); \
int size = poller->QueueSize(); \
ERR_MSG(size == 0, (std::string("No queue in ") + std::string(name) + std::string(" poller\n")).c_str()); \
for (int i = 0; i < size; i++) { ERR_MSG(!poller->Next(&packet), (std::string("No packets in ") + std::string(name) + std::string(" poller\n")).c_str());} \
ERR_MSG(packet.IsEmpty(), (std::string("Pocket in ") + std::string(name) + std::string(" poller is empty\n")).c_str());

bool get_landmarks(OutputStreamPoller *poller, NormalizedLandmarkList &face){

  
  Packet packet;
  POOLLER_CHECK(poller, packet, kOutputStreamLandmarks)
  face = packet.Get<NormalizedLandmarkList>();

  //auto& output_frame = packet.Get<NormalizedLandmarkList>();
  //if(output_frame.size() == 0)
  //  return false;
  //face = output_frame[0];
  return true;
}

bool GraphLib::out_display() {
  START
  Packet packet;
  POOLLER_CHECK(video_poller, packet, kOutputStream)
  auto& output_frame = packet.Get<ImageFrame>();
  if (_is_window_name_changed){
    cv::namedWindow(window_name, /*flags=WINDOW_AUTOSIZE*/ 1);
    _is_window_name_changed = false;
  }

  display_frame(output_frame, window_name);
  return true;
}

bool GraphLib::out_landmarks(float **landmarks, int *landmarks_size){
  START
  Packet packet;
  POOLLER_CHECK(landmarks_poller, packet, "landmarks")
  NormalizedLandmarkList face = packet.Get<NormalizedLandmarkList>();
  *landmarks_size = face.landmark_size();
  float *l = new float[*landmarks_size * 4];
  int j = 0;
  for (int i = 0; i < *landmarks_size; i++)
  {
    l[j] = face.landmark(i).x();
    l[j+1] = face.landmark(i).y();
    l[j+2] = face.landmark(i).z();
    l[j+3] = face.landmark(i).presence();
    j += 4;
  }
  *landmarks = l;
  if (faces.size() > 0)
    faces.pop_back();
  faces.push_back(face);
  return true;
}  

bool GraphLib::out_iris_landmarks(float **landmarks, int *landmarks_size){
  START
  Packet packet;
  POOLLER_CHECK(iris_landmarks_poller, packet, "iris_landmarks")
  NormalizedLandmarkList face = packet.Get<NormalizedLandmarkList>();
  *landmarks_size = face.landmark_size();
  float *l = new float[*landmarks_size * 4];
  int j = 0;
  for (int i = 0; i < *landmarks_size; i++)
  {
    l[j] = face.landmark(i).x();
    l[j+1] = face.landmark(i).y();
    l[j+2] = face.landmark(i).z();
    l[j+3] = face.landmark(i).presence();
    j += 4;
  }
  *landmarks = l;
  return true;
}  

bool GraphLib::calc_metadata(){
  delete metadata;
  metadata = new GeometryPipelineMetadata;
  for (int i = 0; i < landmark_id_size; i++)
  {
    WeightedLandmarkRef *procrustes_landmark_basis = metadata->add_procrustes_landmark_basis();
    procrustes_landmark_basis->set_landmark_id(landmark_ids[i]);
    procrustes_landmark_basis->set_weight(weights[i]);
  }
  Mesh3d *canonical_mesh = new Mesh3d;
  canonical_mesh->set_primitive_type(Mesh3d::TRIANGLE);
  canonical_mesh->set_vertex_type(Mesh3d::VERTEX_PT);
  for (int i = 0; i < vertex_buffer_size; i++)
  {
    canonical_mesh->add_vertex_buffer(vertex_buffer[i]);
  }
  for (int i = 0; i < index_buffer_size; i++)
  {
    canonical_mesh->add_index_buffer(index_buffer[i]);
  }
  metadata->set_allocated_canonical_mesh(canonical_mesh);
  
  return true;
}


bool GraphLib::calc_environment(){
  delete environment;
  environment = new Environment;
  environment->set_origin_point_location(face_geometry::OriginPointLocation::TOP_LEFT_CORNER);
  PerspectiveCamera *perspective_camera = new PerspectiveCamera;
  perspective_camera->set_vertical_fov_degrees(vertical_fov_degrees);
  perspective_camera->set_near(1.0);
  perspective_camera->set_far(10000.0);
  environment->set_allocated_perspective_camera(perspective_camera); //error
  environment->set_origin_point_location(OriginPointLocation::TOP_LEFT_CORNER);
  return true;
}

bool GraphLib::calc_geometry_pipeline(){
  geometry_pipeline.reset();
  StatusOr<std::unique_ptr<GeometryPipeline>> status_or_geometry_pipeline = CreateGeometryPipeline(*environment, *metadata);
  ERR_MSG(!status_or_geometry_pipeline.status().ok(), "CreateGeometryPipeline error");
  geometry_pipeline = std::move(status_or_geometry_pipeline.ValueOrDie());
  return true;
}


bool GraphLib::out_polygon(float **polygon, int **polygons, int *polygon_size, int *polygons_size, float **transform_matrix){
  START
  // Packet packet;
  // POOLLER_CHECK(geometry_poller, packet, kOutputStreamGeometry)
  // std::vector<FaceGeometry> face_geometry_list = packet.Get<std::vector<FaceGeometry>>();
  ERR_MSG(faces.size() == 0, "No landmark data");
  ERR_MSG(faces[0].landmark_size() == 0, "No landmark data");
  if (environment == nullptr){
    ERR_MSG(!calc_environment(), "Metadata error");
  }
  if (metadata == nullptr){

    ERR_MSG(!calc_metadata(), "Metadata error");
  }
  if (geometry_pipeline == nullptr){
    ERR_MSG(!calc_geometry_pipeline(), "GeometryPipeline error");
  }
  StatusOr<std::vector<FaceGeometry>> status_or_geometry = geometry_pipeline->EstimateFaceGeometry(faces, input_width, input_height);
  ERR_MSG(!status_or_geometry.status().ok(), "EstimateFaceGeometry error");
  std::vector<FaceGeometry> face_geometry_list = std::move(status_or_geometry.ValueOrDie());
  ERR_MSG(face_geometry_list.size() == 0, "No faces generated");

  FaceGeometry face_geometry = face_geometry_list[0];
  auto& output_mesh = face_geometry.mesh();


  *polygon_size = output_mesh.vertex_buffer_size();
  *polygons_size = output_mesh.index_buffer_size();
  float *p = new float[*polygon_size];
  for (int i = 0; i < *polygon_size; i++) {
    p[i] = output_mesh.vertex_buffer(i);
  }
  *polygon = p;

  int *ps = new int[*polygons_size];
  for (int i = 0; i < *polygons_size; i++) {
    ps[i] = output_mesh.index_buffer(i);
  }
  *polygons = ps;
  
  MatrixData matrix = face_geometry.pose_transform_matrix();
  float *md = new float[16];
  for (int i = 0; i < 16; i++) {
    md[i] = matrix.packed_data(i);
  }
  *transform_matrix = md;
  return true;
}

bool GraphLib::is_started() {
  return _is_started;
}

bool GraphLib::is_key_pressed(){
  const int pressed_key = cv::waitKey(5);
  return (pressed_key >= 0 && pressed_key != 255);
}

bool vector_init(std::unique_ptr<NormalizedLandmarkList> *vector) {
  if(*vector == nullptr){
    *vector = std::make_unique<NormalizedLandmarkList>();
    NormalizedLandmark *l1 = (*vector)->add_landmark();
    l1->set_x(0);
    l1->set_y(0);
    l1->set_z(0);
    NormalizedLandmark *l2 = (*vector)->add_landmark();
    l2->set_x(0);
    l2->set_y(0);
    l2->set_z(0);
  }
  return true;
}

void GraphLib::set_eye_l_offsets(float *p_offsets){
  eye_offsets_l.reset();
  eye_offsets_l = std::make_unique<NormalizedLandmarkList>();
  NormalizedLandmark *l1 = eye_offsets_l->add_landmark();
  l1->set_x(p_offsets[3]);
  l1->set_y(p_offsets[4]);
  l1->set_z(p_offsets[5]);
  NormalizedLandmark *l2 = eye_offsets_l->add_landmark();
  l2->set_x(p_offsets[0]);
  l2->set_y(p_offsets[1]);
  l2->set_z(p_offsets[2]);
}

void GraphLib::set_eye_r_offsets(float *p_offsets){
  eye_offsets_r.reset();
  eye_offsets_r = std::make_unique<NormalizedLandmarkList>();
  NormalizedLandmark *l1 = eye_offsets_r->add_landmark();
  l1->set_x(p_offsets[0]);
  l1->set_y(p_offsets[1]);
  l1->set_z(p_offsets[2]);
  NormalizedLandmark *l2 = eye_offsets_r->add_landmark();
  l2->set_x(p_offsets[3]);
  l2->set_y(p_offsets[4]);
  l2->set_z(p_offsets[5]);
}

void GraphLib::set_face_offsets(int i, float *p_offsets){
  if(face_offsets.landmark_size() == 0){
    for (int j = 0; j < 468; j++)
    {
      NormalizedLandmark *l = face_offsets.add_landmark();
      l->clear_x();
      l->clear_y();
      l->clear_z();
      l->set_presence(0.01);
    }
  }
  NormalizedLandmark *l = face_offsets.mutable_landmark(i);
  l->set_x(p_offsets[0]);
  l->set_y(p_offsets[1]);
  l->set_z(p_offsets[2]);
  l->set_presence(p_offsets[3]);
}

bool GraphLib::_in_exec() {
  ERR_MSG(graph == nullptr, "Trying to access null graph in input\n");
  size_t frame_timestamp_us = (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;


  // std::unique_ptr<NormalizedLandmarkList> face_offsets_ptr = std::make_unique<NormalizedLandmarkList>(face_offsets);
  // Status status = graph->AddPacketToInputStream(kFaceOffsets, Adopt(face_offsets_ptr.release()).At(Timestamp(frame_timestamp_us)));
  // ERR_MSG(!status.ok(), "Can't add eye offsets l input stream\n");

  // if(true){
  //   vector_init(&eye_offsets_l);
  //   std::unique_ptr<NormalizedLandmarkList> offsets_in = std::make_unique<NormalizedLandmarkList>();
  //   NormalizedLandmark *l1 = offsets_in->add_landmark();
  //   l1->set_x(eye_offsets_l->landmark(0).x());
  //   l1->set_y(eye_offsets_l->landmark(0).y());
  //   l1->set_z(eye_offsets_l->landmark(0).z());
  //   NormalizedLandmark *l2 = offsets_in->add_landmark();
  //   l2->set_x(eye_offsets_l->landmark(1).x());
  //   l2->set_y(eye_offsets_l->landmark(1).y());
  //   l2->set_z(eye_offsets_l->landmark(1).z());
  //   Status status = graph->AddPacketToInputStream(kEyeOffsetsL, Adopt(offsets_in.release()).At(Timestamp(frame_timestamp_us)));
  //   ERR_MSG(!status.ok(), "Can't add eye offsets l input stream\n");
  // }
  // if(true){
  //   vector_init(&eye_offsets_r);
  //   std::unique_ptr<NormalizedLandmarkList> offsets_in = std::make_unique<NormalizedLandmarkList>();
  //   NormalizedLandmark *l1 = offsets_in->add_landmark();
  //   l1->set_x(eye_offsets_r->landmark(0).x());
  //   l1->set_y(eye_offsets_r->landmark(0).y());
  //   l1->set_z(eye_offsets_r->landmark(0).z());
  //   NormalizedLandmark *l2 = offsets_in->add_landmark();
  //   l2->set_x(eye_offsets_r->landmark(1).x());
  //   l2->set_y(eye_offsets_r->landmark(1).y());
  //   l2->set_z(eye_offsets_r->landmark(1).z());
  //   Status status = graph->AddPacketToInputStream(kEyeOffsetsR, Adopt(offsets_in.release()).At(Timestamp(frame_timestamp_us)));
  //   ERR_MSG(!status.ok(), "Can't add eye offsets r input stream\n");
  // }
  if(input_frame != nullptr){
    Status status = graph->AddPacketToInputStream(kInputStream, Adopt(input_frame.release()).At(Timestamp(frame_timestamp_us)));
  }
  return true;
}
GraphLib::GraphLib(){
  graph = nullptr;
  video_poller = nullptr;
  geometry_poller = nullptr;
  landmarks_poller = nullptr;
  iris_landmarks_poller = nullptr;

  environment = nullptr;
  metadata = nullptr;
  geometry_pipeline = nullptr;
  eye_offsets_l = nullptr;
  eye_offsets_r = nullptr;

  _is_window_name_changed = true;
  _is_started = false;
  _finished = false;
  _is_capture_started = false;
  graph_path = "mediapipe/graphs/facegen.pbtxt";
  window_name = "Face Gen";

  input_width = 1024;
  input_height = 1024;

  camera_width = 1280;
  camera_height = 720;
  vertical_fov_degrees = 90.0;//63.0;
}