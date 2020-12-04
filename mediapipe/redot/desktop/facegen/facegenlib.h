#ifndef FACEGEN_LIB_H
#define FACEGEN_LIB_H
#include <string>
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"

#include "mediapipe/modules/face_geometry/libs/geometry_pipeline.h"
#include "mediapipe/modules/face_geometry/geometry_pipeline_calculator.pb.h"

#include "facegen.h"
using namespace ::mediapipe;
using namespace ::face_geometry;

class GraphLib: public Graph {
  private:
    CalculatorGraph *graph;
    OutputStreamPoller *video_poller;
    OutputStreamPoller *geometry_poller;
    OutputStreamPoller *landmarks_poller;
    OutputStreamPoller *iris_landmarks_poller;
    NormalizedLandmarkList face;
    std::vector<NormalizedLandmarkList> faces;
    cv::VideoCapture capture;
    std::unique_ptr<ImageFrame> input_frame;
    bool _is_started;
    bool _finished;
    bool _is_capture_started;
    bool _is_window_name_changed;
    std::string graph_path;
    std::string window_name;

    Environment *environment;
    GeometryPipelineMetadata *metadata;
    std::unique_ptr<GeometryPipeline> geometry_pipeline;

    int input_width;
    int input_height;
    int camera_width;
    int camera_height;

    int vertical_fov_degrees;
    std::unique_ptr<NormalizedLandmarkList> eye_offsets_l;
    std::unique_ptr<NormalizedLandmarkList> eye_offsets_r;

    NormalizedLandmarkList face_offsets;

  public:
    GraphLib();
    ~GraphLib(){
      if (graph != nullptr)
        finish();
        delete graph;
      if (video_poller != nullptr)
        delete video_poller;
      
      delete environment;
      delete metadata;
      input_frame.reset();
    }

    void set_graph_path(const char *p_graph_path);
    const char *get_graph_path();
    void set_window_name(const char *p_window_name);
    const char *get_window_name();
    void set_eye_l_offsets(float *p_offsets);
    void set_eye_r_offsets(float *p_offsets);

    bool start();
    bool start_capture();
    bool finish();
    void free();

    bool in_camera();
    bool in_texture(int width, int height, uint8_t* pixel_data);
    bool out_display();
    bool out_landmarks(float **landmarks, int *landmarks_size);
    bool out_iris_landmarks(float **landmarks, int *landmarks_size);
    bool out_polygon(float **polygon, int **polygons, int *polygon_size, int *polygons_size, float **transform_matrix);
    void set_face_offsets(int i, float *p_offsets);
    bool calc_environment();
    bool calc_metadata();
    bool calc_geometry_pipeline();
    bool is_started();
    bool is_key_pressed();

    bool _in_exec();
};

#endif