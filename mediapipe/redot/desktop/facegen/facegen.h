#ifndef FACEGEN_DLL_H
#define FACEGEN_DLL_H
#include <string>

#ifdef BUILD_DLL
#define IMPORT_EXPORT __declspec(dllexport)
#else
#define IMPORT_EXPORT __declspec(dllimport)
#endif

class Graph {
  public:
    virtual ~Graph() {;}
    virtual void set_graph_path(const char *p_graph_path) = 0;
    virtual const char *get_graph_path() = 0;
    virtual void set_window_name(const char *p_window_name) = 0;
    virtual const char *get_window_name() = 0;
    virtual void set_eye_l_offsets(float *p_offsets) = 0;
    virtual void set_eye_r_offsets(float *p_offsets) = 0;
    virtual void set_face_offsets(int i, float *p_offsets) = 0;
    
    virtual bool start() = 0;
    virtual bool start_capture() = 0;
    virtual bool finish() = 0;
    virtual void free() = 0;

    virtual bool in_camera() = 0;
    virtual bool in_texture(int width, int height, uint8_t* pixel_data) = 0;
    virtual bool out_display() = 0;
    virtual bool out_landmarks(float **landmarks, int *landmarks_size) = 0;
    virtual bool out_iris_landmarks(float **landmarks, int *landmarks_size) = 0;
    virtual bool out_polygon(float **polygon, int **polygons, int *polygon_size, int *polygons_size, float **transform_matrix) = 0;

    virtual bool is_started() = 0;
    virtual bool is_key_pressed() = 0;
};

extern "C"
{
  IMPORT_EXPORT Graph* _cdecl create_graph();
};

typedef Graph* (*GET_GRAPH) ();
#endif