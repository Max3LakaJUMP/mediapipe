"""
This is a simple windows_dll_library rule for builing a DLL Windows
that can be depended on by other cc rules.
Example useage:
  windows_dll_library(
      name = "hellolib",
      srcs = [
          "hello-library.cpp",
      ],
      hdrs = ["hello-library.h"],
      # Define COMPILING_DLL to export symbols during compiling the DLL.
      copts = ["/DCOMPILING_DLL"],
  )
"""

load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_import", "cc_library")

def windows_dll_library(
        name,
        srcs = [],
		deps = [
			# image
			#"//mediapipe/calculators/image:opencv_encoded_image_to_image_frame_calculator",
            # graph_import
            "//mediapipe/calculators/core:constant_side_packet_calculator",
            "//mediapipe/calculators/core:flow_limiter_calculator",
            "//mediapipe/redot/facegen/calculators:add_offset_to_landmarks",
			# "//mediapipe/redot/facegen/subgraphs:iris_landmark_left_and_right_cpu",

            # iris
            "//mediapipe/calculators/core:split_vector_calculator",
            "//mediapipe/redot/facegen/calculators:update_face_landmarks_calculator",
            "//mediapipe/redot/facegen/subgraphs:iris_renderer_cpu",
            "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
            "//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",

            # opencv
            "//mediapipe/framework/port:commandlineflags",
            "//mediapipe/framework/port:opencv_highgui",
            "//mediapipe/framework/port:opencv_imgproc",
            "//mediapipe/framework/port:opencv_video",
            "//mediapipe/framework/port:parse_text_proto",
			# mesh export
			"//mediapipe/modules/face_geometry",
			"//mediapipe/modules/face_geometry:env_generator_calculator",
			"//mediapipe/modules/face_geometry:geometry_pipeline_calculator",
			"//mediapipe/redot/facegen/subgraphs:single_face_smooth_landmark_cpu",
			"//mediapipe/calculators/image:image_properties_calculator",
            "//mediapipe/calculators/core:concatenate_vector_calculator",
            "//mediapipe/calculators/util:landmarks_smoothing_calculator",
            "//mediapipe/calculators/util:collection_has_min_size_calculator",

            "//mediapipe/calculators/core:previous_loopback_calculator",
            "//mediapipe/calculators/core:gate_calculator",
            "//mediapipe/calculators/core:merge_calculator",

            "//mediapipe/calculators/core:concatenate_normalized_landmark_list_calculator",
            
            "//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:image_frame",
            "//mediapipe/framework/formats:image_frame_opencv",
            "//mediapipe/framework/port:opencv_core",
            "//mediapipe/framework/port:opencv_imgcodecs",
            "//mediapipe/framework/port:ret_check",
            "//mediapipe/framework/port:status",
            "//mediapipe/framework/port:statusor",
            "//mediapipe/modules/face_geometry/libs:validation_utils",
            "//mediapipe/modules/face_geometry/protos:environment_cc_proto",
            "//mediapipe/modules/face_geometry/protos:face_geometry_cc_proto",
            "//mediapipe/modules/face_geometry/protos:mesh_3d_cc_proto",
            "//mediapipe/util:resource_util",
            "//mediapipe/framework/port:advanced_proto_lite",
            "//mediapipe/framework/port:advanced_proto",
            "//mediapipe/framework/port:any_proto",
            "@com_google_absl//absl/types:optional",

            ],
        hdrs = [],
        visibility = None,
        **kwargs):
    """A simple windows_dll_library rule for builing a DLL Windows."""
    dll_name = name + ".dll"
    import_lib_name = name + "_import_lib"
    import_target_name = name + "_dll_import"

    # Build the shared library
    cc_binary(
        name = dll_name,
        srcs = srcs + hdrs,
        deps = deps,
		defines = ["BUILD_DLL"],
		linkshared = 1,
        **kwargs
    )

    # Get the import library for the dll
    native.filegroup(
        name = import_lib_name,
        srcs = [":" + dll_name],
        output_group = "interface_library",
    )

    # Because we cannot directly depend on cc_binary from other cc rules in deps attribute,
    # we use cc_import as a bridge to depend on the dll.
    cc_import(
        name = import_target_name,
        interface_library = ":" + import_lib_name,
        shared_library = ":" + dll_name,
    )

    # Create a new cc_library to also include the headers needed for the shared library
    cc_library(
        name = name,
        hdrs = hdrs,
        visibility = visibility,
        deps = deps + [
            ":" + import_target_name,
        ],
    )