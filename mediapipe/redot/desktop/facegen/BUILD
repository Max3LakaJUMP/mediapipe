# Copyright 2019 The MediaPipe Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

licenses(["notice"])

package(default_visibility = ["//mediapipe/examples:__subpackages__"])

cc_binary(
    name = "facegen_cpu",
    deps = [
        "//mediapipe/redot/desktop:demo_run_graph_main",
        # graph_import
        "//mediapipe/calculators/core:constant_side_packet_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        # "//mediapipe/redot/facegen/subgraphs:face_renderer_cpu", iris_tracking/subgraphs
        
        # iris
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/redot/facegen/calculators:update_face_landmarks_calculator",
        "//mediapipe/redot/facegen/subgraphs:iris_renderer_cpu",
        "//mediapipe/modules/face_landmark:face_landmark_front_cpu",
        "//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",

        # mesh export
        # "//mediapipe/modules/face_geometry",
        # "//mediapipe/modules/face_geometry:env_generator_calculator",
        # "//mediapipe/redot/facegen/subgraphs:single_face_smooth_landmark_cpu",
        # "//mediapipe/calculators/image:image_properties_calculator",
    ],
)

load("//mediapipe/redot/desktop/facegen:windows_dll_library.bzl", "windows_dll_library")
load("@rules_cc//cc:defs.bzl", "cc_binary")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//examples:__pkg__"],
)

windows_dll_library(
    name = "facegen",
    srcs = [

        "facegen.cc", 
        "facegenlib.cc",
    ],
    hdrs = [
        "meta.h",
        "facegen.h", 
        "facegenlib.h",
    ],
    copts = ["/DCOMPILING_DLL"],
)

exports_files(
    srcs = [
        "modules/face_detection/face_detection_front.tflite",
    ],
)
cc_library(
    name = "facegen.lib",
    srcs = ["facegen_library2.cc"],
	hdrs = ["facegen_library2.h"],
	deps = [
            # graph_import
            "//mediapipe/calculators/core:constant_side_packet_calculator",
            "//mediapipe/calculators/core:flow_limiter_calculator",
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
		],
    # deps = ["//mediapipe/framework:calculator_framework",
	# 		"//mediapipe/framework/port:commandlineflags",
	# 		"//mediapipe/framework/port:file_helpers",
	# 		"//mediapipe/framework/port:map_util",
	# 		"//mediapipe/framework/port:parse_text_proto",
	# 		"//mediapipe/framework/port:ret_check",
	# 		"//mediapipe/framework/port:status",
	# 		"//mediapipe/framework/port:statusor",
	# 		"@com_google_absl//absl/strings",
	# 		# graph_import
	# 		"//mediapipe/calculators/core:constant_side_packet_calculator",
	# 		"//mediapipe/calculators/core:flow_limiter_calculator",
	# 		# "//mediapipe/redot/facegen/subgraphs:face_renderer_cpu", iris_tracking/subgraphs
			
	# 		# iris
	# 		"//mediapipe/calculators/core:split_vector_calculator",
	# 		"//mediapipe/redot/facegen/calculators:update_face_landmarks_calculator",
	# 		"//mediapipe/redot/facegen/subgraphs:iris_renderer_cpu",
	# 		"//mediapipe/modules/face_landmark:face_landmark_front_cpu",
	# 		"//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",

	# 		# mesh export
	# 		# "//mediapipe/modules/face_geometry",
	# 		# "//mediapipe/modules/face_geometry:env_generator_calculator",
	# 		# "//mediapipe/redot/facegen/subgraphs:single_face_smooth_landmark_cpu",
	# 		# "//mediapipe/calculators/image:image_properties_calculator",
    # ],
)

cc_library(
    name = "facegen0",
    srcs = ["facegen_library.cc"],
    hdrs = ["facegen_library.h"],
    deps = ["//mediapipe/framework:calculator_framework",
			"//mediapipe/framework/port:commandlineflags",
			"//mediapipe/framework/port:file_helpers",
			"//mediapipe/framework/port:map_util",
			"//mediapipe/framework/port:parse_text_proto",
			"//mediapipe/framework/port:ret_check",
			"//mediapipe/framework/port:status",
			"//mediapipe/framework/port:statusor",
			"@com_google_absl//absl/strings",
			# graph_import
			"//mediapipe/calculators/core:constant_side_packet_calculator",
			"//mediapipe/calculators/core:flow_limiter_calculator",
			# "//mediapipe/redot/facegen/subgraphs:face_renderer_cpu", iris_tracking/subgraphs
			
			# iris
			"//mediapipe/calculators/core:split_vector_calculator",
			"//mediapipe/redot/facegen/calculators:update_face_landmarks_calculator",
			"//mediapipe/redot/facegen/subgraphs:iris_renderer_cpu",
			"//mediapipe/modules/face_landmark:face_landmark_front_cpu",
			"//mediapipe/modules/iris_landmark:iris_landmark_left_and_right_cpu",

			# mesh export
			# "//mediapipe/modules/face_geometry",
			# "//mediapipe/modules/face_geometry:env_generator_calculator",
			# "//mediapipe/redot/facegen/subgraphs:single_face_smooth_landmark_cpu",
			# "//mediapipe/calculators/image:image_properties_calculator",
    ],
)