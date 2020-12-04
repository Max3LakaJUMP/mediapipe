// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include <tuple>

namespace {
constexpr char kLandmarks[] = "LANDMARKS";
constexpr char kOffsets[] = "OFFSETS";
constexpr char kUpdatedLandmarks[] = "UPDATED_LANDMARKS";

}  // namespace

namespace mediapipe {

class AddOffsetsToLandmarks : public CalculatorBase {
 public:
  static ::mediapipe::Status GetContract(CalculatorContract* cc) {
    printf("GetContract\n");
    cc->Inputs().Tag(kLandmarks).Set<NormalizedLandmarkList>();
    printf("GetContract end\n");
    cc->Inputs().Tag(kOffsets).Set<NormalizedLandmarkList>();
    cc->Outputs().Tag(kUpdatedLandmarks).Set<NormalizedLandmarkList>();
    // if (cc->Outputs().HasTag(kOffsets)) {
    //   
    // }

    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    return ::mediapipe::OkStatus();
  }

  ::mediapipe::Status Process(CalculatorContext* cc) override {
    if (cc->Inputs().Tag(kLandmarks).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }
    if (cc->Inputs().Tag(kOffsets).IsEmpty()) {
      return ::mediapipe::OkStatus();
    }

    const auto& face_landmarks = cc->Inputs().Tag(kLandmarks).Get<NormalizedLandmarkList>();
    const auto& offsets = cc->Inputs().Tag(kOffsets).Get<NormalizedLandmarkList>();

    auto refined_face_landmarks = absl::make_unique<NormalizedLandmarkList>(face_landmarks);

    int offsets_size = face_landmarks.landmark_size();
    for (int i = 0; i < offsets_size; i++) {
      const auto& l = face_landmarks.landmark(i);
      const auto& o = offsets.landmark(i);
      const auto& m = refined_face_landmarks->mutable_landmark(i);
      m->set_x(l.x() + o.x());
      m->set_y(l.y() + o.y());
      m->set_z(l.z() + o.z());
    }
    cc->Outputs().Tag(kUpdatedLandmarks).Add(refined_face_landmarks.release(), cc->InputTimestamp());

    return ::mediapipe::OkStatus();
  }
};
REGISTER_CALCULATOR(AddOffsetsToLandmarks);

}  // namespace mediapipe
