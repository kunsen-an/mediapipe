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
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.

// This file is based on mediapipe/examples/desktop/demo_run_graph_main.cc and
// https://dev.classmethod.jp/articles/mediapipe-extract-data-from-multi-hand-tracking/ .

#include <cstdlib>

#include <iostream>
#include <filesystem>
#include <fstream>
#include <sys/stat.h>
#include <cstdio>
#include <regex>


#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kWindowName[] = "MediaPipe";


#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/timestamp.h"

#include <direct.h>

constexpr char kOutputPalmDetections[] = "output_palm_detections";
constexpr char kOutputLandmarks[] = "output_landmarks";
constexpr char kOutputPalmRects[] = "output_palm_rects";
constexpr char kOutputHandRects[] = "output_hand_rects";


ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

std::string output_dirpath = std::string("./result");
std::string output_data_extension = std::string(".bin");


std::string outputFilePath(mediapipe::Timestamp timestamp, std::string type, int index, std::string postfix) {
	std::ostringstream os;

	os << output_dirpath + "/"
		<< timestamp << "_"
		<< type << "_"
		<< std::to_string(index)
		<< postfix;
	return os.str();
}

std::string outputFilePath(mediapipe::Timestamp timestamp, std::string postfix) {
	std::ostringstream os;

	os << output_dirpath + "/"
		<< timestamp << "_"
		<< postfix;
	return os.str();
}

template<typename TVector>
bool
processTypeList(mediapipe::OutputStreamPoller& poller, std::string type, bool save) {
	bool ret = false;
	mediapipe::Packet	packet;
	// process landmarks
	if ( poller.QueueSize() > 0 && poller.Next(&packet) ) {
		if ( poller.QueueSize() > 1 ) LOG(WARNING) << "QueueSize: " << poller.QueueSize();
		auto &output = packet.Get<TVector>();
		if ( save ) {	// save data to output file if save is true
			for (int j = 0; j < output.size(); j++)
			{
				std::string filePath = outputFilePath(packet.Timestamp(), type, j,output_data_extension);
				std::ofstream outputfile(filePath);

				std::string serializedStr;
				output[j].SerializeToString(&serializedStr);
				outputfile << serializedStr << std::flush;
			}
			ret = save;
		}
	}
	return ret;
}


absl::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      absl::GetFlag(FLAGS_calculator_graph_config_file),
      &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !absl::GetFlag(FLAGS_input_video_path).empty();
  if (load_video) {
    capture.open(absl::GetFlag(FLAGS_input_video_path));
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !absl::GetFlag(FLAGS_output_video_path).empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  LOG(INFO) << "Start running the calculator graph.";


  // Connect pollers to graph
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_video,
                   graph.AddOutputStreamPoller(kOutputStream));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_palm_detections,
                   graph.AddOutputStreamPoller(kOutputPalmDetections));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_landmarks,
                   graph.AddOutputStreamPoller(kOutputLandmarks));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_palm_rects,
                   graph.AddOutputStreamPoller(kOutputPalmRects));
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller_hand_rects,
                   graph.AddOutputStreamPoller(kOutputHandRects));

  
  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;

  // define output folder path
  std::string flag_output_video_path = absl::GetFlag(FLAGS_output_video_path);
  std::filesystem::path output_video_path = flag_output_video_path;
  std::string output_video_dir = output_video_path.parent_path().string();
  LOG(INFO) << "output_video_dir:" << output_video_dir;

  if ( ! output_video_dir.empty() ) {
	  output_dirpath =  output_video_dir;
  }
  _mkdir(output_dirpath.c_str()); // for windows
  
  while (grab_frames)
  {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;

    if (camera_frame_raw.empty()) {
      if (!load_video) {
        LOG(INFO) << "Ignore empty frames from camera.";
        continue;
      }
      LOG(INFO) << "Empty frame, end of video reached.";
      break;
    }
    cv::Mat camera_frame, flipped_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Send image packet into the graph.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
        kInputStream, mediapipe::Adopt(input_frame.release())
                          .At(mediapipe::Timestamp(frame_timestamp_us))));



    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet_video;

    // check if data exist from graph
    if (!poller_video.Next(&packet_video) ) {
		break;
	}

    // get output from graph
    auto &output_video = packet_video.Get<mediapipe::ImageFrame>();
	auto packet_video_timestamp =  packet_video.Timestamp();
	LOG(INFO) << "packet_video.Timestamp:" << packet_video_timestamp;
	
	// save_data is true if all data is available (If partial data should be saved, change the following conditions)
	bool save_data = (poller_palm_detections.QueueSize() > 0)
					&& (poller_palm_rects.QueueSize() > 0)
					&& (poller_hand_rects.QueueSize() > 0)
					&& (poller_landmarks.QueueSize() > 0);

	// process palm detections
	processTypeList<std::vector<mediapipe::Detection>>(poller_palm_detections, std::string("palm_detections"), save_data);

	// process palm rects
	processTypeList<std::vector<mediapipe::NormalizedRect>>(poller_palm_rects, std::string("palm_rects"), save_data);

	// process hand rects
	processTypeList<std::vector<mediapipe::NormalizedRect>>(poller_hand_rects, std::string("hand_rects"), save_data);

	// process landmarks
	processTypeList<std::vector<mediapipe::NormalizedLandmarkList>>(poller_landmarks, std::string("landmarks"), save_data);


	
    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_video);
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);
    if (save_video)
    {
      if (!writer.isOpened())
      {
        LOG(INFO) << "Prepare video writer:" << flag_output_video_path;
        writer.open(flag_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'), // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    }


	// show output image
	cv::imshow(kWindowName, output_frame_mat);
	// Press any key to exit.
	const int pressed_key = cv::waitKey(5);
	if (pressed_key >= 0 && pressed_key != 255)
		grab_frames = false;


	if ( !save_data ) continue;

    // save input frame to file
	std::string inputFramePath = outputFilePath(mediapipe::Timestamp(frame_timestamp_us), std::string("inputFrame.jpg"));
    cv::imwrite(inputFramePath, input_frame_mat);
	
    // save output frame to file
	std::string outputFramePath = outputFilePath(packet_video_timestamp, "outputFrame.jpg");
    cv::imwrite(outputFramePath, output_frame_mat);
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  absl::ParseCommandLine(argc, argv);
  absl::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
