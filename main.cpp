#include <iostream>
#include <string>
#include <memory>
#include <atomic>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"

mediapipe::Status run() {
    using namespace std;
    using namespace mediapipe;

    string protoG = R"(
        input_stream: "in",
        output_stream: "out",
        node {
            calculator: "PassThroughCalculator",
            input_stream: "in",
            output_stream: "out",
        }
        )";

    const char* model_path = "/home/narinder/Documents/mediapipe/mediapipe/graphs/holistic_tracking/holistic_tracking_cpu.pbtxt";
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(file::GetContents(model_path, &calculator_graph_config_contents));
    std::cout << "mediapipe::file::GetContents success" << std::endl;

    /*CalculatorGraphConfig config;
    if (!ParseTextProto<mediapipe::CalculatorGraphConfig>(protoG, &config))
    {
        return absl::InternalError("Cannot parse the graph config !");
    }*/
    CalculatorGraphConfig config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(calculator_graph_config_contents);

    CalculatorGraph graph;

    MP_RETURN_IF_ERROR(graph.Initialize(config));
    auto cb = [](const Packet &packet)->Status{
        cout << packet.Timestamp() << ": RECEIVED VIDEO PACKET !" << endl;

        const ImageFrame & outputFrame = packet.Get<ImageFrame>();
        cv::Mat ofMat = formats::MatView(&outputFrame);

        cv::Mat frameOut;
        cvtColor(ofMat, frameOut, cv::COLOR_RGB2BGR);

        double scale_x = 0.1;
        double scale_y = 0.1;

        cv::Mat resized_down;
        cv::resize(frameOut, resized_down, cv::Size(),scale_x, scale_y, cv::INTER_LINEAR);

        cv::imshow("frameOut", resized_down);

        if (27 == cv::waitKey(1))
            return absl::CancelledError("It's time to QUIT !");
        else
            return OkStatus();
    };

    MP_RETURN_IF_ERROR(graph.ObserveOutputStream("output_video", cb));
    //MP_RETURN_IF_ERROR(graph.ObserveOutputStream("out", cb));
    graph.StartRun({});

    cv::VideoCapture cap("/home/narinder/Documents/mediapipe/mediapipe/examples/first_steps/2_1/video.mp4");

    if (!cap.isOpened())
            return absl::NotFoundError("CANNOT OPEN CAMERA !");
    cv::Mat frameIn, frameInRGB;

    for (int i=0; ; ++i)
    {
        cap.read(frameIn);

        if (frameIn.empty())
            return absl::NotFoundError("CANNOT OPEN CAMERA !");

        cv::cvtColor(frameIn, frameInRGB, cv::COLOR_BGR2RGB);

        ImageFrame *inputFrame =  new ImageFrame(ImageFormat::SRGB, frameInRGB.cols, frameInRGB.rows, ImageFrame::kDefaultAlignmentBoundary);

        frameInRGB.copyTo(formats::MatView(inputFrame));

        uint64 ts = i;
        //MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("in",Adopt(inputFrame).At(Timestamp(ts))));
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream("input_video",Adopt(inputFrame).At(Timestamp(ts))));
    }
}

int main(int argc, char** argv)
{
    using namespace std;

    cout << "Example 2.1 : Video pipeline" << endl;
    mediapipe::Status status = run();
    cout << "status =" << status << endl;
    cout << "status.ok() = " << status.ok() << endl;
    return 0;
}
