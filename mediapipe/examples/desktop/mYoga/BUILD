cc_binary(
    name="mYoga",
    srcs=["main.cpp"],
    deps=[
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/calculators/core:pass_through_calculator",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",

        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/graphs/pose_tracking/subgraphs:pose_renderer_cpu",
        "//mediapipe/modules/pose_landmark:pose_landmark_cpu",
    ],
)
