#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/features2d.hpp>
#include "dnn_feature_extractor.hpp"
#include "dnn_matcher.hpp"

#include <iostream>
#include <gflags/gflags.h>

DEFINE_string(model_fe, "", "Path to the ONNX feature extractor model file");
DEFINE_string(model_mat, "", "Path to the ONNX matcher model file");
DEFINE_string(extractor, "superpoint", "Type of feature extractor to use (e.g., superpoint, disk, orb)");
DEFINE_string(matcher, "lightglue", "Type of matcher to use (e.g., lightglue, bf)");
DEFINE_string(device, "cpu", "Computation device: 'cpu' or 'cuda'");
DEFINE_string(ins_fe, "", "Input layer names of feature extractor separated by space in quotes (e.g., \"input1 input2\")");
DEFINE_string(outs_fe, "", "Output layer names of feature extractor separated by space in quotes (e.g., \"output1\")");
DEFINE_string(ins_mat, "", "Input layer names of matcher separated by space in quotes (e.g., \"input1 input2\")");
DEFINE_string(outs_mat, "", "Output layer names of matcher separated by space in quotes (e.g., \"output1\")");
DEFINE_double(f_thr, -1.0, "Feature extraction threshold (float)");
DEFINE_double(m_thr, -1.0, "Matching threshold (float)");
DEFINE_bool(d3, false, "Internally creates three chunks of each image to increase stitching success");
DEFINE_string(mode, "panorama", "Stitcher configuration: 'panorama' (default) or 'scans' (affine transformation)");
DEFINE_string(output, "result.jpg", "Name of the resulting image file");
DEFINE_bool(h, false, "Print this help message");

struct ArgsValues {
    std::string model_path_fe;
    std::string model_path_mat;
    std::string feature_extractor;
    std::string matcher;
    std::string device;
    std::vector<std::string> input_names_fe;
    std::vector<std::string> output_names_fe;
    std::vector<std::string> input_names_mat;
    std::vector<std::string> output_names_mat;
    double feature_threshold = -1.0;
    double matcher_threshold = -1.0;
};

bool divide_images = false;
cv::Stitcher::Mode mode = cv::Stitcher::PANORAMA;
std::vector<cv::Mat> imgs;
std::vector<std::string> paths;
std::string result_name = "result.jpg";

std::vector<std::string> tokenize(const std::string& str) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string temp;
    while (ss >> temp) tokens.push_back(temp);
    return tokens;
}

void read_images() {
    for (const auto& filename : paths) {
        cv::Mat img = cv::imread(cv::samples::findFile(filename));
        if (img.empty()) {
            std::cerr << "Can't read image: " << filename << std::endl;
        }
        if (divide_images) {
            imgs.push_back(img(cv::Rect(0, 0, img.cols / 2, img.rows)).clone());
            imgs.push_back(img(cv::Rect(img.cols / 3, 0, img.cols / 2, img.rows)).clone());
            imgs.push_back(img(cv::Rect(img.cols / 2, 0, img.cols / 2, img.rows)).clone());
        } else {
            imgs.push_back(img);
        }
    }
}

int parseCmdArgs(int argc, char** argv, ArgsValues& global_args) {
    if (FLAGS_h) {
        std::cout << "Advanced Images Stitcher\n\n"
                  << "Usage: " << argv[0] << " [OPTIONS] image1 image2 ... imageN\n\n"
                  << "Options:\n"
                  << "  --help                  Show this help message\n"
                  << "  --model                 Path to the ONNX model file\n"
                  << "  --extractor             Type of feature extractor (default: superpoint)\n"
                  << "  --matcher               Type of matcher (default: lightglue)\n"
                  << "  --device                Computation device: cpu or cuda (default: cpu)\n"
                  << "  --ins_fe                Input layer names for feature extractor (quoted, space-separated)\n"
                  << "  --outs_fe               Output layer names for feature extractor (quoted, space-separated)\n"
                  << "  --ins_mat               Input layer names for matcher (quoted, space-separated)\n"
                  << "  --outs_mat              Output layer names for matcher (quoted, space-separated)\n"
                  << "  --f_thr                 Feature extraction threshold (default: -1.0)\n"
                  << "  --m_thr                 Matching threshold (default: -1.0)\n"
                  << "  --d3                    Internally split each image into three chunks\n"
                  << "  --mode                  Stitcher mode: panorama or scans (default: panorama)\n"
                  << "  --output                Output image filename (default: result.jpg)\n"
                  << "\nPositional arguments:\n"
                  << "  image1 image2 ...       Input images for stitching\n";
        return EXIT_FAILURE;
    }

    global_args.model_path_fe = FLAGS_model_fe;
    global_args.model_path_mat = FLAGS_model_mat;
    global_args.feature_extractor = FLAGS_extractor;
    global_args.matcher = FLAGS_matcher;
    global_args.device = FLAGS_device;
    global_args.feature_threshold = FLAGS_f_thr;
    global_args.matcher_threshold = FLAGS_m_thr;
    global_args.input_names_fe = tokenize(FLAGS_ins_fe);
    global_args.output_names_fe = tokenize(FLAGS_outs_fe);
    global_args.input_names_mat = tokenize(FLAGS_ins_mat);
    global_args.output_names_mat = tokenize(FLAGS_outs_mat);

    divide_images = FLAGS_d3;
    result_name = FLAGS_output;

    std::string mode_str = FLAGS_mode;
    if (mode_str == "panorama") mode = cv::Stitcher::PANORAMA;
    else if (mode_str == "scans") mode = cv::Stitcher::SCANS;
    else {
        std::cerr << "Bad --mode value: " << mode_str << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 1; i < argc; ++i) {
        paths.push_back(argv[i]);
    }

    if (paths.empty()) {
        std::cerr << "No input images provided!" << std::endl;
        return EXIT_FAILURE;
    }

    read_images();
    if (imgs.empty()) {
        std::cerr << "Failed to read any input images!" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

cv::Ptr<cv::Feature2D> get_feature_extractor(ArgsValues& args) {
    if (args.feature_extractor == "superpoint") {
        return DNNFeaturesExtractor::create(args.feature_extractor,
                                            args.model_path_fe,
                                            args.device,
                                            args.input_names_fe,
                                            args.output_names_fe,
                                            args.feature_threshold);
    } else {
        throw std::runtime_error("Invalid feature extractor type");
    }
}

cv::Ptr<cv::detail::FeaturesMatcher> get_feature_matcher(ArgsValues& args) {
    if (args.matcher == "lightglue") {
        return DNNFeaturesMatcher::create(args.matcher,
                                          args.model_path_mat,
                                          args.device,
                                          args.input_names_mat,
                                          args.output_names_mat,
                                          args.matcher_threshold);
    } else {
        throw std::runtime_error("Invalid matcher type");
    }
}

int main(int argc, char* argv[]) {
    google::ParseCommandLineFlags(&argc, &argv, true);

    ArgsValues parsed_args;
    if (parseCmdArgs(argc, argv, parsed_args) == EXIT_FAILURE)
        return EXIT_FAILURE;

    cv::Mat pano;
    auto superpoint_extractor = get_feature_extractor(parsed_args);
    auto lightglue_matcher = get_feature_matcher(parsed_args);
    cv::Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(mode);
    stitcher->setFeaturesFinder(superpoint_extractor);
    stitcher->setFeaturesMatcher(lightglue_matcher);

    std::cout << "Size of images: " << imgs.size() << std::endl;
    cv::Stitcher::Status status = stitcher->stitch(imgs, pano);

    if (status != cv::Stitcher::OK) {
        std::cout << "Can't stitch images, error code = " << int(status) << std::endl;
        return EXIT_FAILURE;
    }

    imwrite(result_name, pano);
    std::cout << "stitching completed successfully\n" << result_name << " saved!" << std::endl;

    cv::Mat img = cv::imread(cv::samples::findFile("../result.jpg"));
    double diff = cv::norm(img, pano, cv::NORM_L1);
    std::cout << img.size() << std::endl;
    std::cout << pano.size() << std::endl;
    std::cout << diff << std::endl;
    if (diff == 0) {
        std::cout << "Test succeed" << std::endl;
        return EXIT_SUCCESS;
    } else {
        std::cout << "Images not equal" << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
