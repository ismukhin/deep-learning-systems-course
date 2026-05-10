#pragma once
#include <opencv2/features2d.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <string>
#include <onnxruntime_cxx_api.h>

class DNNFeaturesExtractor : public cv::Feature2D {
public:
    static cv::Ptr<DNNFeaturesExtractor> create(const std::string& extractor_name,
                                                const std::string& model_path,
                                                const std::string& device,
                                                const std::vector<std::string>& input_names,
                                                const std::vector<std::string>& output_names,
                                                double threshold);

    void detect(cv::InputArray image,
                std::vector<cv::KeyPoint>& keypoints,
                cv::InputArray mask) override {

        if(image.empty()) {
            keypoints.clear();
            return;
        }
    }

    virtual cv::String getDefaultName() const override {
        return "DNNFeaturesExtractor";
    }
};

class SuperPoint : public DNNFeaturesExtractor {
private:
    auto preprocess_input(cv::Mat& ref_image);

    auto forward(std::vector<float>& input_data, std::vector<int64_t>& input_shape);

    auto postprocess(std::vector<Ort::Value>& outputs,
                     std::vector<cv::KeyPoint>& keypoints,
                     cv::OutputArray& _descriptors,
                     int H, int W);

    std::string modelPath;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    double threshold;
    Ort::MemoryInfo memory_info;
    Ort::Env env;
    cv::Ptr<Ort::Session> session;
    cv::Ptr<Ort::SessionOptions> session_options;

public:
    SuperPoint(const std::string& _modelPath,
               const std::vector<std::string>& _input_names,
               const std::vector<std::string>& _output_names,
               const std::string& device,
               double threshold_);

    void detectAndCompute(cv::InputArray _image, cv::InputArray _mask,
                          std::vector<cv::KeyPoint>& keypoints,
                          cv::OutputArray _descriptors,
                          bool /*useProvidedKeypoints*/) override;

    std::string getDefaultName() const override {
        return "Feature2D.SuperPoint";
    }

};
