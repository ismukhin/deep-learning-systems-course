#pragma once
#include <opencv2/stitching/detail/matchers.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "dnn_feature_extractor.hpp"

class DNNFeaturesMatcher : public cv::detail::FeaturesMatcher {
public:

    static cv::Ptr<DNNFeaturesMatcher> create(const std::string& extractor_name,
                                              const std::string& model_path,
                                              const std::string& device,
                                              const std::vector<std::string>& input_names,
                                              const std::vector<std::string>& output_names,
                                              double threshold);
};

class LightGlue : public DNNFeaturesMatcher {
public:
    LightGlue(const std::string& _modelPath,
              const std::vector<std::string>& _input_names,
              const std::vector<std::string>& _output_names,
              const std::string& device,
              double threshold_);

protected:
    void match(const cv::detail::ImageFeatures &features1,
               const cv::detail::ImageFeatures &features2,
               cv::detail::MatchesInfo &matches_info) override;
    void match(const std::vector<cv::detail::ImageFeatures> &features,
               std::vector<cv::detail::MatchesInfo> &pairwise_matches,
               const cv::UMat &mask = cv::UMat()) override;

private:
    auto preprocess_input(const cv::detail::ImageFeatures& features_first,
                          const cv::detail::ImageFeatures& features_second);

    auto forward(std::vector<float>& kpI_flat, std::vector<float>& kpJ_flat,
                 std::vector<int64_t>& s_kpI, std::vector<int64_t>& s_kpJ,
                 std::vector<float>& descI_flat, std::vector<float>& descJ_flat,
                 std::vector<int64_t>& s_descI, std::vector<int64_t>& s_descJ);


    std::string modelPath;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    double threshold;
    Ort::MemoryInfo memory_info;
    Ort::Env env;
    cv::Ptr<Ort::Session> session;
    cv::Ptr<Ort::SessionOptions> session_options;

};
