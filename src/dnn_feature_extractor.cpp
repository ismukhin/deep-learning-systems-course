#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/cvstd_wrapper.hpp>
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <string>
#include "dnn_feature_extractor.hpp"

cv::Ptr<DNNFeaturesExtractor> DNNFeaturesExtractor::create(const std::string& extractor_name,
                                                           const std::string& model_path,
                                                           const std::string& device,
                                                           const std::vector<std::string>& input_names,
                                                           const std::vector<std::string>& output_names,
                                                           double threshold) {

    if (extractor_name == "superpoint") {
        return cv::makePtr<SuperPoint>(model_path, input_names, output_names, device, threshold);
    } else {
        throw std::runtime_error("Method " + extractor_name + " not implemented");
    }
}

auto SuperPoint::preprocess_input(cv::Mat& ref_image) {
    cv::Mat norm_image, gray_image;
    cv::cvtColor(ref_image, gray_image, cv::COLOR_BGR2GRAY);
    gray_image.convertTo(norm_image, CV_32FC1, 1.0/255.0);
    const int H = norm_image.rows;
    const int W = norm_image.cols;
    std::vector<int64_t> input_shape = {1, 1, H, W};
    std::vector<float> input_data(1 * 1 * H * W);
    std::memcpy(input_data.data(), norm_image.ptr<float>(), input_data.size() * sizeof(float));
    return std::make_pair(input_data, input_shape);
}

auto SuperPoint::forward(std::vector<float>& input_data, std::vector<int64_t>& input_shape) {
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    const char* input_names[] = {this->input_names[0].c_str()};
    const char* output_names[] = {this->output_names[0].c_str(),
                                  this->output_names[1].c_str(),
                                  this->output_names[2].c_str()};
    std::vector<Ort::Value> ort_inputs;
    ort_inputs.push_back(std::move(input_tensor));
    return session->Run(Ort::RunOptions{nullptr}, input_names,
                        ort_inputs.data(), 1,
                        output_names, 3);
};

auto SuperPoint::postprocess(std::vector<Ort::Value>& outputs, std::vector<cv::KeyPoint>& keypoints,
                             cv::OutputArray& _descriptors, int H, int W) {
    const float* keypoints_ptr = outputs[0].GetTensorData<float>();
    const float* scores_ptr = outputs[1].GetTensorData<float>();
    const float* descriptors_ptr = outputs[2].GetTensorData<float>();
    size_t count = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
    auto keypoints_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int64_t N = keypoints_shape[1];
    cv::Mat keypoints_mat(static_cast<int>(N), 2, CV_32FC1, const_cast<float*>(keypoints_ptr));
    cv::Mat descriptors_mat_full(static_cast<int>(N), 256, CV_32FC1, const_cast<float*>(descriptors_ptr));
    cv::Mat scores_mat(static_cast<int>(N), 1, CV_32FC1, const_cast<float*>(scores_ptr));
    std::vector<int> indices;
    std::vector<float> scores_vec(scores_ptr, scores_ptr + count);
    for (int i = 0; i < keypoints_mat.rows; ++i) {
        if (scores_vec[i] >= threshold) {
            indices.push_back(i);
            float x = keypoints_mat.at<float>(i, 0);
            float y = keypoints_mat.at<float>(i, 1);
            keypoints.emplace_back(cv::KeyPoint(x * (W), y * (H), 1.0f, -1.0f, scores_vec[i]));
        }             
    }
    cv::Mat descriptors_mat;
    for (const auto& indx : indices) {
        descriptors_mat.push_back(descriptors_mat_full.row(indx));
    }
    cv::UMat umat;
    descriptors_mat.copyTo(umat);
    if (_descriptors.needed()) {
        _descriptors.create(descriptors_mat.rows, descriptors_mat.cols, descriptors_mat.type());
        cv::Mat dst = _descriptors.getMat();
        umat.copyTo(dst);
    }
}

SuperPoint::SuperPoint(const std::string& _modelPath,
                       const std::vector<std::string>& _input_names,
                       const std::vector<std::string>& _output_names, 
                       const std::string& device, double threshold_) :
                       modelPath(_modelPath),
                       input_names(_input_names),
                       output_names(_output_names),
                       threshold(threshold_),
                       memory_info(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU)),
                       env(ORT_LOGGING_LEVEL_WARNING, "test") {
        session_options = cv::makePtr<Ort::SessionOptions>();
        if (device == "cuda") {
            OrtCUDAProviderOptions cuda_options;
            cuda_options.device_id = 0;
            session_options->AppendExecutionProvider_CUDA(cuda_options);            
        }
        session = cv::makePtr<Ort::Session>(env, _modelPath.c_str(), *session_options);
        if (input_names.empty()) {
            input_names.push_back("input");
        }
        if (output_names.empty()) {
            output_names.push_back("keypoints");
            output_names.push_back("scores");
            output_names.push_back("descriptors");
        }
}

void SuperPoint::detectAndCompute(cv::InputArray _image,
                                  cv::InputArray _mask,
                                  std::vector<cv::KeyPoint>& keypoints,
                                  cv::OutputArray _descriptors, bool) {
    cv::Mat img = _image.getMat();
    auto [input_data, input_shape] = preprocess_input(img);
    auto outputs = forward(input_data, input_shape);
    postprocess(outputs, keypoints, _descriptors, img.rows, img.cols);
}
