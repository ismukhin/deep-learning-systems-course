#include "dnn_matcher.hpp"
#include <opencv2/calib3d.hpp>
#include <iostream>

LightGlue::LightGlue(const std::string& _modelPath,
               const std::vector<std::string>& _input_names,
               const std::vector<std::string>& _output_names,
               const std::string& device,
               double threshold_) : modelPath(_modelPath),
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
            input_names.push_back("kpts0");
            input_names.push_back("kpts1");
            input_names.push_back("desc0");
            input_names.push_back("desc1");
        }
        if (output_names.empty()) {
            output_names.push_back("matches0");
            output_names.push_back("matches1");
            output_names.push_back("mscores0");
            output_names.push_back("mscores1");
        }
}

void LightGlue::match(const cv::detail::ImageFeatures &features1, const cv::detail::ImageFeatures &features2, cv::detail::MatchesInfo &matches_info) {
    throw std::runtime_error("Match function not implemented between two features arguments");
}

auto LightGlue::preprocess_input(const cv::detail::ImageFeatures& features_first,
                                 const cv::detail::ImageFeatures& features_second) {
    int64_t kpI_size = features_first.keypoints.size();
    int64_t kpJ_size = features_second.keypoints.size();
    std::cout << kpI_size << std::endl;
    std::cout << kpJ_size << std::endl;
    std::vector<float> kpI_flat;
    std::vector<float> kpJ_flat;
    std::vector<int64_t> s_kpI = {1, kpI_size, 2};
    std::vector<int64_t> s_kpJ = {1, kpJ_size, 2};
    kpI_flat.reserve(1 * kpI_size * 2);
    kpJ_flat.reserve(1 * kpJ_size * 2);
    
    cv::Mat descI;
    features_first.descriptors.getMat(cv::ACCESS_READ).copyTo(descI);
    
    cv::Mat descJ;
    features_second.descriptors.getMat(cv::ACCESS_READ).copyTo(descJ);
    
    std::vector<int64_t> s_descI = {1, descI.rows, descI.cols};            
    std::vector<int64_t> s_descJ = {1, descJ.rows, descJ.cols};
    
    std::vector<float> descI_flat(1 * descI.rows * descI.cols);
    std::vector<float> descJ_flat(1 * descJ.rows * descJ.cols);
    std::memcpy(descI_flat.data(), descI.ptr<float>(), descI.total() * sizeof(float));
    std::memcpy(descJ_flat.data(), descJ.ptr<float>(), descJ.total() * sizeof(float));
    
    for (const auto& kp : features_first.keypoints) {
        kpI_flat.push_back(kp.pt.x / features_first.img_size.width);
        kpI_flat.push_back(kp.pt.y / features_first.img_size.height);
    }
    
    for (const auto& kp : features_second.keypoints) {
        kpJ_flat.push_back(kp.pt.x / features_second.img_size.width);
        kpJ_flat.push_back(kp.pt.y / features_second.img_size.height);
    }
    return std::make_tuple(kpI_flat, kpJ_flat, s_kpI, s_kpJ,
                           descI_flat, descJ_flat, s_descI, s_descJ);
}

auto LightGlue::forward(std::vector<float>& kpI_flat, std::vector<float>& kpJ_flat,
                        std::vector<int64_t>& s_kpI, std::vector<int64_t>& s_kpJ,
                        std::vector<float>& descI_flat, std::vector<float>& descJ_flat,
                        std::vector<int64_t>& s_descI, std::vector<int64_t>& s_descJ) {
    const char* input_names[] = {this->input_names[0].c_str(),
                                 this->input_names[1].c_str(),
                                 this->input_names[2].c_str(),
                                 this->input_names[3].c_str()};
    const char* output_names[] = {this->output_names[0].c_str(),
                                  this->output_names[1].c_str(),
                                  this->output_names[2].c_str(),
                                  this->output_names[3].c_str()};

    Ort::Value inputs[] = {
        Ort::Value::CreateTensor<float>(memory_info, kpI_flat.data(), kpI_flat.size(), s_kpI.data(), s_kpI.size()),
        Ort::Value::CreateTensor<float>(memory_info, kpJ_flat.data(), kpJ_flat.size(), s_kpJ.data(), s_kpJ.size()),
        Ort::Value::CreateTensor<float>(memory_info, descI_flat.data(), descI_flat.size(), s_descI.data(), s_descI.size()),
        Ort::Value::CreateTensor<float>(memory_info, descJ_flat.data(), descJ_flat.size(), s_descJ.data(), s_descJ.size())
    };
    return session->Run(Ort::RunOptions{nullptr},
                        input_names, inputs, 4, output_names, 4);
}

void LightGlue::match(const std::vector<cv::detail::ImageFeatures> &features, std::vector<cv::detail::MatchesInfo> &pairwise_matches, const cv::UMat &mask) {
    pairwise_matches.resize(features.size() * features.size());
    for (int i = 0; i < features.size(); ++i) {
        cv::detail::MatchesInfo diag;
        diag.src_img_idx = -1;
        diag.dst_img_idx = -1;
        diag.confidence = 0;
        pairwise_matches[i * features.size() + i] = diag;
    }

    for (int i = 0; i < features.size(); ++i) {
        for (int j = i + 1; j < features.size(); ++j) {
            auto [kpI_flat, kpJ_flat, s_kpI, s_kpJ,
                  descI_flat, descJ_flat, s_descI, s_descJ] = preprocess_input(features[i], features[j]);

            Ort::Value inputs[] = {
                Ort::Value::CreateTensor<float>(memory_info, kpI_flat.data(), kpI_flat.size(), s_kpI.data(), s_kpI.size()),
                Ort::Value::CreateTensor<float>(memory_info, kpJ_flat.data(), kpJ_flat.size(), s_kpJ.data(), s_kpJ.size()),
                Ort::Value::CreateTensor<float>(memory_info, descI_flat.data(), descI_flat.size(), s_descI.data(), s_descI.size()),
                Ort::Value::CreateTensor<float>(memory_info, descJ_flat.data(), descJ_flat.size(), s_descJ.data(), s_descJ.size())
            };

            auto output_values = forward(kpI_flat, kpJ_flat,
                                         s_kpI, s_kpJ,
                                         descI_flat, descJ_flat,
                                         s_descI, s_descJ);

            auto* matches0 = output_values[0].GetTensorMutableData<int64_t>();
            auto* matches1 = output_values[1].GetTensorMutableData<int64_t>();
            auto* mscores0 = output_values[2].GetTensorMutableData<float>();
            auto* mscores1 = output_values[3].GetTensorMutableData<float>();
            size_t num_matches1 = output_values[0].GetTensorTypeAndShapeInfo().GetElementCount();
            size_t num_matches2 = output_values[1].GetTensorTypeAndShapeInfo().GetElementCount();
            cv::detail::MatchesInfo info;
            info.src_img_idx = i;
            info.dst_img_idx = j;
            double total_conf = 0.0;
            for (int l = 0; l < num_matches1; l++) {
                int t = static_cast<int>(matches0[l]);
                if (t >= 0 && t < (int)features[j].keypoints.size()) {
                    float prob = mscores0[l];
                    if (prob >= threshold)
                        info.matches.push_back(cv::DMatch(static_cast<int>(l), t, 1.0f - prob));
                }
            }
            if (!info.matches.empty()) {
                std::vector<cv::Point2f> pts1, pts2;
                for (const auto& m : info.matches) {
                    pts1.push_back(features[i].keypoints[m.queryIdx].pt);
                    pts2.push_back(features[j].keypoints[m.trainIdx].pt);
                }
                if (pts1.size() >= 4) {
                    info.H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, info.inliers_mask);
                    int inliers = std::count(info.inliers_mask.begin(), info.inliers_mask.end(), 1);
                    info.num_inliers = inliers;
                    info.confidence = info.num_inliers / (8 + 0.3 * info.matches.size());
                } else {
                    info.H = cv::Mat::eye(3, 3, CV_64F);
                    info.num_inliers = 0;
                    info.confidence = 0;
                }
            } else{
                    info.H = cv::Mat::eye(3, 3, CV_64F);
                    info.num_inliers = 0;
                    info.confidence = 0;      
            }
            pairwise_matches[i * features.size() + j] = info;

            cv::detail::MatchesInfo info1;
            for (int l = 0; l < num_matches1; l++) {
                int t = static_cast<int>(matches0[l]);
                if (t >= 0 && t < (int)features[j].keypoints.size()) {
                    float prob = mscores0[l];
                    if (prob >= threshold)
                        info1.matches.push_back(cv::DMatch(t, static_cast<int>(l), 1.0f - prob));
                }
            }
            info1.src_img_idx = j;
            info1.dst_img_idx = i;
            info1.H = info.H.inv();
            info1.num_inliers = info.num_inliers;
            info1.confidence = info.confidence;
            pairwise_matches[j * features.size() + i] = info1;
        }
    }
}

cv::Ptr<DNNFeaturesMatcher> DNNFeaturesMatcher::create(const std::string& matcher_name,
                                                       const std::string& model_path,
                                                       const std::string& device,
                                                       const std::vector<std::string>& input_names,
                                                       const std::vector<std::string>& output_names,
                                                       double threshold) {
    if (matcher_name == "lightglue") {
        return cv::makePtr<LightGlue>(model_path, input_names, output_names, device, threshold);
    } else {
        throw std::runtime_error("Not implemented");
    }
}
