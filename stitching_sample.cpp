#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include <opencv2/calib3d.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include <onnxruntime/onnxruntime_cxx_api.h>

#ifdef HAVE_OPENCV_XFEATURES2D
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#endif

#define ENABLE_LOG 1
#define LOG_cv(msg) std::cout << msg
#define LOGLN(msg) std::cout << msg << std::endl

using namespace std;
using namespace cv;
using namespace cv::detail;

namespace my {
std::vector<int> leaveBiggestComponent(std::vector<ImageFeatures> &features,  std::vector<MatchesInfo> &pairwise_matches,
                                      float conf_threshold)
{
    const int num_images = static_cast<int>(features.size());
    std::cout << num_images << std::endl;
    DisjointSets comps(num_images);
    for (int i = 0; i < num_images; ++i)
    {
        for (int j = 0; j < num_images; ++j)
        {
            if (pairwise_matches[i*num_images + j].confidence < conf_threshold)
                continue;
            int comp1 = comps.findSetByElem(i);
            int comp2 = comps.findSetByElem(j);
            if (comp1 != comp2)
                comps.mergeSets(comp1, comp2);
        }
    }
    std::cout << "Cycle 1 end" << std::endl;
    int max_comp = static_cast<int>(std::max_element(comps.size.begin(), comps.size.end()) - comps.size.begin());

    std::vector<int> indices;
    std::vector<int> indices_removed;
    for (int i = 0; i < num_images; ++i)
        if (comps.findSetByElem(i) == max_comp)
            indices.push_back(i);
        else
            indices_removed.push_back(i);
    std::cout << "Cycle 2 end" << std::endl;
    std::vector<ImageFeatures> features_subset;
    std::vector<MatchesInfo> pairwise_matches_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        std::cout << indices[i] << " " << features.size() << std::endl;
        features_subset.push_back(features[indices[i]]);
        for (size_t j = 0; j < indices.size(); ++j)
        {
            pairwise_matches_subset.push_back(pairwise_matches[indices[i]*num_images + indices[j]]);
            pairwise_matches_subset.back().src_img_idx = static_cast<int>(i);
            pairwise_matches_subset.back().dst_img_idx = static_cast<int>(j);
        }
    }
    for (const auto& elem : indices) std::cout << elem << std::endl;
    if (static_cast<int>(features_subset.size()) == num_images)
        return indices;

    //LOG("Removed some images, because can't match them or there are too similar images: (");
    //LOG(indices_removed[0] + 1);
    //for (size_t i = 1; i < indices_removed.size(); ++i)
    //    LOG(", " << indices_removed[i]+1);
    //LOGLN(").");
    //LOGLN("Try to decrease the match confidence threshold and/or check if you're stitching duplicates.");

    features = features_subset;
    pairwise_matches = pairwise_matches_subset;

    return indices;
}
}
static void printUsage(char** argv)
{
    cout <<
        "Rotation model images stitcher.\n\n"
         << argv[0] << " img1 img2 [...imgN] [flags]\n\n"
        "Flags:\n"
        "  --preview\n"
        "      Run stitching in the preview mode. Works faster than usual mode,\n"
        "      but output image will have lower resolution.\n"
        "  --try_cuda (yes|no)\n"
        "      Try to use CUDA. The default value is 'no'. All default values\n"
        "      are for CPU mode.\n"
        "\nMotion Estimation Flags:\n"
        "  --work_megapix <float>\n"
        "      Resolution for image registration step. The default is 0.6 Mpx.\n"
        "  --features (surf|orb|sift|akaze)\n"
        "      Type of features used for images matching.\n"
        "      The default is surf if available, orb otherwise.\n"
        "  --matcher (homography|affine)\n"
        "      Matcher used for pairwise image matching.\n"
        "  --estimator (homography|affine)\n"
        "      Type of estimator used for transformation estimation.\n"
        "  --match_conf <float>\n"
        "      Confidence for feature matching step. The default is 0.65 for surf and 0.3 for orb.\n"
        "  --conf_thresh <float>\n"
        "      Threshold for two images are from the same panorama confidence.\n"
        "      The default is 1.0.\n"
        "  --ba (no|reproj|ray|affine)\n"
        "      Bundle adjustment cost function. The default is ray.\n"
        "  --ba_refine_mask (mask)\n"
        "      Set refinement mask for bundle adjustment. It looks like 'x_xxx',\n"
        "      where 'x' means refine respective parameter and '_' means don't\n"
        "      refine one, and has the following format:\n"
        "      <fx><skew><ppx><aspect><ppy>. The default mask is 'xxxxx'. If bundle\n"
        "      adjustment doesn't support estimation of selected parameter then\n"
        "      the respective flag is ignored.\n"
        "  --wave_correct (no|horiz|vert)\n"
        "      Perform wave effect correction. The default is 'horiz'.\n"
        "  --save_graph <file_name>\n"
        "      Save matches graph represented in DOT language to <file_name> file.\n"
        "      Labels description: Nm is number of matches, Ni is number of inliers,\n"
        "      C is confidence.\n"
        "\nCompositing Flags:\n"
        "  --warp (affine|plane|cylindrical|spherical|fisheye|stereographic|compressedPlaneA2B1|compressedPlaneA1.5B1|compressedPlanePortraitA2B1|compressedPlanePortraitA1.5B1|paniniA2B1|paniniA1.5B1|paniniPortraitA2B1|paniniPortraitA1.5B1|mercator|transverseMercator)\n"
        "      Warp surface type. The default is 'spherical'.\n"
        "  --seam_megapix <float>\n"
        "      Resolution for seam estimation step. The default is 0.1 Mpx.\n"
        "  --seam (no|voronoi|gc_color|gc_colorgrad)\n"
        "      Seam estimation method. The default is 'gc_color'.\n"
        "  --compose_megapix <float>\n"
        "      Resolution for compositing step. Use -1 for original resolution.\n"
        "      The default is -1.\n"
        "  --expos_comp (no|gain|gain_blocks|channels|channels_blocks)\n"
        "      Exposure compensation method. The default is 'gain_blocks'.\n"
        "  --expos_comp_nr_feeds <int>\n"
        "      Number of exposure compensation feed. The default is 1.\n"
        "  --expos_comp_nr_filtering <int>\n"
        "      Number of filtering iterations of the exposure compensation gains.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 2.\n"
        "  --expos_comp_block_size <int>\n"
        "      BLock size in pixels used by the exposure compensator.\n"
        "      Only used when using a block exposure compensation method.\n"
        "      The default is 32.\n"
        "  --blend (no|feather|multiband)\n"
        "      Blending method. The default is 'multiband'.\n"
        "  --blend_strength <float>\n"
        "      Blending strength from [0,100] range. The default is 5.\n"
        "  --output <result_img>\n"
        "      The default is 'result.jpg'.\n"
        "  --timelapse (as_is|crop) \n"
        "      Output warped images separately as frames of a time lapse movie, with 'fixed_' prepended to input file names.\n"
        "  --rangewidth <int>\n"
        "      uses range_width to limit number of images to match with.\n";
}


// Default command line args
vector<String> img_names;
bool preview = false;
bool try_cuda = false;
double work_megapix = 0.6;
//double work_megapix = -1.0;
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
#ifdef HAVE_OPENCV_XFEATURES2D
string features_type = "surf";
float match_conf = 0.65f;
#else
string features_type = "orb";
float match_conf = 0.4f;
#endif
string matcher_type = "homography";
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
int expos_comp_nr_feeds = 1;
int expos_comp_nr_filtering = 2;
int expos_comp_block_size = 32;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;


static int parseCmdArgs(int argc, char** argv)
{
    if (argc == 1)
    {
        printUsage(argv);
        return -1;
    }
    for (int i = 1; i < argc; ++i)
    {
        if (string(argv[i]) == "--help" || string(argv[i]) == "/?")
        {
            printUsage(argv);
            return -1;
        }
        else if (string(argv[i]) == "--preview")
        {
            preview = true;
        }
        else if (string(argv[i]) == "--try_cuda")
        {
            if (string(argv[i + 1]) == "no")
                try_cuda = false;
            else if (string(argv[i + 1]) == "yes")
                try_cuda = true;
            else
            {
                cout << "Bad --try_cuda flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--work_megapix")
        {
            work_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam_megapix")
        {
            seam_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--compose_megapix")
        {
            compose_megapix = atof(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--result")
        {
            result_name = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--features")
        {
            features_type = argv[i + 1];
            if (string(features_type) == "orb")
                match_conf = 0.3f;
            i++;
        }
        else if (string(argv[i]) == "--matcher")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
                matcher_type = argv[i + 1];
            else
            {
                cout << "Bad --matcher flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--estimator")
        {
            if (string(argv[i + 1]) == "homography" || string(argv[i + 1]) == "affine")
                estimator_type = argv[i + 1];
            else
            {
                cout << "Bad --estimator flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--match_conf")
        {
            match_conf = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--conf_thresh")
        {
            conf_thresh = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--ba")
        {
            ba_cost_func = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--ba_refine_mask")
        {
            ba_refine_mask = argv[i + 1];
            if (ba_refine_mask.size() != 5)
            {
                cout << "Incorrect refinement mask length.\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--wave_correct")
        {
            if (string(argv[i + 1]) == "no")
                do_wave_correct = false;
            else if (string(argv[i + 1]) == "horiz")
            {
                do_wave_correct = true;
                wave_correct = cv::detail::WAVE_CORRECT_HORIZ;
            }
            else if (string(argv[i + 1]) == "vert")
            {
                do_wave_correct = true;
                wave_correct = cv::detail::WAVE_CORRECT_VERT;
            }
            else
            {
                cout << "Bad --wave_correct flag value\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--save_graph")
        {
            save_graph = true;
            save_graph_to = argv[i + 1];
            i++;
        }
        else if (string(argv[i]) == "--warp")
        {
            warp_type = string(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp")
        {
            if (string(argv[i + 1]) == "no")
                expos_comp_type = ExposureCompensator::NO;
            else if (string(argv[i + 1]) == "gain")
                expos_comp_type = ExposureCompensator::GAIN;
            else if (string(argv[i + 1]) == "gain_blocks")
                expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
            else if (string(argv[i + 1]) == "channels")
                expos_comp_type = ExposureCompensator::CHANNELS;
            else if (string(argv[i + 1]) == "channels_blocks")
                expos_comp_type = ExposureCompensator::CHANNELS_BLOCKS;
            else
            {
                cout << "Bad exposure compensation method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_nr_feeds")
        {
            expos_comp_nr_feeds = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_nr_filtering")
        {
            expos_comp_nr_filtering = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--expos_comp_block_size")
        {
            expos_comp_block_size = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--seam")
        {
            if (string(argv[i + 1]) == "no" ||
                string(argv[i + 1]) == "voronoi" ||
                string(argv[i + 1]) == "gc_color" ||
                string(argv[i + 1]) == "gc_colorgrad" ||
                string(argv[i + 1]) == "dp_color" ||
                string(argv[i + 1]) == "dp_colorgrad")
                seam_find_type = argv[i + 1];
            else
            {
                cout << "Bad seam finding method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--blend")
        {
            if (string(argv[i + 1]) == "no")
                blend_type = Blender::NO;
            else if (string(argv[i + 1]) == "feather")
                blend_type = Blender::FEATHER;
            else if (string(argv[i + 1]) == "multiband")
                blend_type = Blender::MULTI_BAND;
            else
            {
                cout << "Bad blending method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--timelapse")
        {
            timelapse = true;

            if (string(argv[i + 1]) == "as_is")
                timelapse_type = Timelapser::AS_IS;
            else if (string(argv[i + 1]) == "crop")
                timelapse_type = Timelapser::CROP;
            else
            {
                cout << "Bad timelapse method\n";
                return -1;
            }
            i++;
        }
        else if (string(argv[i]) == "--rangewidth")
        {
            range_width = atoi(argv[i + 1]);
            i++;
        }
        else if (string(argv[i]) == "--blend_strength")
        {
            blend_strength = static_cast<float>(atof(argv[i + 1]));
            i++;
        }
        else if (string(argv[i]) == "--output")
        {
            result_name = argv[i + 1];
            i++;
        }
        else
            img_names.push_back(argv[i]);
    }
    if (preview)
    {
        compose_megapix = 0.6;
    }
    return 0;
}


int main(int argc, char* argv[])
{
#if ENABLE_LOG
    int64 app_start_time = getTickCount();
#endif

#if 0
    cv::setBreakOnError(true);
#endif

    int retval = parseCmdArgs(argc, argv);
    if (retval)
        return retval;

    // Check if have enough images
    int num_images = static_cast<int>(img_names.size());
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    double work_scale = 1, seam_scale = 1, compose_scale = 1;
    bool is_work_scale_set = false, is_seam_scale_set = false, is_compose_scale_set = false;

    LOGLN("Finding features...");
#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    Ptr<Feature2D> finder;
    if (features_type == "orb")
    {
        finder = ORB::create();
    }
    else if (features_type == "akaze")
    {
        finder = AKAZE::create();
    }
#ifdef HAVE_OPENCV_XFEATURES2D
    else if (features_type == "surf")
    {
        finder = xfeatures2d::SURF::create();
    }
#endif
    else if (features_type == "sift")
    {
        finder = SIFT::create();
    }
    else
    {
        cout << "Unknown 2D features type: '" << features_type << "'.\n";
        return -1;
    }

    Mat full_img, img;
    vector<ImageFeatures> features(num_images);
    vector<Ort::Value> keypoints_onnx;
    vector<Ort::Value> descriptors_onnx;
    vector<Mat> images(num_images);
    vector<Size> full_img_sizes(num_images);
    double seam_work_aspect = 1;

    
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    auto providers = Ort::GetAvailableProviders();
    for (auto& p : providers) std::cout << "Available: " << p << std::endl;
    Ort::SessionOptions session_options;
    OrtCUDAProviderOptions cuda_options;
    cuda_options.device_id = 0;
    //auto& api = Ort::GetApi();
    //
    //// Создаем структуру через API (это предотвратит segfault из-за разницы версий)
    //api.CreateCUDAProviderOptions(&cuda_options);
    //const char* keys[] = {"device_id"};
    //const char* values[] = {"0"};
    //api.UpdateCUDAProviderOptions(cuda_options, keys, values, 1);
    //session_options.AppendExecutionProvider_CUDA_V2(*cuda_options);
    //session_options.AppendExecutionProvider_CUDA(cuda_options);

    try {
        std::cout << "HERE";
        session_options.AppendExecutionProvider_CUDA(cuda_options);
        std::cout << "HERE";
    } catch (const Ort::Exception& e) {
        std::cerr << "CUDA не поддерживается: " << e.what() << ". Используем CPU." << std::endl;
    }
    //std::cout << "HERE";
    Ort::Session session(env, "/home/ismukhin/projects/3dscanner/samples/panorama_stitching_samples/cpp/superpoint.onnx", session_options);
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    //Ort::MemoryInfo memory_info = Ort::MemoryInfo(
    //                                                "CUDA",                // Имя провайдера
    //                                                OrtAllocatorType::OrtArenaAllocator, 
    //                                                0,                     // ID устройства (обычно 0)
    //                                                OrtMemType::OrtMemTypeDefault
    //                                                );
    for (int i = 0; i < num_images; ++i)
    {
        full_img = imread(samples::findFile(img_names[i]));
        full_img_sizes[i] = full_img.size();
        if (full_img.empty())
        {
            LOGLN("Can't open image " << img_names[i]);
            return -1;
        }
        if (work_megapix < 0)
        {
            img = full_img;
            work_scale = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
        }
        if (!is_seam_scale_set)
        {
            seam_scale = min(1.0, sqrt(seam_megapix * 1e6 / full_img.size().area()));
            seam_work_aspect = seam_scale / work_scale;
            is_seam_scale_set = true;
        }
        cv::Mat img_float, img_gr;

        cv::cvtColor(img, img_gr, cv::COLOR_BGR2GRAY);
        img_gr.convertTo(img_float, CV_32FC1, 1.0/255.0);
        const int H = img_float.rows;
        const int W = img_float.cols;
        const int h = img.rows;
        const int w = img.cols;
        std::cout << H << " " << h << " " << W << " " << w << std::endl;
        std::vector<int64_t> input_shape = {1, 1, H, W};
        std::vector<float> input_data(1 * 1 * H * W);
        std::memcpy(input_data.data(), img_float.ptr<float>(), input_data.size() * sizeof(float));

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

        const char* input_names[] = {"input"};
        const char* output_names[] = {"keypoints", "scores", "descriptors"};
        std::vector<Ort::Value> ort_inputs;
        ort_inputs.push_back(std::move(input_tensor));

        auto outputs = session.Run(Ort::RunOptions{nullptr}, input_names, ort_inputs.data(), 1,
                                   output_names, 3);
        

        const float* keypoints_ptr = outputs[0].GetTensorData<float>();
        const float* scores_ptr    = outputs[1].GetTensorData<float>();
        const float* descriptors_ptr = outputs[2].GetTensorData<float>();
        size_t count = outputs[1].GetTensorTypeAndShapeInfo().GetElementCount();
        auto keypoints_shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t N = keypoints_shape[1];

        cv::Mat keypoints_mat(static_cast<int>(N), 2, CV_32FC1, const_cast<float*>(keypoints_ptr));
        cv::Mat descriptors_mat(static_cast<int>(N), 256, CV_32FC1, const_cast<float*>(descriptors_ptr));
        cv::Mat scores_mat(static_cast<int>(N), 1, CV_32FC1, const_cast<float*>(scores_ptr));

        // int target_height = img.size[1];   // из processor.size['height']
        // int target_width = img.size[0];    // из processor.size['width']
        // int scaleX = img.size[1] / target_height;
        // int scaleY = img.size[0] / target_width;
        // std::vector<float> mean = {0.485f, 0.456f, 0.406f};
        // std::vector<float> std = {0.229f, 0.224f, 0.225f};

        // torch::Tensor img_tensor = preprocess_image(img, target_height, target_width, mean, std);

        // auto output = module.forward({img_tensor});
        // auto dict = output.toGenericDict();

        // torch::Tensor keypoints = dict.at("keypoints").toTensor();
        // torch::Tensor scores = dict.at("scores").toTensor();
        // torch::Tensor descriptors = dict.at("descriptors").toTensor();

        // at::Tensor keypoints_tensor = keypoints.detach().cpu().squeeze().contiguous();
        // cv::Mat keypoint_mat(
        //     keypoints_tensor.size(0), 
        //     keypoints_tensor.size(1), 
        //     CV_32FC1, 
        //     keypoints_tensor.data_ptr<float>()
        // );

        // at::Tensor descriptors_tensor = descriptors.detach().cpu().squeeze().contiguous();
        // cv::Mat descriptors_mat(
        //     descriptors_tensor.size(0), 
        //     descriptors_tensor.size(1), 
        //     CV_32FC1, 
        //     descriptors_tensor.data_ptr<float>()
        // );

        // at::Tensor scores_tensor = scores.detach().cpu().squeeze().contiguous();
        // std::vector<float> scores_vec(scores_tensor.numel());
        // std::memcpy(scores_vec.data(), scores_tensor.data_ptr<float>(), sizeof(float) * scores_tensor.numel());
        std::vector<float> scores_vec(scores_ptr, scores_ptr + count);
        std::vector<cv::KeyPoint> keypoints_vec;
        keypoints_vec.reserve(keypoints_mat.rows);
        for (int i = 0; i < keypoints_mat.rows; ++i) {
            float x = keypoints_mat.at<float>(i, 0);
            float y = keypoints_mat.at<float>(i, 1);
            keypoints_vec.emplace_back(KeyPoint(x * (W), y * (H), 1.0f, -1.0f, scores_vec[i]));              
        }
        features[i].keypoints = keypoints_vec;
        cv::UMat umat;
        descriptors_mat.copyTo(umat);
        features[i].descriptors = umat.clone();
        std::cout << "Keypoints shape: " << keypoints_mat.size() << std::endl;
        std::cout << "Descriptors shape: " << descriptors_mat.size() << std::endl;
        std::cout << "Scores shape: " << scores_vec.size() << std::endl;
        //return 0;
        //computeImageFeatures(finder, img, features[i]);
        features[i].img_size = img.size();
        features[i].img_idx = i;
        LOGLN("Features in image #" << i+1 << ": " << features[i].keypoints.size());

        resize(full_img, img, Size(), seam_scale, seam_scale, INTER_LINEAR_EXACT);
        images[i] = img.clone();
        keypoints_onnx.push_back(std::move(outputs[0]));
        descriptors_onnx.push_back(std::move(outputs[2]));
    }

    full_img.release();
    img.release();

    LOGLN("Finding features, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOG_cv("Pairwise matching");
#if ENABLE_LOG
    t = getTickCount();
#endif
    vector<MatchesInfo> pairwise_matches(num_images * num_images);
    Ptr<FeaturesMatcher> matcher;
    if (matcher_type == "affine")
        matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
    else if (range_width==-1)
        matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
    else
        matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);

    Ort::Session lightglue(env, "/home/ismukhin/projects/3dscanner/samples/panorama_stitching_samples/cpp/superpoint_lightglue.onnx", session_options);

    const char* input_names[] = {"kpts0", "kpts1", "desc0", "desc1"};
    const char* output_names[] = {"matches0", "matches1", "mscores0", "mscores1"};
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    for (int i = 0; i < num_images; ++i) {
        MatchesInfo diag;
        diag.src_img_idx = -1;
        diag.dst_img_idx = -1;
        diag.confidence = 0;
        //diag.H = cv::Mat::eye(3, 3, CV_64F);
        pairwise_matches[i * num_images + i] = diag;
    }
    for (int i = 0; i < keypoints_onnx.size(); ++i) {
        for (int j = i + 1; j < keypoints_onnx.size(); ++j) {
            
            Ort::Value inputs[] = {
                Ort::Value::CreateTensor<float>(mem_info, keypoints_onnx[i].GetTensorMutableData<float>(), 
                    keypoints_onnx[i].GetTensorTypeAndShapeInfo().GetElementCount(), 
                    keypoints_onnx[i].GetTensorTypeAndShapeInfo().GetShape().data(), 
                    keypoints_onnx[i].GetTensorTypeAndShapeInfo().GetShape().size()),

                Ort::Value::CreateTensor<float>(mem_info, keypoints_onnx[j].GetTensorMutableData<float>(), 
                    keypoints_onnx[j].GetTensorTypeAndShapeInfo().GetElementCount(), 
                    keypoints_onnx[j].GetTensorTypeAndShapeInfo().GetShape().data(), 
                    keypoints_onnx[j].GetTensorTypeAndShapeInfo().GetShape().size()),

                Ort::Value::CreateTensor<float>(mem_info, descriptors_onnx[i].GetTensorMutableData<float>(), 
                    descriptors_onnx[i].GetTensorTypeAndShapeInfo().GetElementCount(), 
                    descriptors_onnx[i].GetTensorTypeAndShapeInfo().GetShape().data(), 
                    descriptors_onnx[i].GetTensorTypeAndShapeInfo().GetShape().size()),

                Ort::Value::CreateTensor<float>(mem_info, descriptors_onnx[j].GetTensorMutableData<float>(), 
                    descriptors_onnx[j].GetTensorTypeAndShapeInfo().GetElementCount(), 
                    descriptors_onnx[j].GetTensorTypeAndShapeInfo().GetShape().data(), 
                    descriptors_onnx[j].GetTensorTypeAndShapeInfo().GetShape().size())
            };
            auto output_values = lightglue.Run(Ort::RunOptions{nullptr},
                                               input_names, inputs, 4, output_names, 4);

            auto* matches0 = output_values[0].GetTensorMutableData<int64_t>();
            auto* matches1 = output_values[1].GetTensorMutableData<int64_t>();
            auto* mscores0 = output_values[2].GetTensorMutableData<float>();
            auto* mscores1 = output_values[3].GetTensorMutableData<float>();
            size_t num_matches1 = output_values[0].GetTensorTypeAndShapeInfo().GetElementCount();
            size_t num_matches2 = output_values[1].GetTensorTypeAndShapeInfo().GetElementCount();
            //std::cout << "Pair (" << i << "," << j << ") got " << num_matches1 << " matches and " << num_matches2 << "matches\n";
            //std::cout << "Keypoints size: img" << i << " = " << features[i].keypoints.size() 
            //          << ", img" << j << " = " << features[j].keypoints.size() << "\n";
//
            //for (size_t k = 0; k < std::min(num_matches1, (size_t)10); ++k) {
            //    std::cout << " match " << k << ": " << matches0[k] << " -> " << matches1[k] << " score1 = " << mscores0[k] << " score2 = " << mscores1[k] << "\n";
            //}
            MatchesInfo info;
            info.src_img_idx = i;
            info.dst_img_idx = j;
            double total_conf = 0.0;
            for (int l = 0; l < num_matches1; l++) {
                int t = static_cast<int>(matches0[l]);
                if (t >= 0 && t < (int)features[j].keypoints.size()) {
                    float prob = mscores0[l];
                    if (prob >= 0.5f)
                        info.matches.push_back(cv::DMatch(static_cast<int>(l), t, 1.0f - prob));
                }
            }
            std::cout << info.matches.size() << std::endl;
            if (!info.matches.empty()) {
                std::vector<cv::Point2f> pts1, pts2;
                for (const auto& m : info.matches) {
                    //std::cout << "keypoints[i] size" << features[i].keypoints.size() << "keypoints[j] size" << features[j].keypoints.size() << "query = " << m.queryIdx << "train = " << m.trainIdx << std::endl;
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
            pairwise_matches[i * num_images + j] = info;

            MatchesInfo info1;
            info1.src_img_idx = j;
            info1.dst_img_idx = i;
            for (int l = 0; l < num_matches2; l++) {
                int t = static_cast<int>(matches1[l]);
                if (t >= 0 && t < (int)features[i].keypoints.size()) {
                    float prob = mscores1[l];
                    if (prob >= 0.5f)
                        info1.matches.push_back(cv::DMatch(static_cast<int>(l), t, 1.0f - prob));
                }
            }

            if (!info1.matches.empty()) {
                std::vector<cv::Point2f> pts1, pts2;
                for (const auto& m : info1.matches) {
                    //std::cout << "keypoints[i] size" << features[i].keypoints.size() << "keypoints[j] size" << features[j].keypoints.size() << "query = " << m.queryIdx << "train = " << m.trainIdx << std::endl;
                    pts1.push_back(features[j].keypoints[m.queryIdx].pt);
                    pts2.push_back(features[i].keypoints[m.trainIdx].pt);
                }
                if (pts1.size() >= 4) {
                    info1.H = cv::findHomography(pts1, pts2, cv::RANSAC, 3.0, info1.inliers_mask);
                    int inliers = std::count(info1.inliers_mask.begin(), info1.inliers_mask.end(), 1);
                    info1.num_inliers = inliers;
                    info1.confidence = info1.num_inliers / (8 + 0.3 * info1.matches.size());
                } else {
                    info1.H = cv::Mat::eye(3, 3, CV_64F);
                    info1.num_inliers = 0;
                    info1.confidence = 0;
                }
            } else{
                    info1.H = cv::Mat::eye(3, 3, CV_64F);
                    info1.num_inliers = 0;
                    info1.confidence = 0;      
            }

            pairwise_matches[j * num_images + i] = info1;
        }
    }
    //return 0;
    vector<MatchesInfo> pairwise_matches1(num_images * num_images);
    (*matcher)(features, pairwise_matches1);
    matcher->collectGarbage();

    LOGLN("Pairwise matching, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");
    std::cout << pairwise_matches.size() << std::endl;
    // Check if we should save matches graph
    if (save_graph)
    {
        LOGLN("Saving matches graph...");
        ofstream f(save_graph_to.c_str());
        f << matchesGraphAsString(img_names, pairwise_matches, conf_thresh);
    }

    // Leave only images we are sure are from the same panorama
    std::cout << features.size() << std::endl;
    vector<int> indices = my::leaveBiggestComponent(features, pairwise_matches, conf_thresh);
    vector<Mat> img_subset;
    vector<String> img_names_subset;
    vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices.size(); ++i)
    {
        img_names_subset.push_back(img_names[indices[i]]);
        img_subset.push_back(images[indices[i]]);
        full_img_sizes_subset.push_back(full_img_sizes[indices[i]]);
    }

    images = img_subset;
    img_names = img_names_subset;
    full_img_sizes = full_img_sizes_subset;
    for (const auto& elem : img_names) std::cout << elem << std::endl;
    std::cout << "LIGHTGLUE\n\n\n\n";
    for (const auto& elem : pairwise_matches) {
        std::cout << "src_idx" << elem.src_img_idx << " " << "dst_idx" << elem.dst_img_idx << std::endl;;
        std::cout << "H" << elem.H << std::endl;
    }
    std::cout << "\n\n\n\n";

    std::cout << "OPENCV\n\n\n\n";
    for (const auto& elem : pairwise_matches1) {
        std::cout << "src_idx" << elem.src_img_idx << " " << "dst_idx" << elem.dst_img_idx << std::endl;;
        std::cout << "H" << elem.H << std::endl;
    }
    std::cout << "\n\n\n\n";

    //return 0;
    // Check if we still have enough images
    num_images = static_cast<int>(img_names.size());
    std::cout << "Num images " << num_images << std::endl;
    if (num_images < 2)
    {
        LOGLN("Need more images");
        return -1;
    }

    Ptr<Estimator> estimator;
    if (estimator_type == "affine")
        estimator = makePtr<AffineBasedEstimator>();
    else
        estimator = makePtr<HomographyBasedEstimator>();

    vector<CameraParams> cameras;
    if (!(*estimator)(features, pairwise_matches, cameras))
    {
        cout << "Homography estimation failed.\n";
        return -1;
    }

    for (size_t i = 0; i < cameras.size(); ++i)
    {
        Mat R;
        cameras[i].R.convertTo(R, CV_32F);
        cameras[i].R = R;
        LOGLN("Initial camera intrinsics #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
    }

    Ptr<cv::detail::BundleAdjusterBase> adjuster;
    if (ba_cost_func == "reproj") adjuster = makePtr<cv::detail::BundleAdjusterReproj>();
    else if (ba_cost_func == "ray") adjuster = makePtr<cv::detail::BundleAdjusterRay>();
    else if (ba_cost_func == "affine") adjuster = makePtr<cv::detail::BundleAdjusterAffinePartial>();
    else if (ba_cost_func == "no") adjuster = makePtr<NoBundleAdjuster>();
    else
    {
        cout << "Unknown bundle adjustment cost function: '" << ba_cost_func << "'.\n";
        return -1;
    }
    adjuster->setConfThresh(conf_thresh);
    Mat_<uchar> refine_mask = Mat::zeros(3, 3, CV_8U);
    if (ba_refine_mask[0] == 'x') refine_mask(0,0) = 1;
    if (ba_refine_mask[1] == 'x') refine_mask(0,1) = 1;
    if (ba_refine_mask[2] == 'x') refine_mask(0,2) = 1;
    if (ba_refine_mask[3] == 'x') refine_mask(1,1) = 1;
    if (ba_refine_mask[4] == 'x') refine_mask(1,2) = 1;
    adjuster->setRefinementMask(refine_mask);
    if (!(*adjuster)(features, pairwise_matches, cameras))
    {
        cout << "Camera parameters adjusting failed.\n";
        return -1;
    }

    // Find median focal length

    vector<double> focals;
    for (size_t i = 0; i < cameras.size(); ++i)
    {
        LOGLN("Camera #" << indices[i]+1 << ":\nK:\n" << cameras[i].K() << "\nR:\n" << cameras[i].R);
        focals.push_back(cameras[i].focal);
    }

    sort(focals.begin(), focals.end());
    float warped_image_scale;
    if (focals.size() % 2 == 1)
        warped_image_scale = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct)
    {
        vector<Mat> rmats;
        for (size_t i = 0; i < cameras.size(); ++i)
            rmats.push_back(cameras[i].R.clone());
        waveCorrect(rmats, wave_correct);
        for (size_t i = 0; i < cameras.size(); ++i)
            cameras[i].R = rmats[i];
    }

    LOGLN("Warping images (auxiliary)... ");
#if ENABLE_LOG
    t = getTickCount();
#endif

    vector<Point> corners(num_images);
    vector<UMat> masks_warped(num_images);
    vector<UMat> images_warped(num_images);
    vector<Size> sizes(num_images);
    vector<UMat> masks(num_images);

    // Prepare images masks
    for (int i = 0; i < num_images; ++i)
    {
        masks[i].create(images[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    // Warp images and their masks

    Ptr<WarperCreator> warper_creator;
#ifdef HAVE_OPENCV_CUDAWARPING
    if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarperGpu>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarperGpu>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarperGpu>();
    }
    else
#endif
    {
        if (warp_type == "plane")
            warper_creator = makePtr<cv::PlaneWarper>();
        else if (warp_type == "affine")
            warper_creator = makePtr<cv::AffineWarper>();
        else if (warp_type == "cylindrical")
            warper_creator = makePtr<cv::CylindricalWarper>();
        else if (warp_type == "spherical")
            warper_creator = makePtr<cv::SphericalWarper>();
        else if (warp_type == "fisheye")
            warper_creator = makePtr<cv::FisheyeWarper>();
        else if (warp_type == "stereographic")
            warper_creator = makePtr<cv::StereographicWarper>();
        else if (warp_type == "compressedPlaneA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlaneA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearWarper>(1.5f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA2B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "compressedPlanePortraitA1.5B1")
            warper_creator = makePtr<cv::CompressedRectilinearPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniA2B1")
            warper_creator = makePtr<cv::PaniniWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniA1.5B1")
            warper_creator = makePtr<cv::PaniniWarper>(1.5f, 1.0f);
        else if (warp_type == "paniniPortraitA2B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(2.0f, 1.0f);
        else if (warp_type == "paniniPortraitA1.5B1")
            warper_creator = makePtr<cv::PaniniPortraitWarper>(1.5f, 1.0f);
        else if (warp_type == "mercator")
            warper_creator = makePtr<cv::MercatorWarper>();
        else if (warp_type == "transverseMercator")
            warper_creator = makePtr<cv::TransverseMercatorWarper>();
    }

    if (!warper_creator)
    {
        cout << "Can't create the following warper '" << warp_type << "'\n";
        return 1;
    }

    Ptr<RotationWarper> warper = warper_creator->create(static_cast<float>(warped_image_scale * seam_work_aspect));

    for (int i = 0; i < num_images; ++i)
    {
        Mat_<float> K;
        cameras[i].K().convertTo(K, CV_32F);
        float swa = (float)seam_work_aspect;
        K(0,0) *= swa; K(0,2) *= swa;
        K(1,1) *= swa; K(1,2) *= swa;

        corners[i] = warper->warp(images[i], K, cameras[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        warper->warp(masks[i], K, cameras[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }

    vector<UMat> images_warped_f(num_images);
    for (int i = 0; i < num_images; ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);

    LOGLN("Warping images, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Compensating exposure...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<ExposureCompensator> compensator = ExposureCompensator::createDefault(expos_comp_type);
    if (dynamic_cast<GainCompensator*>(compensator.get()))
    {
        GainCompensator* gcompensator = dynamic_cast<GainCompensator*>(compensator.get());
        gcompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<ChannelsCompensator*>(compensator.get()))
    {
        ChannelsCompensator* ccompensator = dynamic_cast<ChannelsCompensator*>(compensator.get());
        ccompensator->setNrFeeds(expos_comp_nr_feeds);
    }

    if (dynamic_cast<BlocksCompensator*>(compensator.get()))
    {
        BlocksCompensator* bcompensator = dynamic_cast<BlocksCompensator*>(compensator.get());
        bcompensator->setNrFeeds(expos_comp_nr_feeds);
        bcompensator->setNrGainsFilteringIterations(expos_comp_nr_filtering);
        bcompensator->setBlockSize(expos_comp_block_size, expos_comp_block_size);
    }

    compensator->feed(corners, images_warped, masks_warped);

    LOGLN("Compensating exposure, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    LOGLN("Finding seams...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Ptr<SeamFinder> seam_finder;
    if (seam_find_type == "no")
        seam_finder = makePtr<cv::detail::NoSeamFinder>();
    else if (seam_find_type == "voronoi")
        seam_finder = makePtr<cv::detail::VoronoiSeamFinder>();
    else if (seam_find_type == "gc_color")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<cv::detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR);
        else
#endif
            seam_finder = makePtr<cv::detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR);
    }
    else if (seam_find_type == "gc_colorgrad")
    {
#ifdef HAVE_OPENCV_CUDALEGACY
        if (try_cuda && cuda::getCudaEnabledDeviceCount() > 0)
            seam_finder = makePtr<cv::detail::GraphCutSeamFinderGpu>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
        else
#endif
            seam_finder = makePtr<cv::detail::GraphCutSeamFinder>(GraphCutSeamFinderBase::COST_COLOR_GRAD);
    }
    else if (seam_find_type == "dp_color")
        seam_finder = makePtr<cv::detail::DpSeamFinder>(DpSeamFinder::COLOR);
    else if (seam_find_type == "dp_colorgrad")
        seam_finder = makePtr<cv::detail::DpSeamFinder>(DpSeamFinder::COLOR_GRAD);
    if (!seam_finder)
    {
        cout << "Can't create the following seam finder '" << seam_find_type << "'\n";
        return 1;
    }

    seam_finder->find(images_warped_f, corners, masks_warped);

    LOGLN("Finding seams, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

    // Release unused memory
    images.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

    LOGLN("Compositing...");
#if ENABLE_LOG
    t = getTickCount();
#endif

    Mat img_warped, img_warped_s;
    Mat dilated_mask, seam_mask, mask, mask_warped;
    Ptr<Blender> blender;
    Ptr<Timelapser> timelapser;
    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;

    for (int img_idx = 0; img_idx < num_images; ++img_idx)
    {
        LOGLN("Compositing image #" << indices[img_idx]+1);

        // Read image and resize it if necessary
        full_img = imread(samples::findFile(img_names[img_idx]));
        if (!is_compose_scale_set)
        {
            if (compose_megapix > 0)
                compose_scale = min(1.0, sqrt(compose_megapix * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale;
            compose_work_aspect = compose_scale / work_scale;

            // Update warped image scale
            warped_image_scale *= static_cast<float>(compose_work_aspect);
            warper = warper_creator->create(warped_image_scale);

            // Update corners and sizes
            for (int i = 0; i < num_images; ++i)
            {
                // Update intrinsics
                cameras[i].focal *= compose_work_aspect;
                cameras[i].ppx *= compose_work_aspect;
                cameras[i].ppy *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes[i].height * compose_scale);
                }

                Mat K;
                cameras[i].K().convertTo(K, CV_32F);
                Rect roi = warper->warpRoi(sz, K, cameras[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (abs(compose_scale - 1) > 1e-1)
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras[img_idx].K().convertTo(K, CV_32F);

        // Warp the current image
        warper->warp(img, K, cameras[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        warper->warp(mask, K, cameras[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);

        // Compensate exposure
        compensator->apply(img_idx, corners[img_idx], img_warped, mask_warped);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);
        mask_warped = seam_mask & mask_warped;

        if (!blender && !timelapse)
        {
            blender = Blender::createDefault(blend_type, try_cuda);
            Size dst_sz = resultRoi(corners, sizes).size();
            float blend_width = sqrt(static_cast<float>(dst_sz.area())) * blend_strength / 100.f;
            if (blend_width < 1.f)
                blender = Blender::createDefault(Blender::NO, try_cuda);
            else if (blend_type == Blender::MULTI_BAND)
            {
                MultiBandBlender* mb = dynamic_cast<MultiBandBlender*>(blender.get());
                mb->setNumBands(static_cast<int>(ceil(log(blend_width)/log(2.)) - 1.));
                LOGLN("Multi-band blender, number of bands: " << mb->numBands());
            }
            else if (blend_type == Blender::FEATHER)
            {
                FeatherBlender* fb = dynamic_cast<FeatherBlender*>(blender.get());
                fb->setSharpness(1.f/blend_width);
                LOGLN("Feather blender, sharpness: " << fb->sharpness());
            }
            blender->prepare(corners, sizes);
        }
        else if (!timelapser && timelapse)
        {
            timelapser = Timelapser::createDefault(timelapse_type);
            timelapser->initialize(corners, sizes);
        }

        // Blend the current image
        if (timelapse)
        {
            timelapser->process(img_warped_s, Mat::ones(img_warped_s.size(), CV_8UC1), corners[img_idx]);
            String fixedFileName;
            size_t pos_s = String(img_names[img_idx]).find_last_of("/\\");
            if (pos_s == String::npos)
            {
                fixedFileName = "fixed_" + img_names[img_idx];
            }
            else
            {
                fixedFileName = "fixed_" + String(img_names[img_idx]).substr(pos_s + 1, String(img_names[img_idx]).length() - pos_s);
            }
            imwrite(fixedFileName, timelapser->getDst());
        }
        else
        {
            blender->feed(img_warped_s, mask_warped, corners[img_idx]);
        }
    }

    if (!timelapse)
    {
        Mat result, result_mask;
        blender->blend(result, result_mask);

        LOGLN("Compositing, time: " << ((getTickCount() - t) / getTickFrequency()) << " sec");

        imwrite(result_name, result);
    }

    LOGLN("Finished, total time: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec");
    return 0;
}
