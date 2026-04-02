/*
 * GEOMETRIC GAZE ESTIMATION - C++ VERSION (Phase 1: YOLOv8 Face Detection)
 * Using ONNX Runtime C++ API for faster GPU-accelerated inference.
 */
#define NOMINMAX
#include <iostream>
#include <vector>
#include <map>
#include <chrono>
#include <string>
#include <filesystem>
#include <algorithm>
#include <fstream>
#include <cmath>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <onnxruntime_cxx_api.h>

namespace fs = std::filesystem;

// ================================================================== //
//  CONFIG & UTILS                                                     //
// ================================================================== //
struct Config {
    float conf_threshold = 0.5f;
    float nms_threshold = 0.45f;
    cv::Size model_size = cv::Size(640, 640);
} cfg;

struct BBox {
    cv::Rect box;
    float confidence;
};

struct FaceResult {
    BBox bbox;
    std::vector<cv::Point3f> landmarks; // 478 points
};

// Simple Letterbox for YOLOv8
cv::Mat letterbox(const cv::Mat& src, cv::Size target_size, float& scale, int& top, int& left) {
    int sw = src.cols, sh = src.rows;
    int tw = target_size.width, th = target_size.height;
    scale = std::min((float)tw / sw, (float)th / sh);
    int nw = (int)(sw * scale), nh = (int)(sh * scale);
    
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    
    top = (th - nh) / 2;
    left = (tw - nw) / 2;
    cv::Mat res = cv::Mat::zeros(target_size, CV_8UC3);
    resized.copyTo(res(cv::Rect(left, top, nw, nh)));
    return res;
}

// ================================================================== //
//  YOLOV8 DETECTOR (ONNX RUNTIME)                                    //
// ================================================================== //
class YOLOv8Detector {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::RunOptions run_options{nullptr};
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
    
public:
    YOLOv8Detector(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8") {
        Ort::SessionOptions session_options;
        
        // Default to CPU for now (DirectML requires specific build features)
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session = Ort::Session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input Info
        auto input_node = session.GetInputNameAllocated(0, allocator);
        input_names.push_back(_strdup(input_node.get()));
        input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        
        // Output Info
        auto output_node = session.GetOutputNameAllocated(0, allocator);
        output_names.push_back(_strdup(output_node.get()));
    }

    std::vector<BBox> detect(const cv::Mat& frame) {
        float scale; int top, left;
        cv::Mat blob_img = letterbox(frame, cfg.model_size, scale, top, left);
        
        // Normalization [0,1] & RGB
        cv::Mat rgb;
        cv::cvtColor(blob_img, rgb, cv::COLOR_BGR2RGB);
        rgb.convertTo(rgb, CV_32FC3, 1.0 / 255.0);
        
        // HWC to CHW
        std::vector<float> input_tensor_values(640 * 640 * 3);
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        for (int i = 0; i < 3; ++i) {
            std::memcpy(input_tensor_values.data() + i * 640 * 640, channels[i].data, 640 * 640 * sizeof(float));
        }
        
        // Run Inference
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
        
        auto output_tensors = session.Run(run_options, input_names.data(), &input_tensor, 1, output_names.data(), 1);
        float* output_data = output_tensors[0].GetTensorMutableData<float>();
        auto output_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape(); // [1, 5, 8400]
        
        // [1, 5, 8400] -> Transpose to [8400, 5]
        int rows = (int)output_shape[2]; // 8400
        int dims = (int)output_shape[1]; // 5 (x,y,w,h,conf)
        
        std::vector<cv::Rect> boxes;
        std::vector<float> confs;
        
        for (int i = 0; i < rows; ++i) {
            float confidence = output_data[4 * rows + i];
            if (confidence > cfg.conf_threshold) {
                float cx = output_data[0 * rows + i];
                float cy = output_data[1 * rows + i];
                float w  = output_data[2 * rows + i];
                float h  = output_data[3 * rows + i];
                
                int x = (int)((cx - w / 2 - left) / scale);
                int y = (int)((cy - h / 2 - top) / scale);
                int width = (int)(w / scale);
                int height = (int)(h / scale);
                
                boxes.push_back(cv::Rect(x, y, width, height));
                confs.push_back(confidence);
            }
        }
        
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confs, cfg.conf_threshold, cfg.nms_threshold, indices);
        
        std::vector<BBox> filtered;
        for (int idx : indices) {
            filtered.push_back({boxes[idx], confs[idx]});
        }
        return filtered;
    }
};

// ================================================================== //
//  FACEMESH DETECTOR (ONNX RUNTIME)                                  //
// ================================================================== //
class FaceMeshDetector {
    Ort::Env env;
    Ort::Session session{nullptr};
    Ort::RunOptions run_options{nullptr};
    
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<int64_t> input_shape;
    cv::Size model_size = cv::Size(192, 192);

public:
    FaceMeshDetector(const std::string& model_path) : env(ORT_LOGGING_LEVEL_WARNING, "FaceMesh") {
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(4);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        
        session = Ort::Session(env, std::wstring(model_path.begin(), model_path.end()).c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_node = session.GetInputNameAllocated(0, allocator);
        input_names.push_back(_strdup(input_node.get()));
        input_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        
        // We expect multiple outputs (landmarks, scores, etc.)
        size_t num_outputs = session.GetOutputCount();
        for (size_t i = 0; i < num_outputs; i++) {
            auto out_node = session.GetOutputNameAllocated(i, allocator);
            output_names.push_back(_strdup(out_node.get()));
        }
    }

    std::vector<cv::Point3f> detect(const cv::Mat& frame, const cv::Rect& face_box) {
        // 1. Crop face with padding (25%)
        int pad_w = (int)(face_box.width * 0.25f);
        int pad_h = (int)(face_box.height * 0.25f);
        
        cv::Rect roi = face_box;
        roi.x = std::max(0, roi.x - pad_w);
        roi.y = std::max(0, roi.y - pad_h);
        roi.width = std::min(frame.cols - roi.x, roi.width + 2 * pad_w);
        roi.height = std::min(frame.rows - roi.y, roi.height + 2 * pad_h);
        
        cv::Mat face_img = frame(roi).clone();
        cv::resize(face_img, face_img, model_size);
        cv::cvtColor(face_img, face_img, cv::COLOR_BGR2RGB);
        face_img.convertTo(face_img, CV_32FC3, 1.0 / 255.0);

        // 2. Inference
        std::vector<float> input_tensor_values(192 * 192 * 3);
        std::vector<cv::Mat> channels(3);
        cv::split(face_img, channels);
        for (int i = 0; i < 3; ++i) {
            std::memcpy(input_tensor_values.data() + i * 192 * 192, channels[i].data, 192 * 192 * sizeof(float));
        }

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size());
        
        auto output_tensors = session.Run(run_options, input_names.data(), &input_tensor, 1, output_names.data(), output_names.size());
        
        // Landmarks are typically in the first output [1, 1434] (478pts * 3)
        float* landmark_data = output_tensors[0].GetTensorMutableData<float>();
        
        std::vector<cv::Point3f> landmarks;
        for (int i = 0; i < 478; ++i) {
            float px = landmark_data[i * 3];
            float py = landmark_data[i * 3 + 1];
            float pz = landmark_data[i * 3 + 2];
            
            // Map back to original image
            float rx = (px / 192.0f) * roi.width + roi.x;
            float ry = (py / 192.0f) * roi.height + roi.y;
            float rz = (pz / 192.0f) * roi.width; // Z scale roughly same as width
            
            landmarks.push_back(cv::Point3f(rx, ry, rz));
        }
        return landmarks;
    }
};

// ================================================================== //
//  GAZE GEOMETRY MATH (3D EYEBALL)                                   //
// ================================================================== //
class GazeMath {
public:
    static std::tuple<float, float, Eigen::Vector3f> calculate_gaze(const std::vector<cv::Point3f>& landmarks) {
        if (landmarks.size() < 478) return {0.0f, 0.0f, Eigen::Vector3f::Zero()};

        auto pt = [&](int idx) -> Eigen::Vector3f {
            return Eigen::Vector3f(landmarks[idx].x, landmarks[idx].y, landmarks[idx].z);
        };

        Eigen::Vector3f p168 = pt(168);
        Eigen::Vector3f p2   = pt(2);
        Eigen::Vector3f p331 = pt(331);
        Eigen::Vector3f p102 = pt(102);

        // 1. Face Basis (Hệ trục mặt phẳng khuôn mặt)
        Eigen::Vector3f U_ref = (p168 - p2).normalized();
        Eigen::Vector3f nf = (p331 - p168).cross(p102 - p168);
        Eigen::Vector3f Vf = -nf.normalized();
        if (Vf.z() > 0) Vf = -Vf;

        Eigen::Vector3f Rf = U_ref.cross(Vf).normalized();
        Eigen::Vector3f Uf = Vf.cross(Rf).normalized();

        // 2. Ước lượng tâm nhãn cầu
        auto get_eyeball_center = [&](const Eigen::Vector3f& P_top, const Eigen::Vector3f& P_bot, 
                                      const Eigen::Vector3f& P_in, const Eigen::Vector3f& P_out, 
                                      const Eigen::Vector3f& V_face) {
            Eigen::Vector3f O_surf = (P_top + P_bot + P_in + P_out) / 4.0f;
            float radius = (P_out - P_in).norm() * 0.4f;
            return O_surf - V_face * radius;
        };

        // Mắt trái
        Eigen::Vector3f eyeL_in = pt(163), eyeL_top = pt(157), eyeL_out = pt(161), eyeL_bot = pt(154), iris_L = pt(468);
        Eigen::Vector3f O_L = get_eyeball_center(eyeL_top, eyeL_bot, eyeL_in, eyeL_out, Vf);
        Eigen::Vector3f gaze_L = (iris_L - O_L).normalized();

        // Mắt phải
        Eigen::Vector3f eyeR_in = pt(390), eyeR_top = pt(384), eyeR_out = pt(388), eyeR_bot = pt(381), iris_R = pt(473);
        Eigen::Vector3f O_R = get_eyeball_center(eyeR_top, eyeR_bot, eyeR_in, eyeR_out, Vf);
        Eigen::Vector3f gaze_R = (iris_R - O_R).normalized();

        // 3. Tính toán Gaze Vector chót
        Eigen::Vector3f V_final = ((gaze_L + gaze_R) / 2.0f).normalized();

        // Tính Yaw và Pitch
        float pitch = std::asin(-V_final.y()) * 180.0f / 3.14159265358979323846f;
        float yaw = std::atan2(V_final.x(), -V_final.z()) * 180.0f / 3.14159265358979323846f;

        return {yaw, pitch, V_final};
    }
};

class EMASmoother {
    float alpha;
    float value;
    bool initialized;
public:
    EMASmoother(float a = 0.2f) : alpha(a), value(0), initialized(false) {}
    float update(float new_val) {
        if (!initialized) {
            value = new_val;
            initialized = true;
        } else {
            value = alpha * new_val + (1.0f - alpha) * value;
        }
        return value;
    }
    void reset() { initialized = false; }
};

// ================================================================== //
//  TEMPORAL SMOOTHING (KALMAN FILTER)                                //
// ================================================================== //
class LandmarkKalmanFilter {
    std::vector<cv::KalmanFilter> filters;
    bool initialized = false;
    
public:
    LandmarkKalmanFilter() {}
    
    void init(const std::vector<cv::Point3f>& initial_pts, float dt) {
        filters.clear();
        for (const auto& pt : initial_pts) {
            cv::KalmanFilter kf(6, 3, 0); // Trạng thái [x, y, z, vx, vy, vz], Đo lường [x, y, z]
            kf.transitionMatrix = (cv::Mat_<float>(6, 6) << 
                1, 0, 0, dt, 0, 0,
                0, 1, 0, 0, dt, 0,
                0, 0, 1, 0, 0, dt,
                0, 0, 0, 1, 0, 0,
                0, 0, 0, 0, 1, 0,
                0, 0, 0, 0, 0, 1);
            
            kf.measurementMatrix = cv::Mat::eye(3, 6, CV_32F);
            
            cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(0.1f));      // Noise từ vận tốc/gia tốc
            cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(0.5f));  // Noise từ output của model
            cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0f));
            
            kf.statePost.at<float>(0) = pt.x;
            kf.statePost.at<float>(1) = pt.y;
            kf.statePost.at<float>(2) = pt.z;
            kf.statePost.at<float>(3) = 0.0f;
            kf.statePost.at<float>(4) = 0.0f;
            kf.statePost.at<float>(5) = 0.0f;
            
            filters.push_back(kf);
        }
        initialized = true;
    }
    
    std::vector<cv::Point3f> update(const std::vector<cv::Point3f>& pts, float dt) {
        if (!initialized || filters.size() != pts.size()) {
            init(pts, dt);
            return pts;
        }
        
        std::vector<cv::Point3f> smoothed(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            // Cập nhật khoảng cách thời gian cho hệ phương trình động lực
            filters[i].transitionMatrix.at<float>(0, 3) = dt;
            filters[i].transitionMatrix.at<float>(1, 4) = dt;
            filters[i].transitionMatrix.at<float>(2, 5) = dt;
            
            filters[i].predict();
            
            cv::Mat meas = (cv::Mat_<float>(3, 1) << pts[i].x, pts[i].y, pts[i].z);
            cv::Mat estimated = filters[i].correct(meas);
            
            smoothed[i].x = estimated.at<float>(0);
            smoothed[i].y = estimated.at<float>(1);
            smoothed[i].z = estimated.at<float>(2);
        }
        return smoothed;
    }
    
    void reset() {
        initialized = false;
    }
};

// ================================================================== //
//  MAIN APP                                                           //
// ================================================================== //
int main() {
    std::string model_path = "models/yolov8_face.onnx";
    std::string mesh_path = "models/face_mesh_attention.onnx";
    if (!fs::exists(model_path)) {
        std::cerr << "[ERR] YOLO Model not found at: " << model_path << "\n";
        return -1;
    }
    
    YOLOv8Detector detector(model_path);
    std::unique_ptr<FaceMeshDetector> mesh_detector;
    
    if (fs::exists(mesh_path)) {
        mesh_detector = std::make_unique<FaceMeshDetector>(mesh_path);
        std::cout << "[INFO] FaceMesh (478 pts) Initialized.\n";
    }
    
    cv::VideoCapture cap(0, cv::CAP_DSHOW);
    if (!cap.isOpened()) {
        std::cerr << "[ERR] Cannot open camera.\n";
        return -1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    cv::Mat frame;
    auto last_time = std::chrono::steady_clock::now();
    
    std::cout << "\n--- Gaze Estimation C++ (Phase 4) Started ---\n";
    std::cout << "Press 'Q' to quit.\n";

    LandmarkKalmanFilter landmark_smoother;
    EMASmoother yaw_smoother(0.15f);
    EMASmoother pitch_smoother(0.15f);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;
        
        auto current_time = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(current_time - last_time).count();
        last_time = current_time;
        if (dt <= 0.001f) dt = 0.033f; // Fallback an toàn (tránh divide zero)
        
        double fps = 1.0 / dt;

        auto results = detector.detect(frame);
        
        // Cập nhật reset filter nếu không có mặt trong khung hình
        if (results.empty()) {
            landmark_smoother.reset();
            yaw_smoother.reset();
            pitch_smoother.reset();
        }        
        for (const auto& r : results) {
            cv::rectangle(frame, r.box, cv::Scalar(0, 255, 0), 1);
            
            // Run Face Mesh if face found
            if (mesh_detector) {
                auto raw_landmarks = mesh_detector->detect(frame, r.box);
                
                // Trơn hóa bằng Kalman Filter
                auto landmarks = landmark_smoother.update(raw_landmarks, dt);
                
                // Toán học Gaze
                auto [yaw_raw, pitch_raw, v_gaze] = GazeMath::calculate_gaze(landmarks);
                float yaw = yaw_smoother.update(yaw_raw);
                float pitch = pitch_smoother.update(pitch_raw);

                // Print Gaze Info
                std::string gaze_lbl = "Yaw: " + std::to_string((int)yaw) + "  Pitch: " + std::to_string((int)pitch);
                cv::putText(frame, gaze_lbl, {r.box.x, r.box.y - 22}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 255}, 2);

                // Vẽ trục 3D mũi (Đơn giản)
                if (landmarks.size() >= 478) {
                    cv::Point2f nose(landmarks[2].x, landmarks[2].y);
                    cv::Point2f gaze_endpoint(nose.x + v_gaze.x() * 100.f, nose.y - v_gaze.y() * 100.f);
                    cv::arrowedLine(frame, nose, gaze_endpoint, cv::Scalar(0, 0, 255), 2);
                }

                // Vẽ Iris 
                if (landmarks.size() >= 478) {
                    for (int i = 468; i < 478; ++i) {
                        cv::circle(frame, cv::Point((int)landmarks[i].x, (int)landmarks[i].y), 1, {0, 0, 255}, -1);
                    }
                }
            }

            std::string label = "Face: " + std::to_string((int)(r.confidence * 100)) + "%";
            cv::putText(frame, label, {r.box.x, r.box.y - 5}, cv::FONT_HERSHEY_SIMPLEX, 0.4, {0, 255, 0}, 1);
        }
        
        cv::putText(frame, "FPS: " + std::to_string((int)fps), {20, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 0, 255}, 2);
        
        cv::imshow("Gaze Estimation - Phase 4 (Gaze Geometry)", frame);
        if (cv::waitKey(1) == 'q') break;
    }
    
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
