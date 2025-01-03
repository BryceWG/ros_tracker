#include "../include/kcftracker.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

// 构造函数
KCFTracker::KCFTracker(bool hog, bool fixed_window, bool multiscale, bool lab)
{
    // 初始化配置参数
    HOG = hog;
    FIXED_WINDOW = fixed_window;
    MULTISCALE = multiscale;
    LAB = lab;
    
    // 初始化跟踪状态
    scale = 1.0f;
    
    // 修改默认参数以提高稳定性
    lambda = 0.0001f;
    padding = 2.0f;  // 减小padding
    output_sigma_factor = 0.1f;
    scale_step = 1.02f;  // 减小尺度变化步长
    scale_weight = 0.95f;
    interp_factor = 0.02f;  // 减小更新率
    sigma = 0.2f;
    cell_size = 4;
    template_size = 96;
}

void KCFTracker::init(cv::Rect bbox, cv::Mat image)
{
    pos.x = bbox.x + bbox.width / 2.0f;
    pos.y = bbox.y + bbox.height / 2.0f;
    target_size = bbox.size();
    
    // 提取特征并训练分类器
    Mat features;
    getFeatures(image, bbox, features);
    train(features, 1.0f);
}

cv::Rect KCFTracker::update(cv::Mat image)
{
    // 提取特征
    Mat features;
    Rect search_window(
        cvRound(pos.x - target_size.width * padding / 2.0f),
        cvRound(pos.y - target_size.height * padding / 2.0f),
        cvRound(target_size.width * padding),
        cvRound(target_size.height * padding)
    );
    getFeatures(image, search_window, features);
    
    // 检测目标
    float peak_value;
    Point2f new_pos = detect(features, tmpl, peak_value);
    pos = new_pos;
    
    // 更新尺度
    if (MULTISCALE) {
        scale *= getScale(image, pos, target_size);
        scale = std::max(scale, 0.2f);
        scale = std::min(scale, 5.0f);
    }
    
    // 更新模型
    Rect new_bbox(
        cvRound(pos.x - target_size.width * scale / 2.0f),
        cvRound(pos.y - target_size.height * scale / 2.0f),
        cvRound(target_size.width * scale),
        cvRound(target_size.height * scale)
    );
    getFeatures(image, new_bbox, features);
    train(features, interp_factor);
    
    return new_bbox;
}

void KCFTracker::getFeatures(const cv::Mat & image, const cv::Rect & roi, cv::Mat & features)
{
    Mat patch = getSubWindow(image, Point2f(roi.x + roi.width/2.0f, roi.y + roi.height/2.0f), roi.size());
    
    // 确保patch大小正确
    if (patch.empty()) {
        features = Mat();
        return;
    }

    // 调整patch大小为模板大小
    Mat resized_patch;
    resize(patch, resized_patch, Size(template_size, template_size));
    
    vector<Mat> feature_channels;
    
    if (HOG) {
        Mat hog_features;
        getHOGFeatures(resized_patch, roi, hog_features);
        if (!hog_features.empty()) {
            feature_channels.push_back(hog_features);
        }
    }
    
    if (LAB) {
        Mat color_features;
        getColorFeatures(resized_patch, roi, color_features);
        if (!color_features.empty()) {
            feature_channels.push_back(color_features);
        }
    }
    
    // 使用vector合并特征
    if (!feature_channels.empty()) {
        if (feature_channels.size() == 1) {
            features = feature_channels[0];
        } else {
            // 确保所有特征维度匹配
            int total_rows = 0;
            for (const Mat& channel : feature_channels) {
                total_rows += channel.rows;
            }
            
            // 预分配内存
            features.create(total_rows, feature_channels[0].cols, feature_channels[0].type());
            int current_row = 0;
            
            // 逐行复制
            for (const Mat& channel : feature_channels) {
                channel.copyTo(features.rowRange(current_row, current_row + channel.rows));
                current_row += channel.rows;
            }
        }
    } else {
        features = Mat();
    }
}

void KCFTracker::getHOGFeatures(const cv::Mat & image, const cv::Rect & roi, cv::Mat & features)
{
    try {
        // 简化版HOG特征提取
        Mat gray;
        if (image.channels() > 1) {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        // 计算梯度
        Mat dx, dy;
        Sobel(gray, dx, CV_32F, 1, 0);
        Sobel(gray, dy, CV_32F, 0, 1);
        
        // 计算梯度幅值和方向
        Mat magnitude, angle;
        cartToPolar(dx, dy, magnitude, angle);
        
        // 确保特征维度正确
        features = magnitude.reshape(1, magnitude.rows * magnitude.cols);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in HOG feature extraction: " << e.what() << std::endl;
        features = Mat();
    }
}

void KCFTracker::getColorFeatures(const cv::Mat & image, const cv::Rect & roi, cv::Mat & features)
{
    try {
        Mat lab;
        if (image.channels() == 3) {
            cvtColor(image, lab, COLOR_BGR2Lab);
        } else {
            lab = image.clone();
        }
        
        // 分离通道
        vector<Mat> channels;
        split(lab, channels);
        
        // 合并所有通道的特征
        Mat all_features;
        for (auto& channel : channels) {
            Mat reshaped = channel.reshape(1, channel.rows * channel.cols);
            if (all_features.empty()) {
                all_features = reshaped;
            } else {
                vconcat(all_features, reshaped, all_features);
            }
        }
        
        features = all_features;
    } catch (const cv::Exception& e) {
        std::cerr << "Error in color feature extraction: " << e.what() << std::endl;
        features = Mat();
    }
}

void KCFTracker::train(cv::Mat x, float interp_factor)
{
    try {
        Mat k = createGaussianPeak(x.rows, x.cols);
        
        if (tmpl.empty()) {
            tmpl = x.clone();
            alphaf = k.clone();
        } else {
            // 确保维度匹配
            if (x.size() != tmpl.size()) {
                resize(x, x, tmpl.size());
            }
            tmpl = tmpl * (1 - interp_factor) + x * interp_factor;
            alphaf = alphaf * (1 - interp_factor) + k * interp_factor;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in training: " << e.what() << std::endl;
    }
}

cv::Point2f KCFTracker::detect(cv::Mat z, cv::Mat x, float &peak_value)
{
    try {
        // 确保特征向量维度匹配
        if (z.size() != x.size()) {
            resize(z, z, x.size());
        }

        Mat k;
        matchTemplate(z, x, k, TM_CCORR_NORMED);
        
        double minVal, maxVal;
        Point minLoc, maxLoc;
        minMaxLoc(k, &minVal, &maxVal, &minLoc, &maxLoc);
        
        peak_value = (float)maxVal;

        // 如果相似度太低，返回上一次位置
        if (peak_value < 0.1f) {
            return pos;
        }

        // 计算相对位移
        float dx = (maxLoc.x - k.cols/2.0f) * (target_size.width / template_size);
        float dy = (maxLoc.y - k.rows/2.0f) * (target_size.height / template_size);
        
        // 添加位移限制
        float max_displacement = target_size.width * 0.2f;  // 限制最大位移为目标宽度的20%
        dx = std::max(-max_displacement, std::min(dx, max_displacement));
        dy = std::max(-max_displacement, std::min(dy, max_displacement));
        
        // 返回新的目标中心位置
        return Point2f(pos.x + dx, pos.y + dy);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in detection: " << e.what() << std::endl;
        peak_value = 0;
        return pos;
    }
}

cv::Mat KCFTracker::getSubWindow(const cv::Mat &image, const cv::Point2f &center, const cv::Size &size)
{
    Rect roi(
        cvRound(center.x - size.width/2.0f),
        cvRound(center.y - size.height/2.0f),
        size.width,
        size.height
    );
    
    Mat patch;
    if (roi.x < 0 || roi.y < 0 || roi.x + roi.width > image.cols || roi.y + roi.height > image.rows) {
        // 处理边界情况
        roi = roi & Rect(0, 0, image.cols, image.rows);
        patch = Mat::zeros(size, image.type());
        image(roi).copyTo(patch(Rect(0, 0, roi.width, roi.height)));
    } else {
        patch = image(roi).clone();
    }
    
    return patch;
}

cv::Mat KCFTracker::createGaussianPeak(int sizey, int sizex)
{
    Mat_<float> res(sizey, sizex);
    
    int syh = (sizey) / 2;
    int sxh = (sizex) / 2;
    
    float output_sigma = std::sqrt((float)sizex * sizey) / padding * output_sigma_factor;
    float mult = -0.5f / (output_sigma * output_sigma);
    
    for (int i = 0; i < sizey; i++) {
        for (int j = 0; j < sizex; j++) {
            int ih = i - syh;
            int jh = j - sxh;
            res(i, j) = std::exp(mult * (float)(ih * ih + jh * jh));
        }
    }
    return res;
}

float KCFTracker::getScale(const cv::Mat &image, const cv::Point2f &pos, const cv::Size &base_size)
{
    vector<float> scales;
    scales.push_back(1.0f);
    
    for (int i = 1; i <= 2; ++i) {
        scales.push_back(pow(scale_step, i));
        scales.push_back(pow(1.0f/scale_step, i));
    }
    
    vector<float> responses;
    float best_scale = 1.0f;
    float best_response = -1;
    
    for (float scale : scales) {
        Size scaled_size(cvRound(base_size.width * scale), cvRound(base_size.height * scale));
        Mat patch = getSubWindow(image, pos, scaled_size);
        
        Mat features;
        getFeatures(patch, Rect(0, 0, patch.cols, patch.rows), features);
        
        float response;
        detect(features, tmpl, response);
        
        if (response > best_response) {
            best_response = response;
            best_scale = scale;
        }
    }
    
    return best_scale;
}
