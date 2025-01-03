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
    MULTISCALE = false; // 关闭多尺度更新以减少漂移
    LAB = lab;
    
    // 初始化跟踪状态
    scale = 1.0f;
    
    // 优化参数以提高稳定性
    lambda = 0.0001f;
    padding = 1.5f;  // 减小搜索区域
    output_sigma_factor = 0.1f;
    scale_step = 1.01f;  // 进一步减小尺度变化步长
    scale_weight = 0.95f;
    interp_factor = 0.01f;  // 减小更新率以提高稳定性
    sigma = 0.2f;
    cell_size = 4;
    template_size = 96;
    detection_threshold = 0.4f;  // 提高检测阈值
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
        Mat gray;
        if (image.channels() > 1) {
            cvtColor(image, gray, COLOR_BGR2GRAY);
        } else {
            gray = image.clone();
        }
        
        // 使用更强的预处理
        GaussianBlur(gray, gray, Size(3,3), 0);
        
        // 计算梯度
        Mat dx, dy;
        Sobel(gray, dx, CV_32F, 1, 0, 3);
        Sobel(gray, dy, CV_32F, 0, 1, 3);
        
        // 计算梯度幅值和方向
        Mat magnitude, angle;
        cartToPolar(dx, dy, magnitude, angle);
        
        // 归一化梯度幅值
        normalize(magnitude, magnitude, 0, 1, NORM_MINMAX);
        
        // 计算HOG特征
        const int n_bins = 9;
        const float angle_step = 180.0f / n_bins;
        
        vector<Mat> hist_features;
        for(int i = 0; i < n_bins; i++) {
            Mat bin = Mat::zeros(magnitude.size(), CV_32F);
            float angle_low = i * angle_step;
            float angle_high = (i + 1) * angle_step;
            
            for(int y = 0; y < angle.rows; y++) {
                for(int x = 0; x < angle.cols; x++) {
                    float ang = angle.at<float>(y,x) * 180.0f / CV_PI;
                    if(ang < 0) ang += 180.0f;
                    
                    if(ang >= angle_low && ang < angle_high) {
                        bin.at<float>(y,x) = magnitude.at<float>(y,x);
                    }
                }
            }
            hist_features.push_back(bin);
        }
        
        // 合并特征
        Mat all_features;
        for(const Mat& hist : hist_features) {
            Mat feat = hist.reshape(1, 1);
            if(all_features.empty()) {
                all_features = feat;
            } else {
                hconcat(all_features, feat, all_features);
            }
        }
        
        features = all_features;
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
            // 添加高斯模糊以减少噪声
            Mat blurred;
            GaussianBlur(image, blurred, Size(3,3), 0);
            cvtColor(blurred, lab, COLOR_BGR2Lab);
        } else {
            lab = image.clone();
        }
        
        // 分离通道
        vector<Mat> channels;
        split(lab, channels);
        
        // 对每个通道进行直方图均衡化和归一化
        for(auto& channel : channels) {
            // 归一化到0-1范围
            normalize(channel, channel, 0, 1, NORM_MINMAX);
        }
        
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
            
            // 计算新模板与当前模板的相似度
            Mat correlation;
            matchTemplate(x, tmpl, correlation, TM_CCORR_NORMED);
            double similarity;
            minMaxLoc(correlation, nullptr, &similarity);
            
            // 如果相似度太低，减小更新率以防止模型漂移
            float current_interp_factor = similarity > 0.8 ? interp_factor : interp_factor * 0.5f;
            
            tmpl = tmpl * (1 - current_interp_factor) + x * current_interp_factor;
            alphaf = alphaf * (1 - current_interp_factor) + k * current_interp_factor;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "Error in training: " << e.what() << std::endl;
    }
}

float detection_threshold = 0.3f; 

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
        
        // 增强的跟踪质量评估
        Mat response_map;
        normalize(k, response_map, 0, 1, NORM_MINMAX);
        
        // 计算响应图的峰值与均值比
        Scalar mean_val = mean(response_map);
        float psr = (peak_value - mean_val[0]) / mean_val[0];  // PSR: Peak to Sidelobe Ratio
        
        if (peak_value < detection_threshold || psr < 2.0) {
            // 当相似度太低或PSR太小时，不更新位置
            return pos;
        }

        // 计算亚像素级别的峰值位置
        Point2f subpixel_delta = getSubPixelPeak(k, maxLoc);
        
        // 计算相对位移
        float dx = (maxLoc.x + subpixel_delta.x - k.cols/2.0f) * (target_size.width / template_size);
        float dy = (maxLoc.y + subpixel_delta.y - k.rows/2.0f) * (target_size.height / template_size);
        
        // 添加位移限制和平滑
        float max_displacement = target_size.width * 0.15f;  // 限制最大位移为目标宽度的15%
        dx = std::max(-max_displacement, std::min(dx, max_displacement));
        dy = std::max(-max_displacement, std::min(dy, max_displacement));
        
        // 使用指数平滑来平滑位移
        static float smooth_factor = 0.6f;
        static float prev_dx = 0, prev_dy = 0;
        dx = smooth_factor * dx + (1 - smooth_factor) * prev_dx;
        dy = smooth_factor * dy + (1 - smooth_factor) * prev_dy;
        prev_dx = dx;
        prev_dy = dy;
        
        return Point2f(pos.x + dx, pos.y + dy);
    } catch (const cv::Exception& e) {
        std::cerr << "Error in detection: " << e.what() << std::endl;
        return pos;
    }
}

Point2f KCFTracker::getSubPixelPeak(const Mat& response, const Point& peak_loc)
{
    // 获取峰值周围的3x3区域
    int x = peak_loc.x;
    int y = peak_loc.y;
    
    // 确保我们有足够的边界
    if (x < 1 || x >= response.cols-1 || y < 1 || y >= response.rows-1)
        return Point2f(0,0);
    
    // 拟合二次函数进行亚像素插值
    float dx = (response.at<float>(y, x+1) - response.at<float>(y, x-1)) * 0.5f;
    float dy = (response.at<float>(y+1, x) - response.at<float>(y-1, x)) * 0.5f;
    
    float dxx = response.at<float>(y, x+1) + response.at<float>(y, x-1) - 2.0f * response.at<float>(y, x);
    float dyy = response.at<float>(y+1, x) + response.at<float>(y-1, x) - 2.0f * response.at<float>(y, x);
    
    if (dxx == 0 || dyy == 0)
        return Point2f(0,0);
    
    return Point2f(-dx/dxx, -dy/dyy);
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
