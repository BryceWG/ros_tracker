#ifndef KCFTRACKER_HPP
#define KCFTRACKER_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class KCFTracker
{
public:
    // 构造函数
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    // 初始化跟踪器
    void init(cv::Rect bbox, cv::Mat image);
    
    // 更新跟踪器，返回目标新位置
    cv::Rect update(cv::Mat image);

private:
    // 特征提取
    void getFeatures(const cv::Mat & image, const cv::Rect & roi, cv::Mat & features);
    
    // HOG特征
    void getHOGFeatures(const cv::Mat & image, const cv::Rect & roi, cv::Mat & features);
    
    // 颜色特征
    void getColorFeatures(const cv::Mat & image, const cv::Rect & roi, cv::Mat & features);
    
    // 训练分类器
    void train(cv::Mat x, float interp_factor);
    
    // 检测目标
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);
    
    // 子窗口提取
    cv::Mat getSubWindow(const cv::Mat &image, const cv::Point2f &center, const cv::Size &size);
    
    // 高斯响应图生成
    cv::Mat createGaussianPeak(int sizey, int sizex);
    
    // 尺度估计
    float getScale(const cv::Mat &image, const cv::Point2f &pos, const cv::Size &base_size);

private:
    // 配置参数
    bool HOG;
    bool LAB;
    bool MULTISCALE;
    bool FIXED_WINDOW;
    
    // 跟踪状态
    cv::Point2f pos;
    cv::Size target_size;
    float scale;
    
    // 模型参数
    cv::Mat alphaf;
    cv::Mat prob;
    cv::Mat tmpl;
    float interp_factor;
    float sigma;
    float lambda;
    int cell_size;
    int template_size;
    float padding;
    float output_sigma_factor;
    float scale_step;
    float scale_weight;
};

#endif 