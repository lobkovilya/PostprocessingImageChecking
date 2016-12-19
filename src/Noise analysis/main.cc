#include <iostream>
#include <cstdlib>

#include <opencv2/opencv.hpp>

#include "NoiseIrregularityDetector.h"

class BlockClassifier : public BlockGrid::BaseBlockClassifier {
public:
  BlockClassifier(double thresh, int diff_type)
  : thresh_(thresh), diff_type_(diff_type), need_computation_(true) {};

  virtual int get_class(const BlockGrid & grid, int row, int col) override;

private:
  double thresh_;
  double avg_kurt_;
  double avg_var_;
  int diff_type_;
  bool need_computation_;
};

int BlockClassifier::get_class(const BlockGrid & grid, int row, int col) {
  if (need_computation_) {
    int rows = grid.get_rows();
    int cols = grid.get_cols();
    double sum_kurt = 0.0;
    double sum_var = 0.0;
    double kurt_max = 0.0;
    double kurt_min = 100000000000000.0;
    double var_max = 0.0;
    double var_min = 100000000000000.0;
    int kurt_count = 0;
    int var_count = 0;
    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        double kurt = grid[r][c].original_kurtosis;
        double var = grid[r][c].noise_variance;
        if (std::isnormal(kurt)) {
          sum_kurt += kurt;
          ++kurt_count;
        }
        if (std::isnormal(var)) {
          sum_var += var;
          ++var_count;
        }
      }
    }
    avg_kurt_ = sum_kurt / kurt_count;
    avg_var_ = sum_var / var_count;

    for (int r = 0; r < rows; ++r) {
      for (int c = 0; c < cols; ++c) {
        double kurt = grid[r][c].original_kurtosis;
        double var = grid[r][c].noise_variance;
        if (std::isnormal(kurt)) {
          kurt_max = std::max(kurt, kurt_max);
          kurt_min = std::min(kurt, kurt_min);
        }
        if (std::isnormal(var)) {
          var_max = std::max(var, var_max);
          var_min = std::min(var, var_min);
        }
      }
    }

    std::cout << "Avg kurt: " << avg_kurt_ << " avg var: " << avg_var_ << std::endl;
    std::cout << "Kurt: min " << kurt_min << " max " << kurt_max << std::endl;
    std::cout << "Var: min " << var_min << " max " << var_max << std::endl;
    need_computation_ = false;
  }

  double diff;
  switch (diff_type_) {
    case 0:
    diff = std::abs(avg_kurt_ - grid[row][col].original_kurtosis);
    break;
    case 1:
    diff = std::abs(avg_var_ - grid[row][col].noise_variance);
    break;
    case 2:
    diff = grid[row][col].original_kurtosis;
    break;
    case 3:
    diff = grid[row][col].noise_variance;
    break;
    default:
    diff = grid[row][col].noise_variance;
  }
  return (diff < thresh_ ? 0 : 1);
}


int main(int argc, const char * argv[]) {
  if (argc != 6) {
    std::cout << "Invalid arguments." << std::endl;
    return 1;
  }

  cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
  int dct_basis_max = std::atoi(argv[3]);

  if (image.empty()) {
    std::cout << "Can't load image." << std::endl;
    return 1;
  }

  cv::Mat_<float> float_image;
  image.convertTo(float_image, CV_32FC1);

  NoiseIrregularityDetector detector(std::atoi(argv[2]), dct_basis_max);
  auto block_grid = detector.calc_noise_distribution(float_image);

  BlockClassifier classifier(std::atof(argv[5]), std::atoi(argv[4]));

  cv::Mat colored = cv::imread(argv[1]);
  cv::Mat result = block_grid->highlight_blocks_on(colored, classifier);
  std::cout << std::endl;

  cv::imwrite(cv::String(argv[1]) + "_h.bmp", result);
  cv::imshow("Result", result);
  cv::waitKey(0);

  return 0;
}
