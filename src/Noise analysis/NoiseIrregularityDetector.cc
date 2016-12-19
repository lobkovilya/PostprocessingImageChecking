#include "NoiseIrregularityDetector.h"

#include <iostream>
#include <cstdio>
#include <cmath>

#include <opencv2/opencv.hpp>

const double PI = 3.14159265;

namespace {

double shifted_moment(const std::vector<float> & data, int moment, double shift) {
  double sum = 0.0;
  int total = data.size();
  for (auto it = data.begin(); it != data.end(); ++it) {
    sum += std::pow(*it - shift, moment) / total;
  }
  return sum;
}

/**
 * Calculates moment for image: E((image - shift)^moment)
 */
double shifted_moment(const cv::Mat_<float> & image, int moment, double shift) {
  double sum = 0.0;
  size_t count = 0;
  for (auto it = image.begin(); it != image.end(); ++it) {
    sum += std::pow(*it - shift, moment);
    ++count;
  }
  return sum / count;
}

/**
 * Calculates kurtosis for image
 */
double kurtosis(const cv::Mat_<float> & image) {
  double mean = shifted_moment(image, 1, 0.0);
  double mc4 = shifted_moment(image, 4, mean);
  double mc2 = shifted_moment(image, 2, mean);
  return mc4 / (mc2 * mc2);
}

double kurtosis(const std::vector<float> & data, double mean, double second_central_moment) {
  double mc4 = shifted_moment(data, 4, mean);
  return mc4 / (second_central_moment * second_central_moment);
}

double kurtosis(const cv::Mat_<float> & image, double mean, double second_central_moment) {
  double mc4 = shifted_moment(image, 4, mean);
  return mc4 / (second_central_moment * second_central_moment);
}

}

NoiseIrregularityDetector::NoiseIrregularityDetector(int block_size, int dct_basis_max)
    : block_size_(block_size),
      dct_basis_max_(dct_basis_max),
      blocks_(nullptr) {}

std::shared_ptr<BlockGrid> NoiseIrregularityDetector::calc_noise_distribution(const cv::Mat_<float> & image) {
  source_size_ = image.size();
  compute_blocks(image);
  return blocks_;
}

NoiseIrregularityDetector::Block NoiseIrregularityDetector::optimal_params(const cv::Mat_<float> & block_image) {
  int dct_blocks = block_size_ / dct_basis_max_;
  int freq_resp_count = dct_basis_max_ * dct_basis_max_ - 1;
  std::vector<cv::Mat_<float>> freq_resps(freq_resp_count, cv::Mat_<float>(dct_blocks, dct_blocks));

  for (int rdb = 0; rdb < dct_blocks; ++rdb) {
    for (int cdb = 0; cdb < dct_blocks; ++cdb) {
      cv::Mat_<float> roi = block_image(cv::Rect(cdb * dct_basis_max_,
                                                 rdb * dct_basis_max_,
                                                 dct_basis_max_, dct_basis_max_));
      cv::Mat_<float> dct_result(roi.size());
      cv::dct(roi, dct_result);

      for (int i = 0; i < dct_basis_max_; ++i) {
        for (int j = 0; j < dct_basis_max_; ++j) {
          if(i + j) {
            freq_resps[i * dct_basis_max_ + j - 1](rdb, cdb) = dct_result(i, j);
          }
        }
      }
    }
  }

  const double total = dct_basis_max_ * dct_basis_max_;
  double v[4] = {0.0, 0.0, 0.0, 0.0};
  for (cv::Mat_<float> & fr : freq_resps) {
    cv::Scalar mean;
    cv::Scalar std_dev;
    cv::meanStdDev(fr, mean, std_dev);
    double variance = std_dev[0] * std_dev[0];
    double kurt = kurtosis(fr, mean[0], variance);

    double kurt_sqrt = std::sqrt(kurt);
    v[0] += kurt_sqrt / total;
    v[1] += 1.0 / (variance * variance) / total;
    v[2] += kurt_sqrt / variance / total;
    v[3] += 1.0 / variance / total;
  }

  double original_kurtosis = (v[0] * v[1] - v[2] * v[3]) / (v[1] - v[3] * v[3]);
  double noise_variance = 1.0 / v[3] - v[0] / v[3] * (1.0 / original_kurtosis);

  return Block(original_kurtosis, noise_variance);
}

void NoiseIrregularityDetector::compute_blocks(const cv::Mat_<float> & image) {
  int rows = image.rows;
  int cols = image.cols;
  int block_rows = rows / block_size_;
  int block_cols = cols / block_size_;
  blocks_ = std::shared_ptr<BlockGrid>(new BlockGrid(block_rows, block_cols, block_size_));

  for (int block_row = 0; block_row < block_rows; ++block_row) {
    for (int block_col = 0; block_col < block_cols; ++block_col) {
      cv::Mat_<float> block_image = image(cv::Rect(block_col * block_size_,
                                                   block_row * block_size_,
                                                   block_size_, block_size_));
      (*blocks_)[block_row][block_col] = optimal_params(block_image);
    }
  }
}
