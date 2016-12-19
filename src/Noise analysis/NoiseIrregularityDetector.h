#ifndef NOISE_IRREGULARITY_DETECTOR_H
#define NOISE_IRREGULARITY_DETECTOR_H

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

#include "BlockGrid.h"

class NoiseIrregularityDetector {
public:
  NoiseIrregularityDetector(int block_size, int dct_basis_max);

  std::shared_ptr<BlockGrid> calc_noise_distribution(const cv::Mat_<float> & image);

private:
  using Block = BlockGrid::Block;

  Block optimal_params(const cv::Mat_<float> & block_image);

  void compute_blocks(const cv::Mat_<float> & image);

  const int block_size_;
  const int dct_basis_max_;
  std::shared_ptr<BlockGrid> blocks_;
  cv::Size source_size_;
};

#endif //NOISE_IRREGULARITY_DETECTOR_H
