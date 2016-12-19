#include "BlockGrid.h"

#include <stdexcept>
#include <fstream>
#include <iterator>
#include <opencv2/opencv.hpp>

BlockGrid::BlockGrid(int rows, int cols, int block_size)
    : rows_(rows), cols_(cols),
      block_size_(block_size),
      grid_(rows_, Row(cols_)) {}

BlockGrid::Block & BlockGrid::at(int row, int col) {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("");
  }
  return grid_[row][col];
}

const BlockGrid::Block & BlockGrid::at(int row, int col) const {
  if (row >= rows_ || col >= cols_) {
    throw std::out_of_range("");
  }
  return grid_[row][col];
}

cv::Mat BlockGrid::highlight_blocks_on(const cv::Mat & image, BaseBlockClassifier & classifier) const {
  if (image.cols < cols_ * block_size_ ||
      image.rows < rows_ * block_size_) {
    throw std::invalid_argument("Image too small");
  }

  cv::Mat classes = cv::Mat::zeros(image.size(), image.type());
  cv::Scalar colors[2] = {cv::Scalar(0, 0, 150), cv::Scalar(0, 150, 0)};
  for (int r = 0; r < rows_; ++r) {
    for (int c = 0; c < cols_; ++c) {
      int block_class = classifier.get_class(*this, r, c);
      cv::Mat roi = classes(cv::Rect(c * block_size_,
                                   r * block_size_,
                                   block_size_, block_size_));
      roi.setTo(colors[block_class]);
    }
  }

  cv::Mat ret(image.size(), image.type());
  cv::add(image, classes, ret);
  return ret;
}

std::shared_ptr<BlockGrid> BlockGrid::from_file(const std::string & filename) {
  std::ifstream is(filename);

  int rows, cols, block_size;
  is >> rows >> cols >> block_size;
  std::shared_ptr<BlockGrid> grid(new BlockGrid(rows, cols, block_size));

  for (Row & row : grid->grid_) {
    for (Block & block : row) {
      is >> block;
    }
  }
  return grid;
}

void BlockGrid::dump_to_file(const std::string & filename) const {
  std::ofstream os(filename);

  os << rows_ << ' ' << cols_ << ' ' << block_size_ << std::endl;
  std::ostream_iterator<Block> it(os, "\n");
  for (const Row & row : grid_) {
    std::copy(row.begin(), row.end(), it);
  }
}

std::ostream & operator<<(std::ostream & os, const BlockGrid::Block & block) {
  os << block.original_kurtosis << ' ' << block.noise_variance;
  return os;
}

std::istream & operator>>(std::istream & is, BlockGrid::Block & block) {
  is >> block.original_kurtosis >> block.noise_variance;
  return is;
}
