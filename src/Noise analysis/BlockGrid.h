#ifndef BLOCK_GRID_H
#define BLOCK_GRID_H

#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class BlockGrid {
public:
  struct Block {
    double original_kurtosis;
    double noise_variance;

    Block() : original_kurtosis(0.0), noise_variance(0.0) {}

    Block(double ok, double nv)
        : original_kurtosis(ok), noise_variance(nv) {}
  };

  using Row = std::vector<Block>;
  using Grid = std::vector<Row>;

  class BaseBlockClassifier {
  public:
    virtual int get_class(const BlockGrid & grid, int row, int col) = 0;
  };

  BlockGrid(int rows, int cols, int block_size);

  static std::shared_ptr<BlockGrid> from_file(const std::string & filename);

  int get_cols() const { return  cols_; }

  int get_rows() const { return rows_; }

  Block & at(int row, int col);

  const Block & at(int row, int col) const;

  Row & operator[](int row) { return grid_[row]; }

  const Row & operator[](int row) const { return grid_[row]; }

  void dump_to_file(const std::string & filename) const;

  cv::Mat highlight_blocks_on(const cv::Mat & image, BaseBlockClassifier & classifier) const;

private:

  const int rows_;
  const int cols_;
  const int block_size_;
  Grid grid_;
};

std::ostream & operator<<(std::ostream & os, const BlockGrid::Block & block);

std::istream & operator>>(std::istream & is, BlockGrid::Block & block);

#endif //BLOCK_GRID_H
