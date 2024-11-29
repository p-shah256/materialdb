// TableDetector.hpp
#pragma once

#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/imgproc.hpp> // For image processing functions
#include <string>
#include <vector>

struct TableCell {
    int x;
    int y;
    int width;
    int height;
    cv::Mat contentRoi;

    TableCell(int x_, int y_, int w_, int h_, const cv::Mat &roi)
        : x(x_), y(y_), width(w_), height(h_), contentRoi(roi.clone()) {}
};

struct Table {
    int x;
    int y;
    int width;
    int height;
    std::vector<TableCell> cells;

    Table(int x_, int y_, int w_, int h_) : x(x_), y(y_), width(w_), height(h_) {}
};

class TableDetector {
  public:
    struct Parameters {
        double minTableAreaRatio;
        double maxTableAreaRatio;
        double kernelLengthRatio;
        int thresholdValue;
        int gaussianKernelSize;
        int cellOverlapThreshold;

        Parameters(double minTableArea = 0.05, double maxTableArea = 0.95,
                   double kernelLength = 0.15, int threshold = 200, int gaussianKernel = 5,
                   int overlapThresh = 3)
            : minTableAreaRatio(minTableArea), maxTableAreaRatio(maxTableArea),
              kernelLengthRatio(kernelLength), thresholdValue(threshold),
              gaussianKernelSize(gaussianKernel), cellOverlapThreshold(overlapThresh) {}
    };

    std::pair<cv::Mat, cv::Mat> preprocessImage(const cv::Mat &image);
    explicit TableDetector(const Parameters &params = Parameters());
    std::vector<TableCell> extractCells(const cv::Mat &tableImage, const cv::Mat &gridRoi);
    std::vector<Table> detectTables(const cv::Mat &image);
    static void showImage(const cv::Mat &image, const std::string &title);
    static cv::Mat visualizeTables(const cv::Mat &image, const std::vector<Table> &tables,
                                   bool showCellNumbers = true);

  private:
    Parameters params;
    bool checkCellOverlap(const cv::Rect &cell1, const cv::Rect &cell2) const;
};
