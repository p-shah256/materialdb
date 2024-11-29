#include "gridtables.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace std;

TableDetector::TableDetector(const Parameters &params) : params(params) {}

pair<cv::Mat, cv::Mat> TableDetector::preprocessImage(const cv::Mat &image) {
    cv::Mat gray, binary;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::adaptiveThreshold(gray, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV,
                          11, 2);
    cv::medianBlur(binary, binary, 3);

    // Calculate kernel length based on image size
    int kernelLength =
        static_cast<int>(min(image.rows, image.cols) * params.kernelLengthRatio);
    kernelLength = max(kernelLength, 11);
    cv::Mat vertical= binary.clone();
    cv::Mat verticalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, kernelLength));
    cv::erode(vertical, vertical, verticalKernel, cv::Point(-1, -1), 3);
    cv::dilate(vertical, vertical, verticalKernel, cv::Point(-1, -1), 3);


    cv::Mat horizontal= binary.clone();
    cv::Mat horizontalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelLength, 1));
    cv::erode(horizontal, horizontal, horizontalKernel, cv::Point(-1, -1), 3);
    cv::dilate(horizontal, horizontal, horizontalKernel, cv::Point(-1, -1), 3);

    // Combine lines into grid
    cv::Mat grid;
    cv::add(vertical, horizontal, grid);
    cv::Mat kernel3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(grid, grid, kernel3x3);

    return {binary, grid};
}

std::vector<Table> TableDetector::detectTables(const cv::Mat &image) {
    auto [binary, grid] = preprocessImage(image);
    double totalArea = image.rows * image.cols;
    std::vector<Table> tables;

    // Find table contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(grid, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Process each potential table
    for (const auto &contour : contours) {
        cv::Rect bounds = cv::boundingRect(contour);
        double areaRatio = (bounds.width * bounds.height) / totalArea;

        if (areaRatio >= params.minTableAreaRatio && areaRatio <= params.maxTableAreaRatio) {

            // Extract table ROI
            cv::Mat tableRoi = grid(bounds);
            cv::Mat imageRoi = image(bounds);

            // Create table and extract cells
            Table table(bounds.x, bounds.y, bounds.width, bounds.height);
            table.cells = extractCells(imageRoi, tableRoi);
            tables.push_back(table);
        }
    }

    return tables;
}

std::vector<TableCell> TableDetector::extractCells(const cv::Mat &tableImage,
                                                   const cv::Mat &gridRoi) {

    // Invert to find cells
    cv::Mat invertedRoi;
    cv::bitwise_not(gridRoi, invertedRoi);

    // Find cell contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(invertedRoi, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Sort contours by position
    std::sort(contours.begin(), contours.end(),
              [](const std::vector<cv::Point> &c1, const std::vector<cv::Point> &c2) {
                  cv::Rect r1 = cv::boundingRect(c1);
                  cv::Rect r2 = cv::boundingRect(c2);
                  return ((r1.y / 20) * 1000 + r1.x) < ((r2.y / 20) * 1000 + r2.x);
              });

    std::vector<TableCell> cells;

    // Process cells with duplicate removal
    for (const auto &contour : contours) {
        cv::Rect bounds = cv::boundingRect(contour);

        // Skip if cell is too small
        if (bounds.width < 10 || bounds.height < 10)
            continue;

        // Check for overlap with existing cells
        bool isDuplicate = false;
        for (const auto &existingCell : cells) {
            cv::Rect existingBounds(existingCell.x, existingCell.y, existingCell.width,
                                    existingCell.height);
            if (checkCellOverlap(bounds, existingBounds)) {
                isDuplicate = true;
                break;
            }
        }

        if (!isDuplicate) {
            cv::Mat contentRoi = tableImage(bounds);
            cells.emplace_back(bounds.x, bounds.y, bounds.width, bounds.height, contentRoi);
        }
    }

    return cells;
}

bool TableDetector::checkCellOverlap(const cv::Rect &cell1, const cv::Rect &cell2) const {

    return (
        std::abs(cell1.x - cell2.x) < params.cellOverlapThreshold &&
        std::abs(cell1.y - cell2.y) < params.cellOverlapThreshold &&
        std::abs(cell1.x + cell1.width - (cell2.x + cell2.width)) < params.cellOverlapThreshold &&
        std::abs(cell1.y + cell1.height - (cell2.y + cell2.height)) < params.cellOverlapThreshold);
}

void TableDetector::showImage(const cv::Mat& image, const std::string& title) {
    // Replace spaces and special characters in title with underscores for filename
    std::string filename = title;
    std::replace(filename.begin(), filename.end(), ' ', '_');
    std::replace(filename.begin(), filename.end(), '(', '_');
    std::replace(filename.begin(), filename.end(), ')', '_');

    // Add .png extension
    filename += ".png";

    // Write image to file
    bool success = cv::imwrite(filename, image);
    if (!success) {
        std::cerr << "Failed to write image: " << filename << std::endl;
    } else {
        std::cout << "Saved image to: " << filename << std::endl;
    }
}


cv::Mat TableDetector::visualizeTables(const cv::Mat &image, const std::vector<Table> &tables,
                                       bool showCellNumbers) {

    cv::Mat result;
    if (image.channels() == 1) {
        cv::cvtColor(image, result, cv::COLOR_GRAY2BGR);
    } else {
        result = image.clone();
    }

    for (size_t tableIdx = 0; tableIdx < tables.size(); ++tableIdx) {
        const auto &table = tables[tableIdx];

        // Draw table boundary
        cv::rectangle(result, cv::Point(table.x, table.y),
                      cv::Point(table.x + table.width, table.y + table.height),
                      cv::Scalar(0, 0, 255), 2);

        // Draw cells
        for (size_t cellIdx = 0; cellIdx < table.cells.size(); ++cellIdx) {
            const auto &cell = table.cells[cellIdx];
            int absX = table.x + cell.x;
            int absY = table.y + cell.y;

            cv::rectangle(result, cv::Point(absX, absY),
                          cv::Point(absX + cell.width, absY + cell.height), cv::Scalar(0, 255, 0),
                          1);

            if (showCellNumbers) {
                cv::putText(result, std::to_string(cellIdx), cv::Point(absX + 5, absY + 20),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0), 1);
            }
        }
    }

    return result;
}
