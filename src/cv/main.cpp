#include "gridtables.hpp"
#include <iostream>
#include <opencv4/opencv2/imgcodecs.hpp>

int main() {
    // Read image
    cv::Mat image = cv::imread("resources/grid1.png");
    if (image.empty()) {
        std::cerr << "Could not read image!" << std::endl;
        return -1;
    }

    // Initialize detector with custom parameters if needed
    TableDetector::Parameters params(0.05, 0.95, 0.15);
    TableDetector detector(params);

    // Get preprocessed images
    auto[binary, grid] = detector.preprocessImage(image);

    // Show preprocessing results
    TableDetector::showImage(image, "Original Image");
    TableDetector::showImage(binary, "Binary Image");
    TableDetector::showImage(grid, "Grid Structure");

    // Detect tables
    std::vector<Table> tables = detector.detectTables(image);

    // Visualize final results
    cv::Mat resultImage = TableDetector::visualizeTables(image, tables, true);
    TableDetector::showImage(resultImage,
        "Detected Tables (" + std::to_string(tables.size()) + ")");

    return 0;
}
