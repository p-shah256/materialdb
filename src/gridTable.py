import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TableCell:
    x: int
    y: int
    width: int
    height: int
    content_roi: np.ndarray

@dataclass
class Table:
    x: int
    y: int
    width: int
    height: int
    cells: List[TableCell]

class TableDetector:
    def __init__(
        self,
        min_table_area_ratio: float = 0.05,
        max_table_area_ratio: float = 0.95,
        kernel_length_ratio: float = 0.15,
        threshold_value: int = 200,
        gaussian_kernel_size: int = 5,
        cell_overlap_threshold: int = 3
    ):
        self.min_table_area_ratio = min_table_area_ratio
        self.max_table_area_ratio = max_table_area_ratio
        self.kernel_length_ratio = kernel_length_ratio
        self.threshold_value = threshold_value
        self.gaussian_kernel_size = gaussian_kernel_size
        self.cell_overlap_threshold = cell_overlap_threshold

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply adaptive thresholding instead of simple binary threshold
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,  # Block size
            2    # C constant
        )

        # Denoise using median blur
        binary = cv2.medianBlur(binary, 3)

        img_height, img_width = image.shape[:2]
        kernel_length = int(min(img_height, img_width) * self.kernel_length_ratio)
        kernel_length = max(kernel_length, 11)  # Ensure minimum kernel size

        # Detect lines with improved morphological operations
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))

        # Detect and enhance vertical lines
        vertical = cv2.erode(binary, vertical_kernel, iterations=3)
        vertical = cv2.dilate(vertical, vertical_kernel, iterations=3)

        # Detect and enhance horizontal lines
        horizontal = cv2.erode(binary, horizontal_kernel, iterations=3)
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=3)

        # Combine lines into grid
        grid = cv2.add(vertical, horizontal)
        grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)

        return binary, grid

    def detect_tables(self, image: np.ndarray) -> List[Table]:
        """
        Detect tables in the image.

        Args:
            image: Input image

        Returns:
            List of detected Table objects
        """
        binary, grid = self.preprocess_image(image)
        img_height, img_width = image.shape[:2]
        total_area = img_height * img_width

        # Find table contours
        contours, _ = cv2.findContours(
            grid,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / total_area

            if self.min_table_area_ratio <= area_ratio <= self.max_table_area_ratio:
                # Extract and process cells for this table
                table_roi = grid[y:y+h, x:x+w]
                cells = self._extract_cells(image[y:y+h, x:x+w], table_roi)

                tables.append(Table(x, y, w, h, cells))

        return tables

    def _extract_cells(self, table_image: np.ndarray, grid_roi: np.ndarray) -> List[TableCell]:
        """
        Extract individual cells from a table.

        Args:
            table_image: Original image region containing the table
            grid_roi: Grid structure for the table region

        Returns:
            List of TableCell objects
        """
        # Invert to find cells
        inverted_roi = cv2.bitwise_not(grid_roi)

        # Find cell contours
        contours, _ = cv2.findContours(
            inverted_roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by position
        def get_contour_precedence(contour: np.ndarray) -> float:
            x, y, w, h = cv2.boundingRect(contour)
            return (y // 20) * 1000 + x

        contours = sorted(contours, key=get_contour_precedence)

        # Process cells with duplicate removal
        cells = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)

            # Skip if cell is too small
            if w < 10 or h < 10:
                continue

            # Check for overlap with existing cells
            is_duplicate = False
            for existing_cell in cells:
                if self._check_cell_overlap(
                    (x, y, w, h),
                    (existing_cell.x, existing_cell.y,
                     existing_cell.width, existing_cell.height)
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                # Extract cell content from original image
                content_roi = table_image[y:y+h, x:x+w]
                cells.append(TableCell(x, y, w, h, content_roi))

        return cells

    def _check_cell_overlap(
        self,
        cell1: Tuple[int, int, int, int],
        cell2: Tuple[int, int, int, int]
    ) -> bool:
        """
        Check if two cells overlap significantly.
        """
        x1, y1, w1, h1 = cell1
        x2, y2, w2, h2 = cell2

        return (
            abs(x1 - x2) < self.cell_overlap_threshold and
            abs(y1 - y2) < self.cell_overlap_threshold and
            abs(x1 + w1 - (x2 + w2)) < self.cell_overlap_threshold and
            abs(y1 + h1 - (y2 + h2)) < self.cell_overlap_threshold
        )

def show_image(image: np.ndarray, title: str = 'Image') -> None:
    """
    Display an image using matplotlib.

    Args:
        image: Image to display
        title: Title for the image window
    """
    import matplotlib.pyplot as plt

    # Convert BGR to RGB if image is color
    if len(image.shape) == 3:
        plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        plt_image = image

    plt.figure(figsize=(10, 10))
    plt.title(title)
    plt.imshow(plt_image, cmap='gray' if len(image.shape) == 2 else None)
    plt.axis('off')
    plt.show()

def visualize_tables(
    image: np.ndarray,
    tables: List[Table],
    show_cell_numbers: bool = True
) -> np.ndarray:
    """
    Visualize detected tables and cells.

    Args:
        image: Original image
        tables: List of detected tables
        show_cell_numbers: Whether to show cell numbers in visualization

    Returns:
        Image with visualized tables and cells
    """
    result = image.copy()
    if len(image.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    for table_idx, table in enumerate(tables):
        # Draw table boundary
        cv2.rectangle(
            result,
            (table.x, table.y),
            (table.x + table.width, table.y + table.height),
            (0, 0, 255),
            2
        )

        # Draw cells
        for cell_idx, cell in enumerate(table.cells):
            abs_x = table.x + cell.x
            abs_y = table.y + cell.y

            cv2.rectangle(
                result,
                (abs_x, abs_y),
                (abs_x + cell.width, abs_y + cell.height),
                (0, 255, 0),
                1
            )

            if show_cell_numbers:
                cv2.putText(
                    result,
                    str(cell_idx),
                    (abs_x + 5, abs_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1
                )

    return result

# Complete usage example with visualization:
if __name__ == "__main__":
    # Read your image
    image = cv2.imread('your_image.jpg')
    if image is None:
        raise ValueError("Could not read image!")

    # Initialize detector
    detector = TableDetector(
        min_table_area_ratio=0.05,
        max_table_area_ratio=0.95,
        kernel_length_ratio=0.15
    )

    # Get preprocessed images
    binary, grid = detector.preprocess_image(image)

    # Show preprocessing results
    show_image(image, 'Original Image')
    show_image(binary, 'Binary Image')
    show_image(grid, 'Grid Structure')

    # Detect tables
    tables = detector.detect_tables(image)

    # Visualize final results
    result_image = visualize_tables(image, tables, show_cell_numbers=True)
    show_image(result_image, f'Detected {len(tables)} Tables')

    # Optionally, visualize individual tables and their cells
    for i, table in enumerate(tables):
        # Extract table region
        table_roi = image[table.y:table.y+table.height, table.x:table.x+table.width]

        # Visualize table cells
        roi_with_cells = table_roi.copy()
        for j, cell in enumerate(table.cells):
            # Draw cell rectangle
            cv2.rectangle(
                roi_with_cells,
                (cell.x, cell.y),
                (cell.x + cell.width, cell.y + cell.height),
                (0, 255, 0),
                2
            )
            # Add cell numbers
            cv2.putText(
                roi_with_cells,
                str(j),
                (cell.x + 5, cell.y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1
            )

        show_image(roi_with_cells, f'Table {i+1} - Detected Cells')
"""
# Initialize detector with custom parameters if needed
detector = TableDetector(
    min_table_area_ratio=0.05,
    max_table_area_ratio=0.95,
    kernel_length_ratio=0.15
)

# Detect tables
tables = detector.detect_tables(image)

# Visualize results
result_image = visualize_tables(image, tables)
"""
