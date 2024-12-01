import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import argparse
import matplotlib.pyplot as plt


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
        self.grid = None
        self.binary = None
        self.image = None

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.image = image
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
        kernel_len = int(min(img_height, img_width) * self.kernel_length_ratio)
        kernel_len = max(kernel_len, 11)  # Ensure minimum kernel size

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        vertical = cv2.erode(binary, vertical_kernel, iterations=3)
        vertical = cv2.dilate(vertical, vertical_kernel, iterations=3)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        horizontal = cv2.erode(binary, horizontal_kernel, iterations=3)
        horizontal = cv2.dilate(horizontal, horizontal_kernel, iterations=3)

        grid = cv2.add(vertical, horizontal)
        grid = cv2.dilate(grid, np.ones((3,3), np.uint8), iterations=1)
        self.grid = grid
        self.binary = binary

        return binary, grid

    def detect_tables(self) -> List[Table]:
        img_height, img_width = self.image.shape[:2]
        total_area = img_height * img_width

        contours, _ = cv2.findContours(
            self.grid,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area_ratio = (w * h) / total_area

            if self.min_table_area_ratio <= area_ratio <= self.max_table_area_ratio:
                table_roi = self.grid[y:y+h, x:x+w]
                cells = self._extract_cells(self.image[y:y+h, x:x+w], table_roi)
                tables.append(Table(x, y, w, h, cells))

        return tables

    def _extract_cells(self, table_image: np.ndarray, grid_roi: np.ndarray) -> List[TableCell]:
        inverted_roi = cv2.bitwise_not(grid_roi)
        contours, _ = cv2.findContours(
            inverted_roi,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        def get_contour_precedence(contour: np.ndarray) -> float:
            x, y, w, h = cv2.boundingRect(contour)
            return (y // 20) * 1000 + x

        contours = sorted(contours, key=get_contour_precedence)

        # Process cells with duplicate removal
        cells = []
        for idx, contour in enumerate(contours):
            x, y, w, h = cv2.boundingRect(contour)
            if w < 10 or h < 10:
                continue
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
                content_roi = table_image[y:y+h, x:x+w]
                cells.append(TableCell(x, y, w, h, content_roi))

        return cells

    def _check_cell_overlap(
        self,
        cell1: Tuple[int, int, int, int],
        cell2: Tuple[int, int, int, int]
    ) -> bool:
        x1, y1, w1, h1 = cell1
        x2, y2, w2, h2 = cell2

        return (
            abs(x1 - x2) < self.cell_overlap_threshold and
            abs(y1 - y2) < self.cell_overlap_threshold and
            abs(x1 + w1 - (x2 + w2)) < self.cell_overlap_threshold and
            abs(y1 + h1 - (y2 + h2)) < self.cell_overlap_threshold
        )

def show_image(image: np.ndarray, title: str = 'Image') -> None:
    plt_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
    result = image.copy()

    for table_idx, table in enumerate(tables):
        cv2.rectangle(
            result,
            (table.x, table.y),
            (table.x + table.width, table.y + table.height),
            (0, 0, 255),
            2
        )

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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Table detection script using OpenCV.")
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image."
    )
    parser.add_argument(
        "--min_table_area_ratio",
        type=float,
        default=0.05,
        help="Minimum area ratio for a table to be detected (default: 0.05)."
    )
    parser.add_argument(
        "--max_table_area_ratio",
        type=float,
        default=0.95,
        help="Maximum area ratio for a table to be detected (default: 0.95)."
    )
    parser.add_argument(
        "--kernel_length_ratio",
        type=float,
        default=0.15,
        help="Kernel length ratio for line detection (default: 0.15)."
    )
    parser.add_argument(
        "--show_cell_numbers",
        action="store_true",
        help="Show cell numbers in the visualized table output."
    )

    args = parser.parse_args()

    # Load the image
    image = cv2.imread(args.image_path)
    if image is None:
        raise ValueError(f"Could not read image at path: {args.image_path}")

    # Initialize TableDetector with user-specified or default parameters
    detector = TableDetector(
        min_table_area_ratio=args.min_table_area_ratio,
        max_table_area_ratio=args.max_table_area_ratio,
        kernel_length_ratio=args.kernel_length_ratio
    )

    binary, grid = detector.preprocess_image(image)
    show_image(image, 'Original Image')
    show_image(binary, 'Binary Image')
    show_image(grid, 'Grid Structure')

    tables = detector.detect_tables()
    show_image(grid, f'Number of tables found {tables.count(tables)}')
    viewableTables = visualize_tables(image, tables)
    show_image(viewableTables, "Tables")

if __name__ == "__main__":
    main()
