import cv2
import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """Return points ordered as top-left, top-right, bottom-right, bottom-left."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array(
        [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
        dtype="float32",
    )
    # Use RANSAC to obtain a robust homography in case corners are noisy.
    M, _ = cv2.findHomography(rect, dst, method=cv2.RANSAC)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def detect_document_corners(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2)
    return None


def detect_corners_hough(image: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    if lines is None:
        return None
    points = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        points.extend([(x1, y1), (x2, y2)])
    points = np.array(points, dtype=np.float32)
    if len(points) < 4:
        return None
    # Use k-means to cluster endpoints into four groups representing corners.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(points, 4, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.float32)
    return order_points(centers)


def perspective_correction(image: np.ndarray) -> np.ndarray:
    corners = detect_document_corners(image)
    if corners is None:
        corners = detect_corners_hough(image)
    if corners is None:
        return image
    return four_point_transform(image, corners)


def denoise_and_enhance(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(bilateral)
    thresh = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 5
    )
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def remove_furniture(binary: np.ndarray) -> np.ndarray:
    inv = 255 - binary
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel, iterations=1)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        aspect = w / float(h)
        if 100 < area < 1000 and 0.3 < aspect < 3.0:
            x, y, width, height, _ = stats[i]
            cv2.rectangle(inv, (x, y), (x + width, y + height), 0, -1)
    result = 255 - inv
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=1)
    return result


def preprocess_floorplan(image_path: str, output_path: str | None = None) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")
    corrected = perspective_correction(image)
    enhanced = denoise_and_enhance(corrected)
    without_furniture = remove_furniture(enhanced)
    final = denoise_and_enhance(cv2.cvtColor(without_furniture, cv2.COLOR_GRAY2BGR))
    if output_path is not None:
        cv2.imwrite(output_path, final)
    return final


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess floorplan image")
    parser.add_argument("input", help="Path to input floorplan image")
    parser.add_argument("-o", "--output", help="Path to save preprocessed image")
    args = parser.parse_args()
    preprocess_floorplan(args.input, args.output)
