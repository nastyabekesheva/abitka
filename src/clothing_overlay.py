import cv2
import numpy as np

def overlay_png(frame, png_path, position, scale=1.0):
    overlay = cv2.imread(png_path, cv2.IMREAD_UNCHANGED)
    if overlay is None:
        return frame

    overlay = cv2.resize(overlay, (0, 0), fx=scale, fy=scale)
    h, w = overlay.shape[:2]
    x, y = int(position[0] - w/2), int(position[1] - h/2)
    x, y = max(0, x), max(0, y)
    roi = frame[y:y+h, x:x+w]
    if roi.shape[0] != h or roi.shape[1] != w:
        return frame

    alpha = overlay[:, :, 3] / 255.0
    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha) + overlay[:, :, c] * alpha
    frame[y:y+h, x:x+w] = roi
    return frame
