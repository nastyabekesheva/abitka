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

def overlay_logo(frame, logo_path, opacity=0.7, margin=10):
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)  # load with alpha channel if exists
    if logo is None:
        print(f"Logo image not found: {logo_path}")
        return frame

    # Resize logo if too big (optional)
    max_width = frame.shape[1] // 6  # max 1/6 width of frame
    if logo.shape[1] > max_width:
        scale = max_width / logo.shape[1]
        logo = cv2.resize(logo, (0, 0), fx=scale, fy=scale)

    lh, lw = logo.shape[:2]
    fh, fw = frame.shape[:2]

    # Position: top-right corner with margin
    x, y = fw - lw - margin, margin

    # Separate logo channels
    if logo.shape[2] == 4:
        # logo has alpha channel
        alpha_logo = logo[:, :, 3] / 255.0 * opacity
        alpha_frame = 1.0 - alpha_logo
        for c in range(0, 3):
            frame[y:y+lh, x:x+lw, c] = (alpha_logo * logo[:, :, c] + alpha_frame * frame[y:y+lh, x:x+lw, c])
    else:
        # no alpha channel, blend with opacity
        roi = frame[y:y+lh, x:x+lw]
        blended = cv2.addWeighted(roi, 1 - opacity, logo, opacity, 0)
        frame[y:y+lh, x:x+lw] = blended

    return frame