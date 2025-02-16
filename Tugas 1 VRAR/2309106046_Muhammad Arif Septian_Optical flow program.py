# Nama: Muhammad Arif Septian
# NIM: 2309106046

import cv2
import numpy as np

# Fungsi untuk menghaluskan arah optical flow menggunakan average filtering
def smooth_flow_directions(flow, window_size=7):
    kernel = np.ones((window_size, window_size)) / (window_size * window_size)
    flow_x = cv2.filter2D(flow[..., 0], -1, kernel)
    flow_y = cv2.filter2D(flow[..., 1], -1, kernel)
    return np.stack([flow_x, flow_y], axis=-1)

# Fungsi untuk mendapatkan mask untuk area dengan pergerakan signifikan
def get_significant_motion_mask(gray1, gray2, threshold=25):
    diff = cv2.absdiff(gray1, gray2)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((7,7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


# Fungsi untuk menghitung optical flow dari gambar
def optical_flow_calculate(frame1_path, frame2_path):
    # Baca gambar
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)

    # Convert ke grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Tambahkan Gaussian blur
    gray1 = cv2.GaussianBlur(gray1, (7,7), 0)
    gray2 = cv2.GaussianBlur(gray2, (7,7), 0)

    # Dapatkan mask area pergerakan signifikan
    motion_mask = get_significant_motion_mask(gray1, gray2)

    # Hitung optical flow
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=7,
        winsize=31,
        iterations=7,
        poly_n=7,
        poly_sigma=1.8,
        flags=0
    )

    # Haluskan arah flow
    smoothed_flow = smooth_flow_directions(flow)

    # Buat grid
    h, w = gray1.shape
    step = 20
    y, x = np.mgrid[step//2:h:step, step//2:w:step].reshape(2,-1).astype(int)

    # Ambil flow pada titik-titik grid
    fx, fy = smoothed_flow[y, x].T

    # Gunakan frame pertama sebagai base visualization
    result_frame = frame1.copy()

    # Hitung magnitude dan arah
    magnitude = np.sqrt(fx**2 + fy**2)
    angles = np.arctan2(fy, fx)

    # Filter magnitude
    max_magnitude = np.max(magnitude)
    min_magnitude_threshold = max_magnitude * 0.12

    # Struktur data untuk tracking area yang sudah digambar
    drawn_areas = np.zeros_like(gray1)

    # Urutkan berdasarkan magnitude
    indices = np.argsort(-magnitude)

    # Parameter panah
    base_arrow_size = 12
    direction_window = 5

    for idx in indices:
        if magnitude[idx] < min_magnitude_threshold:
            continue

        x1, y1 = x[idx], y[idx]

        if motion_mask[y1, x1] == 0:
            continue

        check_radius = step//3
        if drawn_areas[max(0, y1-check_radius):min(h, y1+check_radius),
                      max(0, x1-check_radius):min(w, x1+check_radius)].any():
            continue

        y_start = max(0, idx - direction_window)
        y_end = min(len(indices), idx + direction_window + 1)
        neighbor_angles = angles[indices[y_start:y_end]]
        if np.std(neighbor_angles) > np.pi/6:
            continue

        arrow_size = base_arrow_size * min(magnitude[idx] / max_magnitude * 1.5, 1.2)

        x2 = int(x1 + arrow_size * np.cos(angles[idx]))
        y2 = int(y1 + arrow_size * np.sin(angles[idx]))

        arrow_color = (0, 0, 255)  # Pure red

        cv2.arrowedLine(result_frame, (x1, y1), (x2, y2), arrow_color, 1, tipLength=0.4, line_type=cv2.LINE_AA)

        cv2.circle(drawn_areas, (x1, y1), check_radius, 255, -1)

    # Simpan dan tampilkan
    cv2.imwrite('optical_flow_result.jpg', result_frame)
    cv2.imshow('Optical Flow - Press Q to quit', result_frame)
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

optical_flow_calculate('gambar1.png', 'gambar2.png')