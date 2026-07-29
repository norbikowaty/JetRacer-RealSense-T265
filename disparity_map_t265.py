import pyrealsense2 as rs
import cv2
import numpy as np
from math import tan, pi

def get_extrinsics(src, dst):
    extrinsics = src.get_extrinsics_to(dst)
    R = np.reshape(matrix_to_list(extrinsics.rotation), (3, 3))
    T = matrix_to_list(extrinsics.translation)
    return R, T

def matrix_to_list(matrix):
    return [x for x in matrix]

def main():
    pipe = rs.pipeline()
    cfg = rs.config()

    cfg.enable_stream(rs.stream.fisheye, 1)
    cfg.enable_stream(rs.stream.fisheye, 2)

    profile = pipe.start(cfg)

    s1 = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
    s2 = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()

    intr1 = s1.get_intrinsics()
    intr2 = s2.get_intrinsics()

    R, T = get_extrinsics(s1, s2)

    stereo_fov_rad = 90 * (pi / 180)
    stereo_size = (320, 240)
    stereo_cx = (stereo_size[0] - 1) / 2.0
    stereo_cy = (stereo_size[1] - 1) / 2.0
    stereo_focal_px = stereo_size[0] / (2 * tan(stereo_fov_rad / 2))

    P1 = np.array([[stereo_focal_px, 0, stereo_cx, 0],[0, stereo_focal_px, stereo_cy, 0],[0, 0, 1, 0]])
    P2 = np.copy(P1)
    P2[0, 3] = T[0] * stereo_focal_px

    R1 = np.eye(3)
    R2 = R

    K1 = np.array([[intr1.fx, 0, intr1.ppx], [0, intr1.fy, intr1.ppy], [0, 0, 1]])
    D1 = np.array(intr1.coeffs[:4])
    K2 = np.array([[intr2.fx, 0, intr2.ppx], [0, intr2.fy, intr2.ppy], [0, 0, 1]])
    D2 = np.array(intr2.coeffs[:4])

    map1_x, map1_y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1[:3, :3], stereo_size, cv2.CV_32FC1)
    map2_x, map2_y = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2[:3, :3], stereo_size, cv2.CV_32FC1)

    window_size = 7 
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=window_size,
        P1=8 * 1 * window_size**2,
        P2=32 * 1 * window_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=150, 
        speckleRange=1,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    prev_disparity = None
    alpha = 0.1 #0.25 

    print("Uruchomiono! Naciśnij 'q', aby wyjść.")

    try:
        while True:
            frames = pipe.wait_for_frames()
            f1 = frames.get_fisheye_frame(1)
            f2 = frames.get_fisheye_frame(2)

            if not f1 or not f2:
                continue

            img1 = np.asanyarray(f1.get_data())
            img2 = np.asanyarray(f2.get_data())

            rect1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
            rect2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

            disp_raw = stereo.compute(rect1, rect2).astype(np.float32) / 16.0
            disp_raw[disp_raw < 0] = 0

            disparity = cv2.medianBlur(disp_raw.astype(np.uint8), 5).astype(np.float32)

            if prev_disparity is None:
                prev_disparity = disparity
            else:
                disparity = cv2.addWeighted(disparity, alpha, prev_disparity, 1 - alpha, 0)
                prev_disparity = disparity

            disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

            cv2.imshow("Prosty Widok (Lewe Oko)", rect1)
            cv2.imshow("DISPARITY MAP", disp_color)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipe.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


# import pyrealsense2 as rs
# import cv2
# import numpy as np
# from math import tan, pi

# def get_extrinsics(src, dst):
#     extrinsics = src.get_extrinsics_to(dst)
#     R = np.reshape(matrix_to_list(extrinsics.rotation), (3, 3))
#     T = matrix_to_list(extrinsics.translation)
#     return R, T

# def matrix_to_list(matrix):
#     return [x for x in matrix]

# def main():
#     pipe = rs.pipeline()
#     cfg = rs.config()

#     cfg.enable_stream(rs.stream.fisheye, 1)
#     cfg.enable_stream(rs.stream.fisheye, 2)

#     profile = pipe.start(cfg)

#     # 1. Kalibracja T265
#     s1 = profile.get_stream(rs.stream.fisheye, 1).as_video_stream_profile()
#     s2 = profile.get_stream(rs.stream.fisheye, 2).as_video_stream_profile()

#     intr1 = s1.get_intrinsics()
#     intr2 = s2.get_intrinsics()

#     R, T = get_extrinsics(s1, s2)

#     stereo_fov_rad = 90 * (pi / 180)
#     stereo_size = (640, 480)
#     stereo_cx = (stereo_size[0] - 1) / 2.0
#     stereo_cy = (stereo_size[1] - 1) / 2.0
#     stereo_focal_px = stereo_size[0] / (2 * tan(stereo_fov_rad / 2))

#     P1 = np.array([[stereo_focal_px, 0, stereo_cx, 0],
#                    [0, stereo_focal_px, stereo_cy, 0],
#                    [0, 0, 1, 0]])
#     P2 = np.copy(P1)
#     P2[0, 3] = T[0] * stereo_focal_px

#     R1 = np.eye(3)
#     R2 = R

#     K1 = np.array([[intr1.fx, 0, intr1.ppx], [0, intr1.fy, intr1.ppy], [0, 0, 1]])
#     D1 = np.array(intr1.coeffs[:4])
#     K2 = np.array([[intr2.fx, 0, intr2.ppx], [0, intr2.fy, intr2.ppy], [0, 0, 1]])
#     D2 = np.array(intr2.coeffs[:4])

#     map1_x, map1_y = cv2.fisheye.initUndistortRectifyMap(K1, D1, R1, P1[:3, :3], stereo_size, cv2.CV_32FC1)
#     map2_x, map2_y = cv2.fisheye.initUndistortRectifyMap(K2, D2, R2, P2[:3, :3], stereo_size, cv2.CV_32FC1)

#     window_size =7  # Większe okno = mniej szumu
#     stereo = cv2.StereoSGBM_create(
#         minDisparity=0,
#         numDisparities=64,
#         blockSize=window_size,
#         P1=8 * 1 * window_size**2,
#         P2=32 * 1 * window_size**2,
#         disp12MaxDiff=1,
#         uniquenessRatio=15,
#         speckleWindowSize=200, 
#         speckleRange=2,
#         mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
#     )

#     prev_disparity = None
#     alpha = 0.25  

#     print("Uruchomiono! Naciśnij 'q', aby wyjść.")

#     try:
#         while True:
#             frames = pipe.wait_for_frames()
#             f1 = frames.get_fisheye_frame(1)
#             f2 = frames.get_fisheye_frame(2)

#             if not f1 or not f2:
#                 continue

#             img1 = np.asanyarray(f1.get_data())
#             img2 = np.asanyarray(f2.get_data())

#             rect1 = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)
#             rect2 = cv2.remap(img2, map2_x, map2_y, cv2.INTER_LINEAR)

#             disp_raw = stereo.compute(rect1, rect2).astype(np.float32) / 16.0
#             disp_raw[disp_raw < 0] = 0

#             disparity = cv2.medianBlur(disp_raw.astype(np.uint8), 5).astype(np.float32)

#             if prev_disparity is None:
#                 prev_disparity = disparity
#             else:
#                 disparity = cv2.addWeighted(disparity, alpha, prev_disparity, 1 - alpha, 0)
#                 prev_disparity = disparity

#             disp_vis = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
#             disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

#             cv2.imshow("Prosty Widok (Lewe Oko)", rect1)
#             cv2.imshow("DISPARITY MAP", disp_color)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     finally:
#         pipe.stop()
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()


