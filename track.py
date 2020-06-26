import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

RATIO = 0.75
MIN_MATCH = 10

# 路径
obj_img_path = os.path.join(os.curdir, "track_test_materials", "book.jpg")
src_video_path = os.path.join(os.curdir, "track_test_materials", "book.mp4")
dst_video_path = os.path.join(os.curdir, "track_test_materials", "track.mp4")

obj_img = cv2.imread(obj_img_path, 0)
h, w = obj_img.shape

sift = cv2.xfeatures2d.SIFT_create()

kp_obj, des_obj = sift.detectAndCompute(obj_img, None)

# 读取视频的第一帧:
video_capture = cv2.VideoCapture(src_video_path)
status, frame = video_capture.read()
frame_count = 0

# 输出视频:
fps = 24  # 每秒24帧
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
video_writer = cv2.VideoWriter(dst_video_path, fourcc, fps, size)

# 读取接下来的每一帧
while status is True:

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp_frame, des_frame = sift.detectAndCompute(frame, None)

    # 定义FLANN匹配器
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    # 使用KNN算法匹配
    raw_matches = flann.knnMatch(des_obj, des_frame, k=2)

    good_matches = []
    for m, n in raw_matches:
        if m.distance < RATIO * n.distance:
            good_matches.append(m)

    if len(good_matches) < MIN_MATCH:
        print(
            f"ERROR: Not enough good matches: {len(good_matches)} matches in frame {frame_count}"
        )
        continue
    else:
        src_pts = np.float32([kp_obj[m.queryIdx].pt
                              for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt
                              for m in good_matches]).reshape(-1, 1, 2)

    H, homo_status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # pts_vertex是原图像obj的四个顶点
    pts_vertex = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                             [w - 1, 0]]).reshape(-1, 1, 2)

    # 计算变换后当前帧中物体所在的四个顶点坐标位置
    dst_vertex = cv2.perspectiveTransform(pts_vertex, H)

    # 根据四个顶点坐标位置在frame画出变换后的边框
    frame_bbox = cv2.polylines(frame, [np.int32(dst_vertex)], True,
                               (255, 0, 0), 3, cv2.LINE_AA)

    video_writer.write(frame_bbox)

    # 下一帧
    status, frame = video_capture.read()
    frame_count += 1

video_capture.release()
video_writer.release()
