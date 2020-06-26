import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# 读取一张源图和一张场景图, 在场景中检测出源图物体

RATIO = 0.75
MIN_MATCH = 10

# 读取灰度图
img1_gray = cv2.imread(
    os.path.join(os.curdir, "detect_test_images", "box.jpg"), 0)
img2_gray = cv2.imread(
    os.path.join(os.curdir, "detect_test_images", "scene.jpg"), 0)

# 读取彩图
img1 = cv2.imread(os.path.join(os.curdir, "detect_test_images", "box.jpg"), 1)
img2 = cv2.imread(os.path.join(os.curdir, "detect_test_images", "scene.jpg"),
                  1)

if img1_gray is None or img2_gray is None:
    print(r"ERROR: image is None")
    exit(0)

# sift 算子
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 定义FLANN匹配器
flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
# 使用KNN算法匹配
raw_matches = flann.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in raw_matches:
    if m.distance < RATIO * n.distance:
        good_matches.append(m)

if len(good_matches) < MIN_MATCH:
    print(f"ERROR: Not enough good matches: {len(good_matches)}")
    exit(0)
else:
    src_pts = np.float32([kp1[m.queryIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt
                          for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # 降维并转换成列表格式
    matches_mask = status.ravel().tolist()

    h, w = img1_gray.shape

    # pts_vertex是图像img1_gray的四个顶点
    pts_vertex = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                             [w - 1, 0]]).reshape(-1, 1, 2)
    # 计算变换后的四个顶点坐标位置
    dst_vertex = cv2.perspectiveTransform(pts_vertex, H)

    # 根据四个顶点坐标位置在img2_gray图像画出变换后的边框
    img_bbox = cv2.polylines(img2, [np.int32(dst_vertex)], True, (255, 0, 0),
                             3, cv2.LINE_AA)

    # 画出匹配点
    img_draw_match = cv2.drawMatches(img1,
                                     kp1,
                                     img_bbox,
                                     kp2,
                                     good_matches,
                                     None,
                                     matchColor=(0, 255, 0),
                                     singlePointColor=None,
                                     matchesMask=matches_mask,
                                     flags=2)

    # 保存
    cv2.imwrite(
        os.path.join(os.curdir, "detect_test_images", "img_draw_match.jpg"),
        img_draw_match)

    cv2.imwrite(os.path.join(os.curdir, "detect_test_images", "img_bbox.jpg"),
                img_bbox)

    # 显示
    plt.imshow(cv2.cvtColor(img_draw_match, cv2.COLOR_BGR2RGB))
    plt.show()
