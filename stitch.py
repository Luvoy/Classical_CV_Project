import numpy as np
import cv2
import matplotlib.pyplot as plt
import pysift
import os


def match_homograpy(img1,
                    img2,
                    sift_method="PYSIFT",
                    match_method="FLANN",
                    match_ratio=0.85,
                    draw_match=None,
                    min_match=4):
    r"""返回能将两张图片拼接在一起的单应性矩阵.
    两张图必须是同一相机对同一场景不同视角下的拍摄.
    可以使用python写的sift算法或者opencv自带的生成sift算子,
    可以使用FLANN或者BF匹配器

    Arguments:
        img1 {Numpy 2darray} -- source image of gray channel
        img2 {Numpy 2darray} -- target image of gray channel

    Keyword Arguments:
        sift_method {str} -- which algorithm of sift to use:
        "PYSIFT": a from-scratch algorithm implemented by python;
        "OPENCV": packaged sift algorithm in opencv (default: {"PYSIFT"})
        match_method {str} -- FLANN or BF matcher (default: {"FLANN"})
        match_ratio {float} -- distance raito threshold (default: {0.85})
        draw_match {[type]} -- match_img path to be saved. if None, do not generate a match_img(default: {None})
        min_match {int} -- minimum good match points (default: {4})

    Returns:
        [Numpy 2darray] -- homography matrix
    """
    # 空值判断
    if img1 is None or img2 is None:
        print("ERROR: input images is None.")
        return None
    if img1.ndim != img2.ndim or img1.ndim != 2:
        print(r"ERROR: image ndim is not 2")
        return None

    # 利用sift算法生成关键点和描述子
    if sift_method == "PYSIFT":
        kp1, des1 = pysift.computeKeypointsAndDescriptors(img1)
        kp2, des2 = pysift.computeKeypointsAndDescriptors(img2)
    elif sift_method == "OPENCV":
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
    else:
        print("ERROR: Undefined sift method.")
        return None

    # 匹配, 调用opencv库, 可用BFmatcher或FlannBasedMatcher
    if match_method == "FLANN":
        flann = cv2.FlannBasedMatcher(dict(algorithm=0, trees=5),
                                      dict(checks=50))
        raw_matches = flann.knnMatch(des1, des2, k=2)
    elif match_method == "BF":
        matcher = cv2.BFMatcher()  # 建立暴力匹配器
        raw_matches = matcher.knnMatch(des1, des2, k=2)
    else:
        print("ERROR: Undefined matching method.")
        return None

    good_points = []  # 用于生成单应性矩阵
    good_matches = []  # 用于画线
    for m1, m2 in raw_matches:
        if m1.distance < match_ratio * m2.distance:  # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            good_points.append((m1.trainIdx, m1.queryIdx))  # 储存索引值
            good_matches.append([m1])

    # 根据需求画出图像上的匹配点, 并保存
    if draw_match is not None:
        img_match = cv2.drawMatchesKnn(img1,
                                       kp1,
                                       img2,
                                       kp2,
                                       good_matches,
                                       None,
                                       flags=0)
        cv2.imwrite(draw_match, img_match)

    if len(good_points) >= min_match:
        image1_kp = np.float32([kp1[i].pt for (_, i) in good_points])

        image2_kp = np.float32([kp2[i].pt for (i, _) in good_points])

        # 计算用于视角变换的单应性矩阵
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC, 5.0)

    return H


def create_mask(img1, img2, phase='LEFT', smoothing_window_size=100):
    r"""返回mask矩阵，用于图像的平滑处理
    图像必须是左右拼接

    Arguments:
        img1 {numpy ndarray} -- left image
        img2 {numpy ndarray} -- right image

    Keyword Arguments:
        phase {str} --  define which part of mask to be generated: "LEFT" or "RIGHT" (default: {'LEFT'})
        smoothing_window_size {int} -- smoothing_window_size (default: {100})

    Returns:
        numpy ndarray -- mask matrix,its height is img1's height, its width is img1's width + img2's width.
    """
    # 空值判断
    if img1 is None or img2 is None:
        print("ERROR: input imagesf is None.")
        return None

    # 定义最终图的最大尺寸
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 + width_img2

    # 区分左右图，按照窗口尺寸进行平滑处理
    barrier = img1.shape[1] - smoothing_window_size
    mask = np.zeros((height_panorama, width_panorama))
    if phase == 'LEFT':
        mask[:, barrier:width_img1] = np.tile(
            np.linspace(1, 0, smoothing_window_size).T, (height_panorama, 1))
        mask[:, :barrier] = 1
    else:
        mask[:, barrier:width_img1] = np.tile(
            np.linspace(0, 1, smoothing_window_size).T, (height_panorama, 1))
        mask[:, width_img1:] = 1

    return cv2.merge([mask, mask, mask])


def stitch(imgr1, img2, H, smooth=True):
    r"""将左右两张图拼接在一起, 两张图必须是同一相机对同一场景不同视角下的拍摄, 在水平方向(左右)不能偏差太多
    会进行平滑处理和去黑边处理

    Arguments:
        img1 {numpy 3darray} -- 左图
        img2 {numpy 3darray} -- 右图
        H {numpy 2darray} -- 单应性矩阵

    Keyword Arguments:
        smooth {bool} -- 是否平滑处理 (default: {True})

    Returns:
        numpy 3darrray -- 拼接后的图片
    """
    # 空值判断
    if img1 is None or img2 is None:
        print("ERROR: input images is None.")
        return None

    # 定义最终尺寸
    height_panorama = img1.shape[0]
    width_panorama = img1.shape[1] + img2.shape[1]

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    # 左边直接填充
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    # 右边使用图像变换
    panorama2 = cv2.warpPerspective(
        img2, H, (width_panorama, height_panorama)).astype(np.float64)

    if smooth:
        # 对左边图像进行平滑处理
        mask1 = create_mask(img1, img2, phase='LEFT')
        panorama1 *= mask1

        # 对右边图像进行变换，并对变换后的图像进行平滑处理
        mask2 = create_mask(img1, img2, phase='RIGHT')
        panorama2 *= mask2

    # 叠加
    img_stitch = panorama1 + panorama2

    # 最大限度地去掉黑边
    rows, cols = np.where(img_stitch[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)
    return img_stitch[min_row:max_row, min_col:max_col, :]


if __name__ == "__main__":

    # 彩度图灰度图分开读
    img1_gray = cv2.imread("left.jpg", 0)
    img2_gray = cv2.imread("right.jpg", 0)

    img1 = cv2.imread("left.jpg", 1)
    img2 = cv2.imread("right.jpg", 1)

    # 先看看 :-)
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.show()

    img_dir = os.path.join(os.cudir, "stitch_test_images")

    # 对于用手撕的sift算法和opencv封装的sift算法分别测试
    # 对FLANN和BF的matcher也分别测试
    H_pysift_bf = match_homograpy(img1_gray,
                                  img2_gray,
                                  sift_method='PYSIFT',
                                  match_method="BF",
                                  match_ratio=0.85,
                                  draw_match=os.path.join(
                                      img_dir, "pysift_bf.jpg"),
                                  min_match=4)

    H_pysift_flann = match_homograpy(img1_gray,
                                     img2_gray,
                                     sift_method='PYSIFT',
                                     match_method="FLANN",
                                     match_ratio=0.85,
                                     draw_match=os.path.join(
                                         img_dir, "pysift_flann.jpg"),
                                     min_match=4)

    H_opcvsift_bf = match_homograpy(img1_gray,
                                    img2_gray,
                                    sift_method='OPENCV',
                                    match_method="BF",
                                    match_ratio=0.85,
                                    draw_match=os.path.join(
                                        img_dir, "opencvsift_bf.jpg"),
                                    min_match=4)
    H_opcvsift_flann = match_homograpy(img1_gray,
                                       img2_gray,
                                       sift_method='OPENCV',
                                       match_method="FLANN",
                                       match_ratio=0.85,
                                       draw_match=os.path.join(
                                           img_dir, "opencvsift_flann.jpg"),
                                       min_match=4)
    print(f"H_pysift_bf:\n{H_pysift_bf}")
    print(f"H_pysift_flann:\n{H_pysift_flann}")
    print(f"H_opcvsift_bf:\n{H_opcvsift_bf}")
    print(f"H_opcvsift_flann:\n{H_opcvsift_flann}")

    img_pysift_bf_smooth = stitch(img1, img2, H_pysift_bf, smooth=True)
    img_pysift_bf_raw = stitch(img1, img2, H_pysift_bf, smooth=False)
    img_pysift_flann_smooth = stitch(img1, img2, H_pysift_flann, smooth=True)
    img_pysift_flann_raw = stitch(img1, img2, H_pysift_flann, smooth=False)
    img_opcvsift_bf_smooth = stitch(img1, img2, H_opcvsift_bf, smooth=True)
    img_opcvsift_bf_raw = stitch(img1, img2, H_opcvsift_bf, smooth=False)
    img_opcvsift_flann_smooth = stitch(img1,
                                       img2,
                                       H_opcvsift_flann,
                                       smooth=True)
    img_opcvsift_flann_raw = stitch(img1, img2, H_opcvsift_flann, smooth=False)

    # 保存图像
    cv2.imwrite(os.path.join(img_dir, "img_pysift_bf_smooth.jpg"),
                img_pysift_bf_smooth)
    cv2.imwrite(os.path.join(img_dir, "img_pysift_flann_smooth.jpg"),
                img_pysift_flann_smooth)
    cv2.imwrite(os.path.join(img_dir, "img_pysift_bf_raw.jpg"),
                img_pysift_bf_raw)
    cv2.imwrite(os.path.join(img_dir, "img_pysift_flann_raw.jpg"),
                img_pysift_flann_raw)
    cv2.imwrite(os.path.join(img_dir, "img_opcvsift_bf_smooth.jpg"),
                img_opcvsift_bf_smooth)
    cv2.imwrite(os.path.join(img_dir, "img_opcvsift_flann_smooth.jpg"),
                img_opcvsift_flann_smooth)
    cv2.imwrite(os.path.join(img_dir, "img_opcvsift_bf_raw.jpg"),
                img_opcvsift_bf_raw)
    cv2.imwrite(os.path.join(img_dir, "img_opcvsift_flann_raw.jpg"),
                img_opcvsift_flann_raw)
