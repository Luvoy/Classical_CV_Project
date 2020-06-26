import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pickle
import logging


def load_cifar100(file_path, num):
    r"""Load cifar100 dataset, please download cifar100 from https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz and unzip it.
    加载cifar100数据集，请从上述网址下载并解压。

    Args:
        file_path (str): the path of unzipped cifar100 dataset. 已经解压的cifar100数据集的路径。
        num (int): the number of dataset to return（from start). 数据集的前多少个。

    Returns:
        tuple of nparray: nparray of images, shape:num*3*32*32, and nparray of labels, shape:num. 返回两个np数组构成的元组，num*3*32*32的图片，和num和标签。
    """
    with open(file_path, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        images = datadict['data']
        labels = datadict['fine_labels']
        images = images.reshape(-1, 3, 32, 32)
        labels = np.array(labels)
    f.close()
    return images[:num], labels[:num]


def get_des_list(imgs, trans_mode="PATHS"):
    r"""Convert an img dataset into a list of SIFT feature descriptors for each img. There are two ways of data transmission, which can transfer the tensor of the img dataset(the first dimension must be the number of imgs, and the picture will be automatically converted into a gray img without considering the channel of the img), or transfer an Iterable object which contains the paths of some images.
    将图片集转化成每个图片的sift特征描述组成的列表。有两种数据传输方式，可以传递图片集的张量（第一维必须是图片个数，此时无需考虑图片的通道，图片会自动转化成灰度图），或者传递一个图片集的路径组成的可迭代对象。

    Args:
        imgs (nparray or Iterable of str): img dataset. 图片集。
        trans_mode (str, optional): transmission mode, PATHS or MATRIX. Defaults to "PATHS". 传输方式，路径方式或张量。默认张量。

    Returns:
        list: the list of SIFT feature descriptors for each img. 每个图片的sift特征描述组成的列表。
    """
    sift = cv2.xfeatures2d.SIFT_create()
    des_list = list()
    if trans_mode == "PATHS":
        for img_path in imgs:
            img_gray = cv2.imread(img_path, 0)
            kp, des = sift.detectAndCompute(img_gray, None)
            if des is not None:
                # 为了避免后续解析出现维度问题， 这里需要升维
                des = np.expand_dims(des, axis=1)
                des_list.append(des)
        return des_list

    elif trans_mode == "MATRIX":
        for img in imgs:

            # 判断图片的通道数, 并转化为灰度图
            if img.ndim == 2 or img.ndim == 3 and (img.shape[-1] == 1
                                                   or img.shape[0] == 1):
                img_gray = img
            if img.ndim == 3 and img.shape[0] == 3:
                img_gray = cv2.cvtColor(cv2.merge(img), cv2.COLOR_RGB2GRAY)
            if img.ndim == 3 and img.shape[-1] == 3:
                img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            kp, des = sift.detectAndCompute(img_gray, None)
            if des is not None:
                # 为了避免后续解析出现维度问题， 这里需要升维
                des = np.expand_dims(des, axis=1)
                des_list.append(des)
        return des_list

    else:
        print(f"ERROR: Wrong trans_mode {trans_mode}")
        return None


def get_cluster_centers(des_list, num_centers, random_state=33):
    r"""The SIFT feature descriptors of many images are clustered. Kmeans algorithm of sklearn is used. Each component of the SIFT feature of each picture is simply traversed.
    对很多个图片的sift特征描述进行聚类。使用了sklearn的KMEANS算法。会朴素地遍历每个图片的sift特征的每个分量。

    Args:
        des_list (Iterable): the SIFT feature descriptors of many images. 多个图片的sift特征描述。
        num_centers (int): the number of centers. 聚类中心个数。
        random_state (int, optional): Determines random number generation for centroid initialization. Use an int to make the randomness deterministic. See https://scikit-learn.org/stable/glossary.html#term-random-state. Defaults to 33. 初始化的随机数。

    Returns:
        nparray: cluster centers. 聚类中心。
    """
    kmeans = KMeans(n_clusters=num_centers, random_state=random_state)
    des_matrix = np.array([d[0] for des in des_list for d in des])
    logger.debug(f"des matrix shape: {des_matrix.shape}")
    kmeans.fit(des_matrix)
    centers = kmeans.cluster_centers_
    return centers


def des_2_feature_vec(des, num_centers, centers):
    r"""Based on the bag-of-words model, the SIFT feature descriptors of an image is mapped to the clustering center, so that each image is described by only one codebook vector. L2 normalization is used。
    基于BOW模型，将一幅图像的SIFT特征描述映射到聚类中心, 这样每一幅图像只用一个码本矢量来描述。使用了L2正则。

    Args:
        des (nparray): SIFT feature descriptors of an image. 一幅图像的SIFT特征描述。
        num_centers (int): the number of centers. 聚类中心个数。
        centers (nparray): cluster centers. 聚类中心。

    Returns:
        nparray: the vector, shape: 1*num_centers. 生成的矢量。
    """
    img_feature_vec = np.zeros((1, num_centers), 'float32')

    for i, d in enumerate(des):
        feature_k_rows = np.ones((num_centers, 128), 'float32')
        feature_k_rows = feature_k_rows * d
        feature_k_rows = np.sum((feature_k_rows - centers)**2, axis=1)
        index = np.argmin(feature_k_rows)
        img_feature_vec[0][index] += 1
        img_feature_vec = preprocessing.normalize(img_feature_vec, norm='l2')
    return img_feature_vec


def get_all_vec(des_list, num_centers, centers):
    r"""Get all the vectors of many images.
    获取所有图片的向量。

    Args:
        des_list (Iterable):  SIFT feature descriptors of many images.
        num_centers (int): the number of centers. 聚类中心个数。
        centers (nparray): cluster centers. 聚类中心。

    Returns:
        nparray: the vector, shape: len(des_list)*num_centers. 生成的矢量。
    """
    all_vec = np.zeros((len(des_list), num_centers), 'float32')
    for i, des in enumerate(des_list):
        if des is None:
            pass
        else:
            all_vec[i] = des_2_feature_vec(des, num_centers, centers)
    return all_vec


def retriveal(gallery,
              one_img,
              num_centers,
              num_close=3,
              gallery_trans_mode="MATRIX",
              one_img_trans_mode="MATRIX"):
    r"""Retrieve similar images from a sufficient number of images. SIFT algorithm, kmeans clustering algorithm and bag-of-words model are used. You can enter an img tensor or paths.
    在足量图片中检索相似图片。利用了sift算法，Kmeans聚类算法和BOW模型。可以传入图片张量或者路径。

    Args:
        gallery (nparray or Iterable of str): img dataset, 4d tensor or list of paths. 图片集，4维张量或者路径列表。
        one_img (nparray or str): one image, 3d tensor or file path. 一张图片，3维张量或者路径列表。
        num_centers (int):  the number of centers. 聚类中心个数。
        num_close (int, optional): number of the most similar imgs. Defaults to 3. 最相似图片个数，默认3。
        gallery_trans_mode (str, optional): transmission mode for train images, PATHS or MATRIX. Defaults to "PATHS". 图片库的传输方式，路径方式或张量。默认张量。
        one_img_trans_mode (str, optional): transmission mode for images to retrieve, PATHS or MATRIX. Defaults to "PATHS". 待检索图片的传输方式，路径方式或张量。默认张量。

    Returns:
        list: list of indexes of the most similar imgs in gallery. 最相似图片在图片库中的索引组成的列表。
    """
    logger.debug("get des list of gallery")
    gallery_des_list = get_des_list(gallery, trans_mode=gallery_trans_mode)
    logger.debug(f"len of list of des: {len(gallery_des_list)}")

    logger.debug("get cluster centers")
    centers = get_cluster_centers(gallery_des_list, num_centers=num_centers)

    logger.debug("get all vectors")
    gallery_vecs = get_all_vec(gallery_des_list,
                               num_centers=num_centers,
                               centers=centers)

    one_img_list = [one_img]
    logger.debug("get des list of test img")
    one_test_des = get_des_list(one_img_list, trans_mode=one_img_trans_mode)
    logger.debug("get vector of test img")
    one_test_vec = des_2_feature_vec(one_test_des[0],
                                     num_centers=num_centers,
                                     centers=centers)

    logger.debug("compute similar images")
    distance = np.sum((gallery_vecs - np.tile(one_test_vec,
                                              (gallery_vecs.shape[0], 1)))**2,
                      axis=1)
    retriveal_index = np.argsort(distance)[:num_close]
    return retriveal_index.tolist()


def img_preprocess(img, resize=500, sharpen=9):
    r"""Preprocess the picture. Includes equal-proportional scaling and sharpening. Sharpening uses 2-D convolution.
    对图片进行预处理。包括等比例放缩和锐化。其中锐化使用了2维卷积。

    Args:
        img (nparray): one input img, cv2 format. 一张输入图片，cv2格式。
        resize (int, optional): the resized target img width. Defaults to 500. 放缩后图片的宽度。
        sharpen (int, optional): the center of the filter kernel for sharpen. Defaults to 9. 用于锐化的滤波的卷积核的中心值。

    Returns:
        nparray: img after preprocessing, cv2 style. 预处理后的图片，cv2格式。
    """
    img_preprocessed = img
    if resize:
        h = img_preprocessed.shape[0]
        w = img_preprocessed.shape[1]
        img_preprocessed = cv2.resize(img_preprocessed,
                                      (resize, int(h * resize / w)))
    if sharpen > 0:
        img_preprocessed = cv2.filter2D(
            img_preprocessed, -1,
            np.array([[-1, -1, -1], [-1, sharpen, -1], [-1, -1, -1]]))

    return img_preprocessed


if __name__ == "__main__":
    # 定义logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s %(name)s %(filename)s %(message)s",
        datefmt="%Y/%m/%d %X")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    # ########################## 测试1 ##########################
    # https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    # train_images, train_labels = load_cifar100(
    #     os.path.join(os.curdir, "retriveal_dataset", "cifar-100-python",
    #                  "train"), 500)
    # test_images, test_labels = load_cifar100(
    #     os.path.join(os.curdir, "retriveal_dataset", "cifar-100-python",
    #                  "test"), 100)
    # one_rand_list = np.random.choice(100, 1)
    # one_test_img_list = test_images[one_rand_list]

    # expected_result_num = 3
    # num_centers = 100
    # search_result_indexes = retriveal(train_images,
    #                                   one_test_img_list[0],
    #                                   num_centers=num_centers,
    #                                   num_close=expected_result_num)

    # # 绘图部分
    # fig, axs = plt.subplots(nrows=1,
    #                         ncols=expected_result_num + 1,
    #                         figsize=(10, 1.5))
    # for i in range(expected_result_num + 1):
    #     axs[i].axis("off")
    #     if i == 0: # 每一行的第一张图放原图,后面放结果图
    #         axs[i].set_title("origin: " + str(test_labels[one_rand_list]))
    #         axs[i].imshow(np.transpose(one_test_img_list[0], (1, 2, 0)))
    #     else:
    #         axs[i].set_title(
    #             str(search_result_indexes[i - 1]) + ": " +
    #             str(train_labels[search_result_indexes[i - 1]]))
    #         axs[i].imshow(
    #             np.transpose(train_images[search_result_indexes[i - 1]],
    #                          (1, 2, 0)))
    # plt.show()

    # ########################## 测试2 ##########################
    # 开始准备训练数据
    cat_dog_raw_images_path = os.path.join(os.curdir, "retriveal_dataset",
                                           "cat-dog", "raw")
    cat_dog_train_images_path = os.path.join(os.curdir, "retriveal_dataset",
                                             "cat-dog", "train-preprocessed")

    train_path_list = []  # 最终的train集, 即所有图片的路径(而不是名称)组成的列表
    for img_name in os.listdir(cat_dog_raw_images_path):
        img_path = os.path.join(cat_dog_raw_images_path, img_name)
        img = cv2.imread(img_path)

        # 预处理图片
        img_preprocessed = img_preprocess(img, resize=300, sharpen=9)
        cv2.imwrite(os.path.join(cat_dog_train_images_path, img_name),
                    img_preprocessed)

        train_path_list.append(
            os.path.join(cat_dog_train_images_path, img_name))

    # 开始准备测试数据
    cat_dog_test_images_path = os.path.join(os.curdir, "retriveal_dataset",
                                            "cat-dog", "test")
    test_path_list = [
        os.path.join(cat_dog_test_images_path, j)
        for j in os.listdir(cat_dog_test_images_path)
    ] + [  # 再从train中随机选择几个
        train_path_list[i] for i in np.random.choice(len(train_path_list), 4)
    ]

    # 检索
    expected_result_num = 3
    num_centers = 5
    result = []
    for test_img in test_path_list:
        logger.debug("**********************")
        logger.debug(f"Retrieving {test_img}")
        result.append(
            retriveal(train_path_list,
                      test_img,
                      num_centers=num_centers,
                      num_close=expected_result_num,
                      gallery_trans_mode="PATHS",
                      one_img_trans_mode="PATHS"))

    # 绘图部分
    fig, axs = plt.subplots(nrows=len(test_path_list),
                            ncols=expected_result_num + 1,
                            figsize=(10, 10))
    for i in range(len(test_path_list)):  # 每一行是一个测试
        for j in range(expected_result_num + 1):
            axs[i, j].axis("off")
            if j == 0:  # 每一行的第一张图放原图,后面放结果图
                axs[i, j].set_title("origin: " +
                                    os.path.basename(test_path_list[i]))
                axs[i, j].imshow(
                    cv2.cvtColor(cv2.imread(test_path_list[i]),
                                 cv2.COLOR_BGR2RGB))
            else:
                axs[i, j].set_title(
                    str(result[i][j - 1]) + ": " +
                    os.path.basename(train_path_list[result[i][j - 1]]))
                axs[i, j].imshow(
                    cv2.cvtColor(cv2.imread(train_path_list[result[i][j - 1]]),
                                 cv2.COLOR_BGR2RGB))
    plt.show()
