# -*- coding: utf-8 -*-
import sys
import csv
import math
import os
import cv2
import dlib
import shutil
from PIL import Image
import numpy as np
import codecs

face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/home/xpp/data/shape_predictor_68_face_landmarks.dat")


def get_dlib_key_pts(img_path, img):
    '''
    #返回（left_eye, right_eye, tip_nose, left_mouth_corner, right_mouth_corner)
    #   dlib检测原图像的人脸区域和68个人脸区域
    #返回None, 则检测不到人脸
    '''
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img[:, :, 0] = clahe.apply(img[:, :, 0])
    img[:, :, 1] = clahe.apply(img[:, :, 1])
    img[:, :, 2] = clahe.apply(img[:, :, 2])
    face_rects = face_detector(img, 1)
    if len(face_rects) <= 0:
        return None
    else:
        rect = face_rects[0]
    shape = predictor(img, rect)

    top = abs(rect.top())
    bottom = abs(rect.bottom())
    left = abs(rect.left())
    right = abs(rect.right())
    # face = img[top:bottom, left:right]
    key_pts = np.array([[abs(p.x), abs(p.y)] for p in shape.parts()])
    # print(key_pts.shape)

    left_eye = [0, 0]
    right_eye = [0, 0]
    tip_nose = [0, 0]
    left_mouth_corner = [0, 0]
    right_mouth_corner = [0, 0]
    # left_eye
    # print(sum(key_pts[36:42, 0]))
    left_eye[0] = int(sum(key_pts[36:42, 0]) / (42 - 36))
    left_eye[1] = int(sum(key_pts[36:42, 1]) / (42 - 36))
    # right_eye
    right_eye[0] = int(sum(key_pts[42:48, 0]) / (48 - 42))
    right_eye[1] = int(sum(key_pts[42:48, 1]) / (48 - 42))
    # tip_nose
    tip_nose[0] = key_pts[30][0]
    tip_nose[1] = key_pts[30][1]
    # left_mouth_corner
    left_mouth_corner[0] = key_pts[48][0]
    left_mouth_corner[1] = key_pts[48][1]
    # right_mouth_corner
    right_mouth_corner[0] = key_pts[54][0]
    right_mouth_corner[1] = key_pts[54][1]
    return key_pts, [left_eye, right_eye, tip_nose, left_mouth_corner, right_mouth_corner]

def calculate_angle(elx, ely, erx, ery):
    """
    计算图片旋转角度
    :param elx: 左眼中心坐标x
    :param ely: 左眼中心坐标y
    :param erx: 右眼中心坐标x
    :param ery: 右眼中心坐标y
    :return: 旋转角度
    """
    dx = erx - elx
    dy = ery - ely
    angle = math.atan(dy / dx) * 180 / math.pi
    return angle

def image_rote(img, angle, elx, ely, erx, ery, mlx, mly, mrx, mry, expand=1):
    """
    图片旋转
    :param img: PIL类图像
    :param angle: 旋转角度
    :param elx: 左眼中心坐标x
    :param ely: 左眼中心坐标y
    :param erx: 右眼中心坐标x
    :param ery: 右眼中心坐标y
    :param mlx: 左边嘴角坐标x
    :param mly: 左边嘴角坐标y
    :param mrx: 右边嘴角坐标x
    :param mry: 右边嘴角坐标y
    :param expand: 旋转图像时是否扩展，默认为扩展
    :return: 旋转后的图片，旋转后的各个特征点的新坐标
    """
    w, h = img.size
    img = img.rotate(angle, expand=1)  # 这里控制旋转后图像大小是否扩展
    # img.show()

    # expand = 0
    # elx, ely = pos_transform_samesize(angle, elx, ely, w, h)
    # erx, ery = pos_transform_samesize(angle, erx, ery, w, h)
    # mlx, mly = pos_transform_samesize(angle, mlx, mly, w, h)
    # mrx, mry = pos_transform_samesize(angle, mrx, mry, w, h)

    # expand = 1
    elx, ely = pos_transform_resize(angle, elx, ely, w, h)
    erx, ery = pos_transform_resize(angle, erx, ery, w, h)
    mlx, mly = pos_transform_resize(angle, mlx, mly, w, h)
    mrx, mry = pos_transform_resize(angle, mrx, mry, w, h)

    return img, elx, ely, erx, ery, mlx, mly, mrx, mry

def pos_transform_samesize(angle, x, y, w, h):
    """
    图像旋转不扩展时，点旋转后的新坐标
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    angle = angle * math.pi / 180
    matrix = [
        math.cos(angle), math.sin(angle), 0.0,
        -math.sin(angle), math.cos(angle), 0.0
    ]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = w / 2.0 - cx
    matrix[5] = h / 2.0 - cy
    x, y = transform(x, y)
    return x, y

def pos_transform_resize(angle, x, y, w, h):
    """
    图像旋转扩展时，点旋转后的新坐标
    :param angle:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    angle = angle * math.pi / 180
    matrix = [
        math.cos(angle), math.sin(angle), 0.0,
        -math.sin(angle), math.cos(angle), 0.0
    ]

    def transform(x, y, matrix=matrix):
        (a, b, c, d, e, f) = matrix
        return a * x + b * y + c, d * x + e * y + f

    # calculate output size
    xx = []
    yy = []
    for x_, y_ in ((0, 0), (w, 0), (w, h), (0, h)):
        x_, y_ = transform(x_, y_)
        xx.append(x_)
        yy.append(y_)
    ww = int(math.ceil(max(xx)) - math.floor(min(xx)))
    hh = int(math.ceil(max(yy)) - math.floor(min(yy)))

    # adjust center
    cx, cy = transform(w / 2.0, h / 2.0)
    matrix[2] = ww / 2.0 - cx
    matrix[5] = hh / 2.0 - cy

    tx, ty = transform(x, y)
    return tx, ty

def image_align_crop(image_path, image):
    """
    将人脸的眼睛和眉毛，以及嘴巴单独裁剪
    使用mtcnn进行人脸检测，定位，裁剪
    更精确的裁剪人脸
    :param img_path:
    :return:
    """
    _dlip_pts = get_dlib_key_pts(image_path, image)
    if _dlip_pts is None:
        # print("dlip_pts")
        return None
    key_pts, dlip_pts = _dlip_pts
    w, h = image.size
    lex, ley = dlip_pts[0][0], dlip_pts[0][1]  # 左眼坐标， 中心值
    rex, rey = dlip_pts[1][0], dlip_pts[1][1]  # 右眼坐标， 中心值
    nex, ney = dlip_pts[2][0], dlip_pts[2][1]  # 鼻尖坐标
    lmx, lmy = dlip_pts[3][0], dlip_pts[3][1]  # 左边嘴角左边
    rmx, rmy = dlip_pts[4][0], dlip_pts[4][1]  # 右边嘴角坐标

    # 对图像进行旋转校正
    angle = calculate_angle(lex, ley, rex, rey)  # 计算旋转角度
    image, lex, ley, rex, rey, lmx, lmy, rmx, rmy \
        = image_rote(image, angle, lex, ley, rex, rey, lmx, lmy, rmx, rmy)  # 旋转图像

    # 基本参数
    eye_width = rex - lex  # 两眼之间的距离
    ecx, ecy = (lex + rex) / 2.0, (ley + rey) / 2.0  # 两眼中心坐标
    mouth_width = rmx - lmx  # 嘴巴的宽度
    mcx, mcy = (lmx + rmx) / 2.0, (lmy + rmy) / 2.0  # 嘴巴中心坐标
    em_height = mcy - ecy  # 两眼睛中心到嘴巴中心高度
    fcx, fcy = (ecx + mcx) / 2.0, (ecy + mcy) / 2.0  # 人脸中心坐标

    # 纯脸
    if eye_width > em_height:
        alpha = eye_width
    else:
        alpha = em_height
    g_beta = 2
    g_left = int(fcx - alpha / 2.0 * g_beta)
    g_upper = int(fcy - alpha / 2.0 * g_beta)
    g_right = int(fcx + alpha / 2.0 * g_beta)
    g_lower = int(fcy + alpha / 2.0 * g_beta)
    # g_face = image[g_upper:g_lower, g_left:g_right]
    g_face = image.crop((g_left, g_upper, g_right, g_lower))

    rotate_key_pts = []
    for pt in key_pts:
        rotate_key_pts.append(pos_transform_resize(angle, pt[0], pt[1], w, h))
    return image, np.array(rotate_key_pts), g_face, g_left, g_upper, g_right, g_lower

def extract_face_key_pts_from_frame(path, face_path):
    if os.path.isdir(face_path):
        shutil.rmtree(face_path)
    os.mkdir(face_path)

    dir1 = os.listdir(path)
    for d1 in dir1:
        os.mkdir(face_path+'/'+d1)
        dir2 = os.listdir(path+'/'+d1)
        for d2 in dir2:
            image = Image.open(path+'/'+d1+'/'+d2, mode='r').convert("RGB")
            # print(path+'/'+d1+'/'+d2, image.size)
            try:
                result = image_align_crop(path+'/'+d1+'/'+d2, image)
            except:
                continue
            if result is None:
                # print(path + '/' + d1 + '/' + d2)
                continue
            image, key_pts, face, x1, y1, x2, y2 = result  # image_align_crop(image)
            try:
                face = cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2BGR)
            except:
                continue
            cv2.imwrite(face_path + '/' + d1 + '/' + d2, face)

def alignCropFace(image_path):
    image = Image.open(image_path, mode='r').convert("RGB")
    result = image_align_crop(image_path, image)
    if result is None:
        raise "Can't extract face!"
    image, key_pts, face, x1, y1, x2, y2 = result
    face = cv2.cvtColor(np.asarray(face), cv2.COLOR_RGB2BGR)
    return face

if __name__ == "__main__":
    print("Start")
    extract_face_key_pts_from_frame("/home/xpp/data/VggFace2/train", "/home/xpp/data/VggFace2/train_face2")
    print("End")