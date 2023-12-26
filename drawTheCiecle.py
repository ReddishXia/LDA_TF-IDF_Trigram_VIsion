import numpy as np
import matplotlib.pyplot as plt
import math as Math
from PIL import Image, ImageDraw
import random
import math

def is_overlap(x1, y1, r1, x2, y2, r2):
    # 判断两个圆是否重叠
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance < r1 + r2


def draw_non_overlapping_circles(image_size, r_List):
    # 创建白色图像
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    circles = []

    for i in range(len(r_List)):
        # 随机选择圆心坐标和半径
        x = random.uniform(0, image_size[0])
        y = random.uniform(0, image_size[1])
        radius = r_List[i]

        # 调整位置和半径确保整个圆都在画布内
        x = max(radius, min(x, image_size[0] - radius))
        y = max(radius, min(y, image_size[1] - radius))

        # 检查新圆是否与已有圆重叠
        overlap = False
        for circle in circles:
            if is_overlap(x, y, radius, circle[0], circle[1], circle[2]):
                overlap = True
                break

        # 如果重叠，重新选择位置和半径
        while overlap:
            x = random.uniform(radius, image_size[0] - radius)
            y = random.uniform(radius, image_size[1] - radius)

            overlap = False
            for circle in circles:
                if is_overlap(x, y, radius, circle[0], circle[1], circle[2]):
                    overlap = True
                    break

        # 绘制圆
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red")

        # 记录新圆的信息
        circles.append((x, y, radius))
    return calculate_red_white_ratio(image)


def draw_circles(image_size,x,y, circle_radius):
    # 创建白色图像
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)
    # 绘制圆圈
    for i in range(len(x)):

        draw.ellipse([x[i] - circle_radius[i], y[i] - circle_radius[i],
                      x[i] + circle_radius[i], y[i] + circle_radius[i]],
                     fill="red")

    return image

def calculate_red_white_ratio(image):
    # 统计红色和白色像素数量
    red_pixels = 0
    white_pixels = 0
    for pixel in image.getdata():
        if pixel == (255, 255, 255):  # 白色
            white_pixels += 1
        else:
            red_pixels += 1
    # 计算红色面积/白色面积的比例
    ratio = red_pixels / (red_pixels+white_pixels) if white_pixels != 0 else 0
    return ratio

# 设置参数
 # 图像大小


x_List=[]
y_List=[]
r_List=[]
def draw(xmin, xmax,ymin,ymax,d_x_value,d_y_value,Freq):
    original_Area=0
    xrange = [xmin, xmax]
    xpad = 0.05
    circle_prop = 0.25
    mdsheight = 530
    xdiff = xmax - xmin
    yrange = [ymin, ymax]
    ypad = 0.05
    ydiff = ymax - ymin
    mdswidth=530
    for i in range(len(d_x_value)):
        if (xdiff > ydiff):
            # 创建线性比例尺
            x_scale = np.linspace(xrange[0] - xpad * xdiff, xrange[1] + xpad * xdiff, mdswidth)
            y_scale = np.linspace(yrange[0] - 0.5*(xdiff - ydiff) - ypad*xdiff, yrange[1] + 0.5*(xdiff - ydiff) + ypad*xdiff, mdswidth)
            # 将给定的输入值映射到输出范围
            if xrange[0] - xpad * xdiff <= d_x_value[i] <= xrange[1] + xpad * xdiff:
                # 保证 fp 参数有相同的长度
                fp = np.linspace(0, mdswidth, mdswidth)
                scaled_x = np.interp(d_x_value[i], x_scale, fp)
                scaled_y = np.interp(d_y_value[i], y_scale, fp)
                R = Math.sqrt((Freq[i] / 100) * mdswidth * mdsheight * circle_prop / Math.pi)
                original_Area+=math.pi*(R**2)
                x_List.append(scaled_x+100)
                y_List.append(530 - scaled_y+100)
                r_List.append(R)
                # print(scaled_x)
                # print(530-scaled_y)
                # print(Math.sqrt((Freq/100) * mdswidth * mdsheight * circle_prop / Math.pi))
                # print("")
            else:
                print("Error: d_x_value is outside the range of x_scale.")
        else:
            # 创建线性比例尺
            fp = np.linspace(0, mdswidth, mdswidth)
            x_scale = np.linspace(xrange[0] - 0.5 * (ydiff - xdiff) - xpad * ydiff,
                                  xrange[1] + 0.5 * (ydiff - xdiff) + xpad * ydiff, mdswidth)
            y_scale = np.linspace(yrange[0] - ypad * ydiff, yrange[1] + ypad * ydiff, mdswidth,endpoint=False)

            # 将给定的输入值映射到输出范围
            scaled_x = np.interp(d_x_value[i], x_scale, fp)
            scaled_y = np.interp(d_y_value[i], y_scale, fp)
            R=Math.sqrt((Freq[i]/100) * mdswidth * mdsheight * circle_prop / Math.pi)
            original_Area += math.pi * (R ** 2)
            x_List.append(scaled_x+100)
            y_List.append(530-scaled_y+100)
            r_List.append(R)
        # print(scaled_x)
        # print(530-scaled_y)
        # print(R)
        # print("")

    image_size = (1000, 1000)
    image_size1 = (1000, 1000)
    proportion1=draw_non_overlapping_circles(image_size,r_List)
    image=draw_circles(image_size1,x_List,y_List,r_List)
    proportion=calculate_red_white_ratio(image)
    overlap_Proportion=proportion1-proportion
    if overlap_Proportion<0:
        return 0
    else:
        return proportion1-proportion
