from PIL import Image, ImageDraw
import random
import math

def draw_circles(image_size, circle_radius, num_circles):
    # 创建白色图像
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    # 绘制圆圈
    for _ in range(num_circles):
        center = (random.randint(0, image_size[0]), random.randint(0, image_size[1]))
        draw.ellipse([center[0] - circle_radius, center[1] - circle_radius,
                      center[0] + circle_radius, center[1] + circle_radius],
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
    ratio = red_pixels / white_pixels if white_pixels != 0 else 0
    return ratio

# 设置参数
image_size = (500, 500)  # 图像大小
circle_radius = 20  # 圆圈半径
num_circles = 50  # 圆圈数量

# 生成图像
image = draw_circles(image_size, circle_radius, num_circles)

# 计算红色面积/白色面积的比例
ratio = calculate_red_white_ratio(image)
# 显示图像和比例
print(f"红色面积/白色面积的比例: {ratio}")
