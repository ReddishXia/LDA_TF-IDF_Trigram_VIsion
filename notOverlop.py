from PIL import Image, ImageDraw
import random


def is_overlap(x1, y1, r1, x2, y2, r2):
    # 判断两个圆是否重叠
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance < r1 + r2


def draw_non_overlapping_circles(image_size, num_circles):
    # 创建白色图像
    image = Image.new("RGB", image_size, "white")
    draw = ImageDraw.Draw(image)

    circles = []

    for _ in range(num_circles):
        # 随机选择圆心坐标和半径
        x = random.randint(0, image_size[0])
        y = random.randint(0, image_size[1])
        radius = random.randint(10, min(image_size[0], image_size[1]) // 2)

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
            x = random.randint(radius, image_size[0] - radius)
            y = random.randint(radius, image_size[1] - radius)

            overlap = False
            for circle in circles:
                if is_overlap(x, y, radius, circle[0], circle[1], circle[2]):
                    overlap = True
                    break

        # 绘制圆
        draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="red")

        # 记录新圆的信息
        circles.append((x, y, radius))

    return image


# 示例：绘制5个不重叠的圆
image_size = (500, 500)
num_circles = 3
result_image = draw_non_overlapping_circles(image_size, num_circles)
result_image.show()
