import math
import random
import numpy as np
from shapely.geometry import LineString, Point, MultiPoint


def cvt_coords(coords, height):
    """坐标系转换"""
    return [(x, height - y) for x, y in coords]


def angle_with_x_axis(point1, point2):
    """计算两点连线与X轴的夹角"""
    x1, y1, x2, y2 = point1[0], point1[1], point2[0], point2[1]
    angle = math.atan2(y2 - y1, x2 - x1)
    return math.degrees(angle)


def intersection(rectangle, point1, point2, rectangle_height):
    """计算直线与矩形框的交点"""
    # 构造矩形所有的边
    quad = rectangle + [rectangle[0]]
    line = LineString([point1, point2])

    # 转换坐标系以适配 Shapely
    quad_cv = cvt_coords(quad, rectangle_height)
    line_cv = LineString(cvt_coords(line.coords[:], rectangle_height))
    quad_line = LineString(quad_cv)

    intersect = line_cv.intersection(quad_line)

    if isinstance(intersect, Point):
        return [list(intersect.coords)]
    elif isinstance(intersect, MultiPoint):
        return [list(p.coords) for p in intersect.geoms]
    else:
        return []


def rectangle_circumcircle(vertices):
    """计算矩形的外接圆，并在圆上随机采样生成切割弦"""
    # 1. 计算中心和半径
    center_x = (vertices[0][0] + vertices[1][0] + vertices[2][0] + vertices[3][0]) / 4
    center_y = (vertices[0][1] + vertices[1][1] + vertices[2][1] + vertices[3][1]) / 4
    rectangle_height = 2 * center_y
    radius = math.sqrt((vertices[0][0] - center_x) ** 2 + (vertices[0][1] - center_y) ** 2)
    equation = f'(x-{center_x})^2 + (y-{center_y})^2 = {radius ** 2}'

    # 2. 在圆上随机采样两个点作为切割线的端点
    angle1 = random.uniform(0, 2 * math.pi)
    angle2 = random.uniform(0, 2 * math.pi)
    point1 = (center_x + radius * math.cos(angle1), center_y + radius * math.sin(angle1))
    point2 = (center_x + radius * math.cos(angle2), center_y + radius * math.sin(angle2))

    distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    # 3. 计算切割线与图像矩形的实际交点
    intersections = np.array(intersection(vertices, point1, point2, rectangle_height))
    intersections = np.squeeze(intersections)
    intersections_cv = cvt_coords(intersections, rectangle_height)

    # 4. 计算角度
    angle = 0
    if intersections.size == 4:  # 只有当有两个交点(size=2x2=4)时才有效
        p1, p2 = intersections_cv[0], intersections_cv[1]
        angle = angle_with_x_axis(p1, p2)

    return (center_x, center_y), radius, equation, point1, point2, distance, intersections_cv, angle


def is_in_range(x, a, b):
    return a <= x <= b


def check_list(lst):
    state = 0
    for num in lst:
        if state == 0:
            if num < 0:
                state = 1
            else:
                return False
        elif state == 1:
            if num > 0: state = 2
        elif state == 2:
            if num < 0: state = 3
        elif state == 3:
            if num > 0: return False
    return state == 3