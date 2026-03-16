import os
import random
import cv2
import numpy as np
import math
import open3d
from circle_utils import rectangle_circumcircle, is_in_range, check_list

# ================= 配置区域 =================
INPUT_DIR = r"E:\SSN\dateset"  # 输入图片文件夹路径
OUTPUT_DIR = r"E:\testdate"  # 输出碎片文件夹路径
# 注意：MIN_FRAG_SIZE 现在由代码动态计算，这里的配置仅作为保底
MAX_RECURSION =10 # 切多少刀（生成多少个碎片），建议设为 15-30
DEBUG_SAVE_PROCESS = True  # 是否保存中间过程


# ===========================================

class Fragment(object):
    def __init__(self, img, pcd, transformation, flag, area):
        self.img = img
        self.pcd = pcd
        self.trans = transformation
        self.flag = flag
        self.area = area


def down_sample(pcd, stride):
    if len(pcd) == 0: return pcd
    pcd = np.hstack((pcd, np.ones((len(pcd), 1))))
    point_cloud = open3d.geometry.PointCloud()
    point_cloud.points = open3d.utility.Vector3dVector(pcd)
    point_cloud = point_cloud.voxel_down_sample(stride)
    return np.array(point_cloud.points)[:, 0:2]


def are_points_not_inside_contour(contour, points):
    points = np.array(points, dtype=np.int32)
    for point in points:
        dist = cv2.pointPolygonTest(contour, tuple(map(int, point)), True)
        if dist <= 10:
            return True
    return False


# --- 核心切割类 ---
class CircleCutBase:
    @staticmethod
    def generate_cut_line(x, s, rotated_point, pcd_basic, num_control_points):
        y = np.zeros(len(x))
        paint_line = []

        x_process = np.vstack((x[:, 1], x[:, 0])).transpose()
        width = s[1]
        height = s[0]

        # 波动函数
        def f1_point(point_x):
            y_val = 0
            line_y_val = 0
            # 降低一点波动的剧烈程度，适配长图
            for i in range(1, 15, 1):
                a = np.random.normal(height / 200, height / 80)  # 减小振幅
                T = np.random.normal(1.5 * width, 0.3 * width)
                fi = np.random.uniform(-1, 1)
                w = 2 * np.pi / T
                y_val += ((a / (i + 1)) * np.sin(i * w * x_process[:, 0] + fi))
                line_y_val += ((a / (i + 1)) * np.sin(i * w * point_x + fi))
            return y_val, line_y_val

        a_range = np.arange(int(width / 10), int(width * 9 / 10))  # 放宽控制点范围
        b_center = rotated_point[0][1]
        b_range = np.arange(int(b_center - height / 10), int(b_center + height / 10))

        if len(b_range) == 0: b_range = np.array([int(b_center)])

        random_x, random_y = [], []
        if num_control_points > 0:
            for _ in range(20):
                random_x = np.sort(np.random.choice(a_range, num_control_points, replace=False))
                random_y = np.random.choice(b_range, num_control_points)
                check_pts = [[random_x[i], random_y[i]] for i in range(len(random_x))]

                valid = True
                if ((np.hstack((random_x, width)) - np.hstack((0, random_x))) < width / 20).any(): valid = False
                # if are_points_not_inside_contour(pcd_basic, check_pts): valid = False # 暂时关闭点在轮廓内的强校验
                if valid: break
            else:
                return False

        start_x, start_y = 0, int(rotated_point[0][1])
        end_x, end_y = width, int(rotated_point[0][1])

        key_x = [start_x] + list(random_x) + [end_x]
        key_y = [start_y] + list(random_y) + [end_y]

        for i in range(len(key_x) - 1):
            curr_x, curr_y = key_x[i], key_y[i]
            next_x, next_y = key_x[i + 1], key_y[i + 1]

            if next_x == curr_x: continue
            k = (next_y - curr_y) / (next_x - curr_x)

            mask = np.bitwise_and(x_process[:, 0] >= curr_x, x_process[:, 0] < next_x)
            if i == len(key_x) - 2:
                mask = x_process[:, 0] >= curr_x

            y[mask] += k * x_process[mask][:, 0] - k * curr_x + curr_y

            seg_x_range = np.arange(curr_x, next_x)
            seg_points = [[val, round(k * (val - curr_x) + curr_y)] for val in seg_x_range]

            if random.random() < 0.8:  # 提高波动概率
                noise_y, line_noise = f1_point(seg_x_range)
                y += noise_y
                for j, item in enumerate(seg_points):
                    if j < len(line_noise):
                        item[1] += round(line_noise[j])

            paint_line += seg_points

        res = y - x_process[:, 1]
        mask_p = res <= 0
        mask_n = res > 0

        # 【重要修改】移除严格的交点检查
        # 原代码：if not are_line_have_only_two_intertact(pcd_basic, paint_line): return False

        return mask_p, mask_n, np.count_nonzero(mask_p), np.count_nonzero(mask_n), paint_line


def image_rotate_func(img, angle, intersections, pad_=10):
    height, width = img.shape[:2]
    basic_cover = np.argwhere(img[:, :, 3] != 0)
    if len(basic_cover) == 0: return False

    angle_rad = np.radians(angle)
    cos_, sin_ = np.cos(angle_rad), np.sin(angle_rad)

    pcd = basic_cover
    pcd = down_sample(pcd, 10)

    temp_matrix = np.array([[cos_, -sin_, 0], [sin_, cos_, 0]])
    temp_pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), temp_matrix.T)

    shift_x = -temp_pcd[:, 0].min()
    shift_y = -temp_pcd[:, 1].min()

    rotate_matrix = np.array([[cos_, -sin_, shift_x + pad_],
                              [sin_, cos_, shift_y + pad_]])

    width_max = temp_pcd[:, 0].max() + shift_x
    height_max = temp_pcd[:, 1].max() + shift_y

    rotated_img = cv2.warpAffine(img, rotate_matrix, (int(width_max) + pad_ * 2, int(height_max) + pad_ * 2),
                                 flags=cv2.INTER_NEAREST, borderValue=0)

    new_point = np.matmul(np.hstack((intersections, np.ones((len(intersections), 1)))), rotate_matrix.T)
    rotate_matrix = np.vstack((rotate_matrix, [0, 0, 1]))
    return rotated_img, rotate_matrix, new_point


def execute_segmentation(img_obj):
    if img_obj.shape[-1] != 4:
        img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2BGRA)

    height, width = img_obj.shape[:2]
    vertices = [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)]

    for _ in range(50):
        center, radius, _, p1, p2, dist, intersections, angle = rectangle_circumcircle(vertices)
        if dist > 0.5 * 2 * radius and len(intersections) == 2:  # 放宽对弦长的要求
            break
    else:
        return False

    angle = 0 - angle
    res = image_rotate_func(img_obj, angle, intersections)
    if not res: return False
    img_rot, M, rotated_point = res

    # 放宽边缘距离限制
    if rotated_point[0][1] < 5 or rotated_point[0][1] > img_rot.shape[0] - 5:
        return False

    basic_cover_idx = np.argwhere(img_rot[:, :, 3] != 0)
    if len(basic_cover_idx) == 0: return False

    gray = np.zeros(img_rot.shape[:2], dtype=np.uint8)
    gray[basic_cover_idx[:, 0], basic_cover_idx[:, 1]] = 255
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours: return False
    pcd_basic = max(contours, key=len)

    rand_v = random.uniform(0, 1)
    if rand_v < 0.5:
        num_pts = 3
    elif rand_v < 0.75:
        num_pts = 2
    elif rand_v < 0.875:
        num_pts = 1
    else:
        num_pts = 0

    cut_res = CircleCutBase.generate_cut_line(basic_cover_idx, img_rot.shape[:2], rotated_point, pcd_basic, num_pts)
    if cut_res is False: return False

    mask_p, mask_n, area_p, area_n, paint_line = cut_res

    def create_frag_img(mask_indices):
        new_img = np.zeros_like(img_rot)
        coords = basic_cover_idx[mask_indices]
        if len(coords) == 0: return None, None
        new_img[coords[:, 0], coords[:, 1]] = img_rot[coords[:, 0], coords[:, 1]]

        ys, xs = coords[:, 0], coords[:, 1]
        min_y, max_y = np.min(ys), np.max(ys)
        min_x, max_x = np.min(xs), np.max(xs)
        cropped = new_img[min_y:max_y + 1, min_x:max_x + 1]

        gray_c = np.zeros(cropped.shape[:2], dtype=np.uint8)
        c_mask = cropped[:, :, 3] != 0
        gray_c[c_mask] = 255
        cs, _ = cv2.findContours(gray_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs: return None, None
        return cropped, max(cs, key=len)

    img_p, pcd_p = create_frag_img(mask_p)
    img_n, pcd_n = create_frag_img(mask_n)

    if img_p is None or img_n is None: return False

    return (img_p, pcd_p, area_p), (img_n, pcd_n, area_n), M, img_rot


def save_result(save_path, idx, img, gt_matrix):
    cv2.imwrite(os.path.join(save_path, f'fragment_{str(idx).zfill(4)}.png'), img)
    with open(os.path.join(save_path, 'gt.txt'), 'a') as f:
        f.write(f'{idx}\n')
        f.write(str(gt_matrix.flatten())[1: -1].replace('\n', '') + '\n')


def process_single_image(image_path, save_root):
    img_name = os.path.basename(image_path).split('.')[0]
    curr_save_dir = os.path.join(save_root, img_name)
    if not os.path.exists(curr_save_dir):
        os.makedirs(curr_save_dir)

    origin_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if origin_img is None:
        print(f"无法读取: {image_path}")
        return

    if origin_img.shape[-1] == 3:
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2BGRA)

    # 1. 强制加 Padding，把图像撑大
    pad_h = 100  # 增加 padding
    pad_w = 100
    h, w = origin_img.shape[:2]
    padded_img = np.zeros((h + 2 * pad_h, w + 2 * pad_w, 4), dtype=np.uint8)
    padded_img[pad_h:pad_h + h, pad_w:pad_w + w] = origin_img
    origin_img = padded_img

    bg_color = np.array([0, 0, 0])
    with open(os.path.join(curr_save_dir, 'bg.txt'), 'w') as f:
        f.write(str(bg_color)[1:-1])

    area = np.count_nonzero(origin_img[:, :, 3])
    # 动态阈值：只要大于 50像素 就算有效
    dynamic_min_size = 50
    print(f"[{img_name}] 尺寸: {w}x{h}, 处理中...")

    fragment_pool = [Fragment(origin_img, None, np.eye(3), True, area)]
    final_fragments = []
    split_count = 0

    while fragment_pool and split_count < MAX_RECURSION:
        weights = [f.area for f in fragment_pool]
        if sum(weights) == 0: break

        # 优先切大块
        curr_frag = random.choices(fragment_pool, weights=weights, k=1)[0]
        fragment_pool.remove(curr_frag)

        success = False
        # 【重要修改】尝试 100 次，死磕到底
        for attempt in range(100):
            res = execute_segmentation(curr_frag.img)
            if res:
                (i_p, p_p, a_p), (i_n, p_n, a_n), M, vis_img = res

                h_p, w_p = i_p.shape[:2]
                h_n, w_n = i_n.shape[:2]

                # 【重要修改】极度放宽条件
                if (h_p < dynamic_min_size or w_p < dynamic_min_size or
                        h_n < dynamic_min_size or w_n < dynamic_min_size or
                        a_p < area * 0.005 or  # 0.5% 面积即可
                        a_n < area * 0.005):
                    continue

                trans_p = np.matmul(M, curr_frag.trans)
                trans_n = np.matmul(M, curr_frag.trans)

                fragment_pool.append(Fragment(i_p, p_p, trans_p, True, a_p))
                fragment_pool.append(Fragment(i_n, p_n, trans_n, True, a_n))

                split_count += 1
                success = True
                if split_count % 5 == 0:
                    print(f"  -> 已生成 {split_count} 个碎片...")
                break

        if not success:
            curr_frag.flag = False
            final_fragments.append(curr_frag)

    final_fragments.extend(fragment_pool)

    print(f"图片 {img_name} 处理完成，最终生成 {len(final_fragments)} 个碎片。")
    for idx, frag in enumerate(final_fragments):
        save_result(curr_save_dir, idx, frag.img, frag.trans)


if __name__ == '__main__':
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    supported_ext = ['.jpg', '.png', '.jpeg', '.bmp']
    files = [f for f in os.listdir(INPUT_DIR) if os.path.splitext(f)[1].lower() in supported_ext]

    print(f"找到 {len(files)} 张图片，开始处理...")

    for f in files:
        process_single_image(os.path.join(INPUT_DIR, f), OUTPUT_DIR)