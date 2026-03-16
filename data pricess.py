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
MAX_RECURSION = 8  # 切割总刀数
DEBUG_SAVE_PROCESS = True  # 开启后生成 full_segmentation_map.png


# ===========================================

class Fragment(object):
    def __init__(self, img, pcd, transformation, flag, area):
        self.img = img
        self.pcd = pcd
        self.trans = transformation  # 记录从原图到当前碎片的总变换矩阵
        self.flag = flag
        self.area = area


class CircleCutBase:
    @staticmethod
    def generate_cut_line(basic_idx, shape, rotated_point, num_nodes=4):
        """
        生成具有随机抖动的折线，模拟真实断裂边缘
        """
        h, w = shape[:2]
        # 提取旋转后的关键切割点
        p1, p2 = rotated_point[0], rotated_point[1]

        # 1. 建立基准线：y = kx + b
        if abs(p2[0] - p1[0]) < 1e-5: p2[0] += 0.1  # 防止除以0
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - k * p1[0]

        # 2. 在基准线上生成随机抖动点
        # 我们把宽度分成几段，每段加一个随机垂直位移
        nodes_x = np.linspace(0, w, num_nodes + 2)
        nodes_y = k * nodes_x + b
        # 给中间节点添加随机波动
        nodes_y[1:-1] += np.random.uniform(-h / 15, h / 15, num_nodes)

        # 3. 使用线性插值生成完整的切割线点集
        full_x = np.arange(w)
        full_y = np.interp(full_x, nodes_x, nodes_y)

        # 4. 添加高频噪声（锯齿感）
        noise = np.random.normal(0, 1.2, size=w)
        full_y += noise

        # 5. 将像素分配给正负两面
        # basic_idx 格式为 [y, x]
        mask_res = full_y[basic_idx[:, 1]] - basic_idx[:, 0]
        mask_p = mask_res > 0
        mask_n = mask_res <= 0

        # 构造用于绘制红线的坐标点列表
        paint_line = np.stack([full_x, full_y], axis=1)

        return mask_p, mask_n, np.count_nonzero(mask_p), np.count_nonzero(mask_n), paint_line


def image_rotate_func(img, angle, intersections, pad_=15):
    """
    旋转图像并返回变换矩阵 M
    """
    basic_cover = np.argwhere(img[:, :, 3] != 0)
    if len(basic_cover) == 0: return None

    # 计算旋转
    angle_rad = np.radians(angle)
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)

    # 简化的旋转中心逻辑
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M_rot_cv2 = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 为了防止切出画布，先做变换
    rotated_img = cv2.warpAffine(img, M_rot_cv2, (w, h), flags=cv2.INTER_NEAREST)

    # 重新修正偏移，使物体靠边对齐节省空间
    new_idx = np.argwhere(rotated_img[:, :, 3] != 0)
    if len(new_idx) == 0: return None
    min_y, min_x = np.min(new_idx, axis=0)
    max_y, max_x = np.max(new_idx, axis=0)

    # 最终偏移矩阵
    M_shift = np.array([[1, 0, -min_x + pad_], [0, 1, -min_y + pad_], [0, 0, 1]], dtype=float)
    M_rot_3x3 = np.vstack([M_rot_cv2, [0, 0, 1]])
    M_total = np.matmul(M_shift, M_rot_3x3)

    final_w, final_h = max_x - min_x + pad_ * 2, max_y - min_y + pad_ * 2
    final_img = cv2.warpAffine(img, M_total[:2], (int(final_w), int(final_h)), flags=cv2.INTER_NEAREST)

    # 转换交点坐标
    new_pts = []
    for pt in intersections:
        tmp = np.matmul(M_total, [pt[0], pt[1], 1])
        new_pts.append(tmp[:2])

    return final_img, M_total, new_pts


def execute_segmentation(img_obj):
    if img_obj.shape[-1] != 4:
        img_obj = cv2.cvtColor(img_obj, cv2.COLOR_BGR2BGRA)

    h_now, w_now = img_obj.shape[:2]
    verts = [(0, 0), (w_now - 1, 0), (w_now - 1, h_now - 1), (0, h_now - 1)]

    # 寻找切割线角度
    for _ in range(30):
        circ = rectangle_circumcircle(verts)
        if len(circ[6]) == 2:
            intersections, angle = circ[6], circ[7]
            break
    else:
        return False

    # 旋转对齐
    res_rot = image_rotate_func(img_obj, -angle, intersections)
    if res_rot is None: return False
    img_rot, M_trans, rot_pts = res_rot

    # 找到有效像素区域
    basic_idx = np.argwhere(img_rot[:, :, 3] != 0)
    if len(basic_idx) < 100: return False

    # 生成锯齿切割线
    res_cut = CircleCutBase.generate_cut_line(basic_idx, img_rot.shape, rot_pts)
    mask_p, mask_n, a_p, a_n, paint_line = res_cut

    def crop_frag(mask_indices):
        new_canvas = np.zeros_like(img_rot)
        coords = basic_idx[mask_indices]
        if len(coords) < 10: return None, None
        new_canvas[coords[:, 0], coords[:, 1]] = img_rot[coords[:, 0], coords[:, 1]]

        # 再次裁剪黑边
        ys, xs = coords[:, 0], coords[:, 1]
        y1, y2, x1, x2 = np.min(ys), np.max(ys), np.min(xs), np.max(xs)
        cropped = new_canvas[y1:y2 + 1, x1:x2 + 1]
        M_crop = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=float)
        return cropped, M_crop

    i_p, m_p = crop_frag(mask_p)
    i_n, m_n = crop_frag(mask_n)

    if i_p is None or i_n is None: return False

    return (i_p, a_p, m_p), (i_n, a_n, m_n), M_trans, paint_line


def process_single_image(image_path, save_root):
    img_name = os.path.basename(image_path).split('.')[0]
    curr_save_dir = os.path.join(save_root, img_name)
    os.makedirs(curr_save_dir, exist_ok=True)

    origin_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if origin_img is None: return
    if origin_img.shape[-1] == 3: origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2BGRA)

    full_vis = origin_img.copy()  # 全景展示图
    total_area = np.count_nonzero(origin_img[:, :, 3])

    # 初始碎片（单位矩阵）
    fragment_pool = [Fragment(origin_img, None, np.eye(3), True, total_area)]
    final_fragments = []
    split_count = 0

    while fragment_pool and split_count < MAX_RECURSION:
        # 优先切最大的碎片
        fragment_pool.sort(key=lambda x: x.area, reverse=True)
        curr = fragment_pool.pop(0)

        success = False
        for attempt in range(50):
            res = execute_segmentation(curr.img)
            if res:
                (i_p, a_p, m_p), (i_n, a_n, m_n), M_step, paint_line = res

                # 核心：计算总变换矩阵
                # 坐标关系：原图 -> (curr.trans) -> 当前碎片 -> (M_step) -> 旋转后空间 -> (m_p/m_n) -> 最终碎片
                m_total_p = np.matmul(m_p, np.matmul(M_step, curr.trans))
                m_total_n = np.matmul(m_n, np.matmul(M_step, curr.trans))

                # 全景图映射：将 paint_line 从旋转后空间映回原图
                # 映射公式：P_original = inv(M_step * curr.trans) * P_rotated
                try:
                    m_to_orig = np.linalg.inv(np.matmul(M_step, curr.trans))
                    for pt in paint_line:
                        orig_pt = np.matmul(m_to_orig, [pt[0], pt[1], 1])
                        ox, oy = int(orig_pt[0]), int(orig_pt[1])
                        if 0 <= ox < full_vis.shape[1] and 0 <= oy < full_vis.shape[0]:
                            full_vis[oy, ox] = (0, 0, 255, 255)  # 画红点
                except:
                    pass

                # 尺寸检查
                if i_p.shape[0] < 30 or i_n.shape[0] < 30 or a_p < total_area * 0.002: continue

                fragment_pool.append(Fragment(i_p, None, m_total_p, True, a_p))
                fragment_pool.append(Fragment(i_n, None, m_total_n, True, a_n))
                split_count += 1
                success = True
                break

        if not success:
            final_fragments.append(curr)

    final_fragments.extend(fragment_pool)

    # 保存结果
    cv2.imwrite(os.path.join(curr_save_dir, 'full_segmentation_map.png'), full_vis)
    for idx, frag in enumerate(final_fragments):
        cv2.imwrite(os.path.join(curr_save_dir, f'frag_{idx:03d}.png'), frag.img)
        # 保存 gt.txt 为 3x3 矩阵展平
        with open(os.path.join(curr_save_dir, 'gt.txt'), 'a') as f:
            f.write(f"{idx}\n{str(frag.trans.flatten())[1:-1]}\n")
    print(f"[{img_name}] 处理完毕，碎片数: {len(final_fragments)}")


if __name__ == '__main__':
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg'))]
    for f in files:
        process_single_image(os.path.join(INPUT_DIR, f), OUTPUT_DIR)