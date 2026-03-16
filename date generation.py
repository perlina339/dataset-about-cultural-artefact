import os
import random
import cv2
import numpy as np
import json
from scipy.interpolate import interp1d
from circle_utils import rectangle_circumcircle

# ================= 配置区域 =================
INPUT_DIR = r"E:\SSN\dateset"
OUTPUT_DIR = r"E:\traindate"
MAX_RECURSION =6  # 切割总刀数


# ===========================================

class Fragment(object):
    def __init__(self, img, mask, transformation, area, frag_id):
        self.img = img
        self.mask = mask
        self.trans = transformation
        self.area = area
        self.frag_id = frag_id


class CircleCutBase:
    @staticmethod
    def generate_cut_line(basic_idx, shape, rotated_point, num_nodes=15):
        """
        生成的裂纹通过增加节点数和插值，使其分布更均匀。
        """
        h, w = shape[:2]
        p1, p2 = rotated_point[0], rotated_point[1]

        # 1. 基础路径
        if abs(p2[0] - p1[0]) < 1e-5: p2[0] += 0.1
        k = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - k * p1[0]

        # 2. 生成密集的控制点，保证走向均匀
        nodes_x = np.linspace(0, w, num_nodes + 2)
        nodes_y = k * nodes_x + b

        # 添加中低频扰动（控制在合理范围，防止大拐弯）
        nodes_y[1:-1] += np.random.normal(0, h / 30, size=num_nodes)

        # 3. 二次插值，使线条圆滑自然
        f_interp = interp1d(nodes_x, nodes_y, kind='quadratic')
        full_x = np.arange(w)
        full_y = f_interp(full_x)

        # 4. 添加微小高频锯齿（模拟断裂面粗糙度）
        full_y += np.random.normal(0, 0.7, size=w)

        mask_res = full_y[basic_idx[:, 1]] - basic_idx[:, 0]
        mask_p = mask_res > 0
        mask_n = mask_res <= 0

        paint_line = np.stack([full_x, full_y], axis=1)
        return mask_p, mask_n, np.count_nonzero(mask_p), np.count_nonzero(mask_n), paint_line


def image_rotate_func(img, mask, angle, intersections, pad_=15):
    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    tmp_rot = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    coords = np.argwhere(tmp_rot > 0)
    if len(coords) == 0: return None

    y1, x1 = np.min(coords, axis=0)
    y2, x2 = np.max(coords, axis=0)

    M_shift = np.array([[1, 0, -x1 + pad_], [0, 1, -y1 + pad_], [0, 0, 1]])
    M_total = np.matmul(M_shift, np.vstack([M_rot, [0, 0, 1]]))

    new_w, new_h = x2 - x1 + pad_ * 2, y2 - y1 + pad_ * 2
    rotated_img = cv2.warpAffine(img, M_total[:2], (int(new_w), int(new_h)), flags=cv2.INTER_LINEAR)
    rotated_mask = cv2.warpAffine(mask, M_total[:2], (int(new_w), int(new_h)), flags=cv2.INTER_NEAREST)

    new_pts = [np.matmul(M_total, [p[0], p[1], 1])[:2] for p in intersections]
    return rotated_img, rotated_mask, M_total, new_pts


def execute_segmentation(img_obj, mask_obj):
    h_now, w_now = img_obj.shape[:2]
    verts = [(0, 0), (w_now - 1, 0), (w_now - 1, h_now - 1), (0, h_now - 1)]

    for _ in range(30):
        circ = rectangle_circumcircle(verts)
        if len(circ[6]) == 2:
            intersections, angle = circ[6], circ[7]
            break
    else:
        return False

    res_rot = image_rotate_func(img_obj, mask_obj, -angle, intersections)
    if not res_rot: return False
    img_rot, mask_rot, M_trans, rot_pts = res_rot

    basic_idx = np.argwhere(mask_rot > 0)
    if len(basic_idx) < 100: return False

    res_cut = CircleCutBase.generate_cut_line(basic_idx, img_rot.shape, rot_pts)
    mask_p_idx, mask_n_idx, a_p, a_n, paint_line = res_cut

    def crop_frag(indices):
        coords = basic_idx[indices]
        if len(coords) < 10: return None, None, None
        y1, x1 = np.min(coords, axis=0)
        y2, x2 = np.max(coords, axis=0)
        m_local = np.zeros(mask_rot.shape, dtype=np.uint8)
        m_local[coords[:, 0], coords[:, 1]] = 255

        # ========== 关键修改: 直接裁剪,不要bitwise_and ==========
        cropped_img = img_rot[y1:y2 + 1, x1:x2 + 1].copy()  # 保留原始RGB
        cropped_mask = m_local[y1:y2 + 1, x1:x2 + 1]
        # final_img = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)  # 删除这行
        # =======================================================

        M_crop = np.array([[1, 0, -x1], [0, 1, -y1], [0, 0, 1]], dtype=float)
        return cropped_img, cropped_mask, M_crop  # 直接返回原始裁剪

    i_p, m_p, mat_p = crop_frag(mask_p_idx)
    i_n, m_n, mat_n = crop_frag(mask_n_idx)

    if i_p is None or i_n is None: return False
    return (i_p, m_p, a_p, mat_p), (i_n, m_n, a_n, mat_n), M_trans, paint_line


def process_single_image(image_path, save_root):
    img_name = os.path.basename(image_path).split('.')[0]
    curr_save_dir = os.path.join(save_root, img_name)
    os.makedirs(curr_save_dir, exist_ok=True)

    origin_img = cv2.imread(image_path)
    if origin_img is None: return
    h, w = origin_img.shape[:2]
    origin_mask = np.ones((h, w), dtype=np.uint8) * 255

    # 【改动】全景可视化图
    full_vis = origin_img.copy()
    fragment_pool = [Fragment(origin_img, origin_mask, np.eye(3), h * w, 0)]
    final_fragments = []
    adj_pairs = []

    split_count = 0
    id_counter = 1

    while fragment_pool and split_count < MAX_RECURSION:
        fragment_pool.sort(key=lambda x: x.area, reverse=True)
        curr = fragment_pool.pop(0)

        success = False
        for _ in range(30):
            res = execute_segmentation(curr.img, curr.mask)
            if res:
                (i_p, m_p, a_p, mat_p), (i_n, m_n, a_n, mat_n), M_step, paint_line = res
                m_total_p = np.matmul(mat_p, np.matmul(M_step, curr.trans))
                m_total_n = np.matmul(mat_n, np.matmul(M_step, curr.trans))

                # 【改动】绘制白色切割线 (255, 255, 255)
                try:
                    m_to_orig = np.linalg.inv(np.matmul(M_step, curr.trans))
                    for pt in paint_line:
                        orig_pt = np.matmul(m_to_orig, [pt[0], pt[1], 1])
                        ox, oy = int(orig_pt[0]), int(orig_pt[1])
                        if 0 <= ox < w and 0 <= oy < h:
                            # 使用白色绘制，粗度为 1 像素
                            cv2.circle(full_vis, (ox, oy), 1, (255, 255, 255), -1)
                except:
                    pass

                id_p, id_n = id_counter, id_counter + 1
                adj_pairs.append((id_p, id_n))
                id_counter += 2

                fragment_pool.append(Fragment(i_p, m_p, m_total_p, a_p, id_p))
                fragment_pool.append(Fragment(i_n, m_n, m_total_n, a_n, id_n))
                split_count += 1
                success = True
                break

        if not success:
            final_fragments.append(curr)

    final_fragments.extend(fragment_pool)

    # 保存结果
    cv2.imwrite(os.path.join(curr_save_dir, 'full_segmentation_map.png'), full_vis)
    for f in final_fragments:
        idx_str = f"{f.frag_id:03d}"
        cv2.imwrite(os.path.join(curr_save_dir, f"frag_{idx_str}.png"), f.img)
        cv2.imwrite(os.path.join(curr_save_dir, f"frag_{idx_str}_mask.png"), f.mask)

        # 提取轮廓点并保存
        contours, _ = cv2.findContours(f.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=len).reshape(-1, 2)
            np.save(os.path.join(curr_save_dir, f"frag_{idx_str}_contour.npy"), c)

        with open(os.path.join(curr_save_dir, 'gt.txt'), 'a') as txt_f:
            txt_f.write(f"ID:{f.frag_id} Matrix:{f.trans.flatten().tolist()}\n")

    with open(os.path.join(curr_save_dir, 'adj_pairs.json'), 'w') as jf:
        json.dump(adj_pairs, jf)

    print(f"[{img_name}] 完成，白色线已保存至全景图。")


if __name__ == '__main__':
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.png', '.jpg'))]
    for f in files:
        process_single_image(os.path.join(INPUT_DIR, f), OUTPUT_DIR)