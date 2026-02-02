#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

# ===================== 标定参数（把 T 改为“米”） =====================
left_K = np.array([[370.543287793120, 0.547474330923898, 347.877982470338],
                   [0, 370.511406580083, 206.480879127774],
                   [0.0, 0.0, 1.0]])
left_D = np.array([[0.274216901317054, 0.716760612535803,
                    0.00620279794589385, 0.00865191685801085]])

right_K = np.array([[370.684508469580, -0.491421540915075, 340.693036356299],
                    [0, 370.885956946467, 206.450283518425],
                    [0.0, 0.0, 1.0]])
right_D = np.array([[0.279092474805592, 0.726773271987598,
                     0.00841447715571575, -0.00600743897282762]])

R = np.array([[0.999820522953803, 0.00138976707411422, -0.0188941903204100],
              [-0.00138775755844142, 0.999999029927790, 0.000119467311778264],
              [0.0188943380234174, -9.32253147102481e-05, 0.999821481715460]])

T_mm = np.array([[-96.0797377373969],
                 [  -0.152395283709690],
                 [   1.11912471887180]])
T_m  = T_mm / 1000.0


# ===================== 双目/3D 工具函数 =====================
def build_rectify_maps(frame_size):
    W, H = frame_size
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        left_K, left_D, right_K, right_D, (W, H), R, T_m,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    lmap1, lmap2 = cv2.initUndistortRectifyMap(left_K, left_D, R1, P1, (W, H), cv2.CV_16SC2)
    rmap1, rmap2 = cv2.initUndistortRectifyMap(right_K, right_D, R2, P2, (W, H), cv2.CV_16SC2)
    return lmap1, lmap2, rmap1, rmap2, Q

def disparity_sgbm(grayL, grayR):
    blockSize = 6
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16 * 16, blockSize=blockSize,
        P1=8 * 3 * blockSize**2, P2=32 * 3 * blockSize**2,
        disp12MaxDiff=-1, preFilterCap=1, uniquenessRatio=5,
        speckleWindowSize=60, speckleRange=16,
        mode=cv2.STEREO_SGBM_MODE_SGBM
    )
    return stereo.compute(grayL, grayR)

def reproject_to_3d(disparity_raw, Q):
    disp = disparity_raw.astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan
    # OpenCV不同版本函数名可能不同
    if hasattr(cv2, "reprojectImageTo3D"):
        return cv2.reprojectImageTo3D(disp, Q)
    return cv2.reprojectImageTo_3D(disp, Q)

def bilinear_interpolate_3d(pc, u, v):
    H, W, _ = pc.shape
    if u < 0 or v < 0 or u >= W - 1 or v >= H - 1:
        ui, vi = int(round(u)), int(round(v))
        if 0 <= vi < H and 0 <= ui < W:
            p = pc[vi, ui]
            if not (np.any(np.isnan(p)) or np.isinf(p).any()):
                return p
        return None

    x0, y0 = int(np.floor(u)), int(np.floor(v))
    x1, y1 = x0 + 1, y0 + 1
    Q11 = pc[y0, x0].astype(np.float64)
    Q21 = pc[y0, x1].astype(np.float64)
    Q12 = pc[y1, x0].astype(np.float64)
    Q22 = pc[y1, x1].astype(np.float64)

    if (np.any(np.isnan(Q11)) or np.any(np.isnan(Q21)) or
        np.any(np.isnan(Q12)) or np.any(np.isnan(Q22))):
        win = []
        for yy in range(max(0, y0 - 1), min(H, y0 + 2)):
            for xx in range(max(0, x0 - 1), min(W, x0 + 2)):
                p = pc[yy, xx]
                if not (np.any(np.isnan(p)) or np.isinf(p).any()):
                    win.append(p)
        if not win:
            return None
        return np.median(np.array(win), axis=0)

    dx, dy = u - x0, v - y0
    top = Q11 * (1 - dx) + Q21 * dx
    bottom = Q12 * (1 - dx) + Q22 * dx
    return top * (1 - dy) + bottom * dy

def distance_from_kpts(pc3d, kpt1, kpt2):
    p1 = bilinear_interpolate_3d(pc3d, float(kpt1[0]), float(kpt1[1]))
    p2 = bilinear_interpolate_3d(pc3d, float(kpt2[0]), float(kpt2[1]))
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(p1 - p2))


# ===================== YOLO positive 类筛选 =====================
def get_positive_indices(result, model):
    r = result
    names = r.names if hasattr(r, "names") else getattr(model.model, "names", None)
    pos_id = None
    if isinstance(names, dict):
        for k, v in names.items():
            if str(v).lower() == "positive":
                pos_id = int(k)
                break
    if pos_id is None:
        pos_id = 0  # 兜底：默认0为positive
    cls_np = r.boxes.cls.cpu().numpy().astype(int)
    return np.where(cls_np == pos_id)[0], pos_id


# ===================== 体重公式（按你的拟合） =====================
def calculate_weight(L):
    return 330.8762 + (-27.7648) * L + 1.211230 * L**2


# ===================== MAD 稳健过滤 =====================
def robust_filter_mad(x, mad_k=3.5):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return x
    z = 0.6745 * (x - med) / mad
    return x[np.abs(z) <= mad_k]


# ===================== 简易跟踪：IoU =====================
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

class Track:
    def __init__(self, tid, bbox, frame_idx):
        self.tid = tid
        self.bbox = bbox
        self.last_frame = frame_idx
        self.window_weights = []

def match_tracks(tracks, det_bboxes, frame_idx, iou_th=0.3, max_age=10):
    dead = [tid for tid, trk in tracks.items() if frame_idx - trk.last_frame > max_age]
    for tid in dead:
        del tracks[tid]

    tids = list(tracks.keys())
    assigned = [None] * len(det_bboxes)
    if len(tids) == 0 or len(det_bboxes) == 0:
        return assigned

    used_tracks = set()
    for i, db in enumerate(det_bboxes):
        best_tid = None
        best_iou = 0.0
        for tid in tids:
            if tid in used_tracks:
                continue
            iouv = iou_xyxy(tracks[tid].bbox, db)
            if iouv > best_iou:
                best_iou = iouv
                best_tid = tid
        if best_tid is not None and best_iou >= iou_th:
            assigned[i] = best_tid
            used_tracks.add(best_tid)
            tracks[best_tid].bbox = db
            tracks[best_tid].last_frame = frame_idx
    return assigned


# ===================== 主入口：估重（返回 dict） =====================
def estimate_average_weight(
    left_video: str,
    right_video: str,
    model,
    conf: float = 0.25,
    imgsz: int = 640,
    frame_stride: int = 5,     # 5min视频建议 3~6，默认5
    window_sec: float = 0.5,
    iou_th: float = 0.3,
    max_age: int = 10,
    MIN_G: float = 50.0,
    MAX_G: float = 3000.0,
    use_cm: bool = True,       # 你拟合L大概率是 cm（列名/经验），默认 True
    mad_k: float = 3.5,
):
    capL = cv2.VideoCapture(left_video)
    capR = cv2.VideoCapture(right_video)
    if not capL.isOpened() or not capR.isOpened():
        raise RuntimeError(f"Cannot open videos: {left_video}, {right_video}")

    W = int(capL.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(capL.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W2 = int(capR.get(cv2.CAP_PROP_FRAME_WIDTH))
    H2 = int(capR.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (W, H) != (W2, H2):
        raise RuntimeError(f"Resolution mismatch: L={W}x{H}, R={W2}x{H2}")

    fps = capL.get(cv2.CAP_PROP_FPS)
    if fps <= 1e-6:
        fps = 25.0
    total = int(capL.get(cv2.CAP_PROP_FRAME_COUNT))

    lmap1, lmap2, rmap1, rmap2, Q = build_rectify_maps((W, H))
    window_frames = max(1, int(round(window_sec * fps)))

    tracks = {}
    next_tid = 1
    aggregated_weights = []
    raw_valid_weights_count = 0

    frame_idx = 0
    window_start_frame = 0

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or not retR:
            break

        # 抽帧：只处理每 frame_stride 帧
        if frame_stride > 1 and (frame_idx % frame_stride != 0):
            frame_idx += 1
            continue

        left_rect  = cv2.remap(frameL, lmap1, lmap2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(frameR, rmap1, rmap2, cv2.INTER_LINEAR)

        grayL = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        disp_raw = disparity_sgbm(grayL, grayR)
        pc3d = reproject_to_3d(disp_raw, Q)

        res = model.predict(source=left_rect, conf=conf, imgsz=imgsz, verbose=False)
        r = res[0]

        pos_idx = np.array([], dtype=int)
        if len(r.boxes) > 0:
            pos_idx, _ = get_positive_indices(r, model)

        det_bboxes = []
        det_weights = []

        if pos_idx.size > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            for idx in pos_idx:
                x1, y1, x2, y2 = boxes_xyxy[idx].tolist()
                det_bboxes.append((x1, y1, x2, y2))

                # 关键点两点
                try:
                    kpts_xy = r.keypoints.xy[idx].cpu().numpy()
                except Exception:
                    det_weights.append(None)
                    continue
                if kpts_xy.shape[0] < 2:
                    det_weights.append(None)
                    continue

                (u1, v1) = kpts_xy[0]
                (u2, v2) = kpts_xy[1]
                dist_m = distance_from_kpts(pc3d, (u1, v1), (u2, v2))
                if dist_m is None or not np.isfinite(dist_m):
                    det_weights.append(None)
                    continue

                L = dist_m * 100.0 if use_cm else dist_m
                w = calculate_weight(L)
                if (not np.isfinite(w)) or (w < MIN_G) or (w > MAX_G):
                    det_weights.append(None)
                    continue

                det_weights.append(float(w))
                raw_valid_weights_count += 1

        assigned = match_tracks(tracks, det_bboxes, frame_idx, iou_th=iou_th, max_age=max_age)

        for i, bbox in enumerate(det_bboxes):
            tid = assigned[i]
            if tid is None:
                tid = next_tid
                next_tid += 1
                tracks[tid] = Track(tid, bbox, frame_idx)

            w = det_weights[i] if i < len(det_weights) else None
            if w is not None:
                tracks[tid].window_weights.append(w)

        # 窗结束：对每个track做窗内中位数聚合
        if (frame_idx - window_start_frame + 1) >= window_frames:
            for tid, trk in tracks.items():
                if len(trk.window_weights) > 0:
                    aggregated_weights.append(float(np.median(trk.window_weights)))
                trk.window_weights = []
            window_start_frame = frame_idx + 1

        frame_idx += 1

    # 收尾：最后不足一窗也聚合
    for tid, trk in tracks.items():
        if len(trk.window_weights) > 0:
            aggregated_weights.append(float(np.median(trk.window_weights)))
        trk.window_weights = []

    capL.release()
    capR.release()

    if len(aggregated_weights) == 0:
        return {
            "ok": False,
            "message": "No valid samples (after range filter).",
            "video_frames": total,
            "fps": float(fps),
            "raw_valid_weights_count": int(raw_valid_weights_count),
            "aggregated_count": 0,
            "mad_filtered_count": 0,
        }

    agg = np.array(aggregated_weights, dtype=np.float64)
    filt = robust_filter_mad(agg, mad_k=mad_k)

    if filt.size == 0:
        return {
            "ok": False,
            "message": "MAD filtered empty (too strict mad_k or unstable depth).",
            "video_frames": total,
            "fps": float(fps),
            "raw_valid_weights_count": int(raw_valid_weights_count),
            "aggregated_count": int(len(agg)),
            "mad_filtered_count": 0,
            "agg_mean_g": float(np.mean(agg)),
            "agg_median_g": float(np.median(agg)),
        }

    return {
        "ok": True,
        "message": "success",
        "video_frames": total,
        "fps": float(fps),
        "frame_stride": int(frame_stride),
        "window_sec": float(window_sec),
        "raw_valid_weights_count": int(raw_valid_weights_count),
        "aggregated_count": int(len(agg)),
        "mad_filtered_count": int(len(filt)),
        "agg_mean_g": float(np.mean(agg)),
        "agg_median_g": float(np.median(agg)),
        "avg_weight_g": float(np.mean(filt)),
        "median_weight_g": float(np.median(filt)),
        "min_weight_g": float(np.min(filt)),
        "max_weight_g": float(np.max(filt)),
    }
