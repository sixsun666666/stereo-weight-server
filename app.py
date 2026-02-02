#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastapi.responses import FileResponse
import os
import uuid
import time
import shutil
import traceback
from typing import Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from concurrent.futures import ThreadPoolExecutor
import threading

from ultralytics import YOLO
from estimator import estimate_average_weight

# ===================== 你的权重路径（服务器本地） =====================
MODEL_PATH = "/home/wangyabo/work/results/runs/pose/train41/weights/best.pt"

# ===================== 服务配置 =====================
UPLOAD_DIR = "/tmp/stereo_weight_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 5分钟视频建议：抽帧 5（你可改 3~6）
DEFAULT_FRAME_STRIDE = 5

# 单GPU并发锁（最重要：避免CUDA OOM）
GPU_SEM = threading.Semaphore(1)

# 异步任务池（线程）
EXEC = ThreadPoolExecutor(max_workers=2)

# 任务状态
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Stereo Weight Estimation Server", version="1.0")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

@app.get("/")
def home():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


# 允许跨域（本地网页直接访问服务器接口时需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境你可以改成指定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 模型只加载一次
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model at {MODEL_PATH}: {e}")

def _save_upload(upload: UploadFile, out_path: str):
    with open(out_path, "wb") as f:
        shutil.copyfileobj(upload.file, f)

def _cleanup_files(*paths: str):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

def _run_job(job_id: str, left_path: str, right_path: str,
             conf: float, imgsz: int, frame_stride: int,
             window_sec: float, min_g: float, max_g: float,
             use_cm: bool, mad_k: float):
    start = time.time()
    JOBS[job_id]["status"] = "running"
    JOBS[job_id]["started_at"] = start

    try:
        # 单GPU并发锁
        with GPU_SEM:
            result = estimate_average_weight(
                left_video=left_path,
                right_video=right_path,
                model=model,
                conf=conf,
                imgsz=imgsz,
                frame_stride=frame_stride,
                window_sec=window_sec,
                MIN_G=min_g,
                MAX_G=max_g,
                use_cm=use_cm,
                mad_k=mad_k,
            )

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
        JOBS[job_id]["finished_at"] = time.time()
        JOBS[job_id]["elapsed_sec"] = round(JOBS[job_id]["finished_at"] - start, 3)

    except Exception as e:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["trace"] = traceback.format_exc()
        JOBS[job_id]["finished_at"] = time.time()
        JOBS[job_id]["elapsed_sec"] = round(JOBS[job_id]["finished_at"] - start, 3)

    finally:
        # 清理上传文件，避免 /tmp 爆掉
        _cleanup_files(left_path, right_path)


@app.post("/jobs")
async def create_job(
    left_video: UploadFile = File(...),
    right_video: UploadFile = File(...),

    # 可调参数（给默认值）
    conf: float = 0.25,
    imgsz: int = 640,
    frame_stride: int = DEFAULT_FRAME_STRIDE,   # 5分钟视频默认5
    window_sec: float = 0.5,
    min_g: float = 50.0,
    max_g: float = 3000.0,
    use_cm: bool = True,
    mad_k: float = 3.5,
):
    # 基本校验
    if frame_stride < 1 or frame_stride > 30:
        raise HTTPException(status_code=400, detail="frame_stride must be in [1, 30]")
    if window_sec <= 0 or window_sec > 5:
        raise HTTPException(status_code=400, detail="window_sec must be in (0, 5]")
    if min_g >= max_g:
        raise HTTPException(status_code=400, detail="min_g must be < max_g")

    job_id = uuid.uuid4().hex

    left_path = os.path.join(UPLOAD_DIR, f"{job_id}_left_{os.path.basename(left_video.filename)}")
    right_path = os.path.join(UPLOAD_DIR, f"{job_id}_right_{os.path.basename(right_video.filename)}")

    try:
        _save_upload(left_video, left_path)
        _save_upload(right_video, right_path)
    except Exception as e:
        _cleanup_files(left_path, right_path)
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {e}")

    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": time.time(),
        "params": {
            "conf": conf, "imgsz": imgsz,
            "frame_stride": frame_stride,
            "window_sec": window_sec,
            "min_g": min_g, "max_g": max_g,
            "use_cm": use_cm, "mad_k": mad_k,
        },
    }

    # 异步提交任务
    EXEC.submit(_run_job, job_id, left_path, right_path,
                conf, imgsz, frame_stride, window_sec,
                min_g, max_g, use_cm, mad_k)

    return {"job_id": job_id, "status": "queued"}

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_id not found")
    # failed 的 trace 默认也返回（方便你调试）；若上线可去掉 trace
    return job

@app.post("/estimate")
async def estimate_sync(
    left_video: UploadFile = File(...),
    right_video: UploadFile = File(...),

    conf: float = 0.25,
    imgsz: int = 640,
    frame_stride: int = DEFAULT_FRAME_STRIDE,
    window_sec: float = 0.5,
    min_g: float = 50.0,
    max_g: float = 3000.0,
    use_cm: bool = True,
    mad_k: float = 3.5,
):
    """
    同步接口：不建议用于5分钟视频（可能超时），但调试很方便
    """
    job_id = uuid.uuid4().hex
    left_path = os.path.join(UPLOAD_DIR, f"{job_id}_left_{os.path.basename(left_video.filename)}")
    right_path = os.path.join(UPLOAD_DIR, f"{job_id}_right_{os.path.basename(right_video.filename)}")

    try:
        _save_upload(left_video, left_path)
        _save_upload(right_video, right_path)

        # 单GPU锁：同步也保护
        with GPU_SEM:
            result = estimate_average_weight(
                left_video=left_path,
                right_video=right_path,
                model=model,
                conf=conf,
                imgsz=imgsz,
                frame_stride=frame_stride,
                window_sec=window_sec,
                MIN_G=min_g,
                MAX_G=max_g,
                use_cm=use_cm,
                mad_k=mad_k,
            )
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status_code=500)

    finally:
        _cleanup_files(left_path, right_path)
