#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from fastapi.responses import FileResponse
import os
import uuid
import time
import shutil
import traceback
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from concurrent.futures import ThreadPoolExecutor
import threading

from ultralytics import YOLO
from estimator import estimate_average_weight

import database as db
import auth
from auth import get_current_user, get_current_admin
# #region agent log
try:
    open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:imports","message":"all imports done","hypothesisId":"H1","timestamp":' + str(__import__("time").time()) + '}\n')
except Exception:
    pass
# #endregion

# ===================== 你的权重路径（服务器本地） =====================
MODEL_PATH = "/home/wangyabo/work/results/runs/pose/train41/weights/best.pt"

# ===================== 服务配置 =====================
UPLOAD_DIR = "/tmp/stereo_weight_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

DEFAULT_FRAME_STRIDE = 5
GPU_SEM = threading.Semaphore(1)
EXEC = ThreadPoolExecutor(max_workers=2)
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title="Stereo Weight Estimation Server", version="2.0")
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")


@app.on_event("startup")
def startup():
    # #region agent log
    try:
        open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:startup","message":"startup entry","hypothesisId":"H4","timestamp":' + str(__import__("time").time()) + '}\n')
    except Exception:
        pass
    # #endregion
    db.init_db()
    # #region agent log
    try:
        open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:startup","message":"init_db done","hypothesisId":"H4","timestamp":' + str(__import__("time").time()) + '}\n')
    except Exception:
        pass
    # #endregion
    if not db.user_exists("admin"):
        # #region agent log
        try:
            open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:startup","message":"create_user admin before","hypothesisId":"H4","timestamp":' + str(__import__("time").time()) + '}\n')
        except Exception:
            pass
        # #endregion
        db.create_user("admin", auth.hash_password("admin123"), "admin")
        # #region agent log
        try:
            open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:startup","message":"create_user admin done","hypothesisId":"H4","timestamp":' + str(__import__("time").time()) + '}\n')
        except Exception:
            pass
        # #endregion


# 静态页：/ 登录，/app 估重，/history 历史
@app.get("/")
def home():
    return FileResponse(os.path.join(WEB_DIR, "login.html"))


@app.get("/app")
def app_page():
    return FileResponse(os.path.join(WEB_DIR, "index.html"))


@app.get("/history")
def history_page():
    return FileResponse(os.path.join(WEB_DIR, "history.html"))


@app.get("/pools-page")
def pools_page():
    return FileResponse(os.path.join(WEB_DIR, "pools.html"))


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# #region agent log
try:
    open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:yolo","message":"before YOLO load","hypothesisId":"H2","timestamp":' + str(__import__("time").time()) + '}\n')
except Exception:
    pass
# #endregion
try:
    model = YOLO(MODEL_PATH)
    # #region agent log
    try:
        open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:yolo","message":"YOLO load ok","hypothesisId":"H2","timestamp":' + str(__import__("time").time()) + '}\n')
    except Exception:
        pass
    # #endregion
except Exception as e:
    # #region agent log
    try:
        open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"app.py:yolo","message":"YOLO load failed","data":{"err":str(e)},"hypothesisId":"H2","timestamp":' + str(__import__("time").time()) + '}\n')
    except Exception:
        pass
    # #endregion
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
    db.update_job_record(job_id, "running", started_at=start)

    try:
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
        finished = time.time()
        elapsed = round(finished - start, 3)
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["result"] = result
        JOBS[job_id]["finished_at"] = finished
        JOBS[job_id]["elapsed_sec"] = elapsed
        db.update_job_record(job_id, "done", result=result, finished_at=finished, elapsed_sec=elapsed)
    except Exception as e:
        finished = time.time()
        elapsed = round(finished - start, 3)
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["trace"] = traceback.format_exc()
        JOBS[job_id]["finished_at"] = finished
        JOBS[job_id]["elapsed_sec"] = elapsed
        db.update_job_record(job_id, "failed", finished_at=finished, elapsed_sec=elapsed, error=str(e))
    finally:
        _cleanup_files(left_path, right_path)


# ===================== 登录与注册 =====================
@app.post("/auth/login")
def login(username: str = Form(...), password: str = Form(...)):
    user = db.get_user_by_username(username)
    if not user or not auth.verify_password(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    token = auth.create_access_token({
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"] or "user",
    })
    return {"access_token": token, "role": user["role"] or "user", "username": user["username"]}


@app.post("/auth/register")
def register(username: str = Form(...), password: str = Form(...)):
    username = username.strip()
    if not username or len(username) < 2:
        raise HTTPException(status_code=400, detail="用户名至少 2 个字符")
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="密码至少 6 位")
    if db.user_exists(username):
        raise HTTPException(status_code=400, detail="用户名已存在")
    db.create_user(username, auth.hash_password(password), "user")
    return JSONResponse({"message": "注册成功"}, status_code=201)


# ===================== 估重（需登录，带 pool_name） =====================
@app.post("/jobs")
async def create_job(
    request_user: dict = Depends(get_current_user),
    left_video: UploadFile = File(...),
    right_video: UploadFile = File(...),
    pool_name: str = Form(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
    frame_stride: int = Form(DEFAULT_FRAME_STRIDE),
    window_sec: float = Form(0.5),
    min_g: float = Form(50.0),
    max_g: float = Form(3000.0),
    use_cm: bool = Form(True),
    mad_k: float = Form(3.5),
):
    if frame_stride < 1 or frame_stride > 30:
        raise HTTPException(status_code=400, detail="frame_stride must be in [1, 30]")
    if window_sec <= 0 or window_sec > 5:
        raise HTTPException(status_code=400, detail="window_sec must be in (0, 5]")
    if min_g >= max_g:
        raise HTTPException(status_code=400, detail="min_g must be < max_g")
    if not (pool_name and pool_name.strip()):
        raise HTTPException(status_code=400, detail="请选择鱼池（pool_name）")

    job_id = uuid.uuid4().hex
    left_fn = os.path.basename(left_video.filename or "")
    right_fn = os.path.basename(right_video.filename or "")
    left_path = os.path.join(UPLOAD_DIR, f"{job_id}_left_{left_fn}")
    right_path = os.path.join(UPLOAD_DIR, f"{job_id}_right_{right_fn}")

    try:
        _save_upload(left_video, left_path)
        _save_upload(right_video, right_path)
    except Exception as e:
        _cleanup_files(left_path, right_path)
        raise HTTPException(status_code=500, detail=f"Failed to save uploads: {e}")

    params = {
        "conf": conf, "imgsz": imgsz,
        "frame_stride": frame_stride, "window_sec": window_sec,
        "min_g": min_g, "max_g": max_g, "use_cm": use_cm, "mad_k": mad_k,
    }
    db.create_job_record(job_id, request_user["id"], pool_name.strip(), left_fn, right_fn, params)

    JOBS[job_id] = {
        "job_id": job_id,
        "user_id": request_user["id"],
        "username": request_user["username"],
        "pool_name": pool_name.strip(),
        "status": "queued",
        "created_at": time.time(),
        "params": params,
    }

    EXEC.submit(_run_job, job_id, left_path, right_path,
                conf, imgsz, frame_stride, window_sec,
                min_g, max_g, use_cm, mad_k)
    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str, request_user: dict = Depends(get_current_user)):
    job = JOBS.get(job_id)
    if not job:
        job = db.get_job_by_id(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job_id not found")
        if request_user["role"] != "admin" and job.get("user_id") != request_user["id"]:
            raise HTTPException(status_code=403, detail="无权查看该任务")
    else:
        if request_user["role"] != "admin" and job.get("user_id") != request_user["id"]:
            raise HTTPException(status_code=403, detail="无权查看该任务")
    return job


@app.post("/estimate")
async def estimate_sync(
    request_user: dict = Depends(get_current_user),
    left_video: UploadFile = File(...),
    right_video: UploadFile = File(...),
    pool_name: str = Form(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
    frame_stride: int = Form(DEFAULT_FRAME_STRIDE),
    window_sec: float = Form(0.5),
    min_g: float = Form(50.0),
    max_g: float = Form(3000.0),
    use_cm: bool = Form(True),
    mad_k: float = Form(3.5),
):
    if not (pool_name and pool_name.strip()):
        raise HTTPException(status_code=400, detail="请选择鱼池（pool_name）")
    job_id = uuid.uuid4().hex
    left_fn = os.path.basename(left_video.filename or "")
    right_fn = os.path.basename(right_video.filename or "")
    left_path = os.path.join(UPLOAD_DIR, f"{job_id}_left_{left_fn}")
    right_path = os.path.join(UPLOAD_DIR, f"{job_id}_right_{right_fn}")

    try:
        _save_upload(left_video, left_path)
        _save_upload(right_video, right_path)
        params = {"conf": conf, "imgsz": imgsz, "frame_stride": frame_stride, "window_sec": window_sec,
                  "min_g": min_g, "max_g": max_g, "use_cm": use_cm, "mad_k": mad_k}
        db.create_job_record(job_id, request_user["id"], pool_name.strip(), left_fn, right_fn, params)
        start_t = time.time()

        with GPU_SEM:
            result = estimate_average_weight(
                left_video=left_path, right_video=right_path, model=model,
                conf=conf, imgsz=imgsz, frame_stride=frame_stride, window_sec=window_sec,
                MIN_G=min_g, MAX_G=max_g, use_cm=use_cm, mad_k=mad_k,
            )
        finished = time.time()
        elapsed = round(finished - start_t, 3)
        db.update_job_record(job_id, "done", result=result, started_at=start_t, finished_at=finished, elapsed_sec=elapsed)
        return JSONResponse(result)
    except Exception as e:
        db.update_job_record(job_id, "failed", error=str(e))
        return JSONResponse({"ok": False, "error": str(e), "trace": traceback.format_exc()}, status_code=500)
    finally:
        _cleanup_files(left_path, right_path)


def _parse_date(s: Optional[str]) -> Optional[float]:
    if not s or not s.strip():
        return None
    s = s.strip()[:10]
    try:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def _parse_end_date(s: Optional[str]) -> Optional[float]:
    if not s or not s.strip():
        return None
    s = s.strip()[:10]
    try:
        dt = datetime.strptime(s, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


# ===================== 历史与图表（需登录） =====================
@app.get("/history/list")
def history_list(
    request_user: dict = Depends(get_current_user),
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    pool_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
):
    uid = request_user["id"] if request_user["role"] != "admin" else user_id
    ts_start = _parse_date(start_date)
    ts_end = _parse_end_date(end_date)
    items, total = db.list_jobs(user_id=uid, status=status, pool_name=pool_name or None,
                                 start_date=ts_start, end_date=ts_end, page=page, limit=limit)
    return {"items": items, "total": total}


@app.get("/history/chart")
def history_chart(
    request_user: dict = Depends(get_current_user),
    user_id: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    uid = request_user["id"] if request_user["role"] != "admin" else user_id
    ts_start = _parse_date(start_date)
    ts_end = _parse_end_date(end_date)
    data = db.get_chart_data(user_id=uid, start_date=ts_start, end_date=ts_end)
    return data


# ===================== 鱼池 CRUD（需登录，仅当前用户） =====================
@app.get("/pools")
def pools_list(request_user: dict = Depends(get_current_user)):
    items = db.list_pools(request_user["id"])
    return items


@app.post("/pools")
def pools_create(request_user: dict = Depends(get_current_user), name: str = Form(...)):
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="鱼池名称不能为空")
    existing = db.list_pools(request_user["id"])
    if any(p["name"] == name for p in existing):
        raise HTTPException(status_code=400, detail="该鱼池名称已存在")
    pid = db.create_pool(request_user["id"], name)
    pool = db.get_pool(pid)
    return JSONResponse(pool, status_code=201)


@app.patch("/pools/{pool_id}")
def pools_update(pool_id: int, request_user: dict = Depends(get_current_user), name: str = Form(...)):
    pool = db.get_pool(pool_id)
    if not pool or pool["user_id"] != request_user["id"]:
        raise HTTPException(status_code=404, detail="鱼池不存在或无权操作")
    name = name.strip()
    if not name:
        raise HTTPException(status_code=400, detail="鱼池名称不能为空")
    existing = db.list_pools(request_user["id"])
    if any(p["name"] == name and p["id"] != pool_id for p in existing):
        raise HTTPException(status_code=400, detail="该鱼池名称已存在")
    if not db.update_pool(pool_id, request_user["id"], name):
        raise HTTPException(status_code=500, detail="更新失败")
    return db.get_pool(pool_id)


@app.delete("/pools/{pool_id}")
def pools_delete(pool_id: int, request_user: dict = Depends(get_current_user)):
    ok, err = db.delete_pool(pool_id, request_user["id"])
    if not ok:
        raise HTTPException(status_code=400, detail=err or "删除失败")
    return {"message": "已删除"}


# ===================== 清空历史（需登录） =====================
@app.delete("/history")
def history_delete(
    request_user: dict = Depends(get_current_user),
    user_id: Optional[int] = None,
    job_id: Optional[str] = None,
    pool_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
):
    uid = request_user["id"] if request_user["role"] != "admin" else user_id
    if uid is None and request_user["role"] == "admin":
        uid = user_id  # admin must pass user_id to target
    if uid is None:
        raise HTTPException(status_code=400, detail="请指定操作对象（管理员需传 user_id）")
    if not job_id and not pool_name and (start_date is None and end_date is None):
        raise HTTPException(status_code=400, detail="请指定删除条件：job_id、pool_name 或 start_date&end_date")
    ts_start = _parse_date(start_date) if start_date else None
    ts_end = _parse_end_date(end_date) if end_date else None
    n = db.delete_jobs(user_id=uid, job_id=job_id, pool_name=pool_name or None, start_date=ts_start, end_date=ts_end)
    if job_id and job_id in JOBS:
        del JOBS[job_id]
    return {"deleted": n}
