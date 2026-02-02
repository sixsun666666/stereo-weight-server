#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQLite 数据库：用户表、任务历史表（含 pool_name），初始化与 CRUD。"""

import os
import sqlite3
import json
from typing import Any, Dict, List, Optional, Tuple
from contextlib import contextmanager

# 数据库文件放在项目目录
DB_PATH = os.path.join(os.path.dirname(__file__), "stereo_weight.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    """建表：users, job_history（含 pool_name）。"""
    # #region agent log
    try:
        open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"database.py:init_db","message":"init_db entry","hypothesisId":"H3","timestamp":' + str(__import__("time").time()) + '}\n')
    except Exception:
        pass
    # #endregion
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS job_history (
                job_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                pool_name TEXT NOT NULL,
                status TEXT NOT NULL,
                params TEXT,
                result TEXT,
                created_at REAL,
                started_at REAL,
                finished_at REAL,
                elapsed_sec REAL,
                left_filename TEXT,
                right_filename TEXT,
                error TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_job_history_user_id ON job_history(user_id);
            CREATE INDEX IF NOT EXISTS idx_job_history_created_at ON job_history(created_at);
            CREATE INDEX IF NOT EXISTS idx_job_history_pool_name ON job_history(pool_name);
            CREATE INDEX IF NOT EXISTS idx_job_history_status ON job_history(status);
            CREATE TABLE IF NOT EXISTS pools (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at REAL NOT NULL,
                UNIQUE(user_id, name),
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
            CREATE INDEX IF NOT EXISTS idx_pools_user_id ON pools(user_id);
        """)
    # #region agent log
    try:
        open("/home/wangyabo/work/stereo-weight-server/.cursor/debug.log", "a").write('{"location":"database.py:init_db","message":"init_db exit ok","hypothesisId":"H3","timestamp":' + str(__import__("time").time()) + '}\n')
    except Exception:
        pass
    # #endregion


def create_job_record(
    job_id: str,
    user_id: int,
    pool_name: str,
    left_filename: str = "",
    right_filename: str = "",
    params: Optional[Dict[str, Any]] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO job_history (job_id, user_id, pool_name, status, params, created_at, left_filename, right_filename)
               VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)""",
            (
                job_id,
                user_id,
                pool_name,
                json.dumps(params or {}, ensure_ascii=False),
                __now(),
                left_filename or "",
                right_filename or "",
            ),
        )


def update_job_record(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
    started_at: Optional[float] = None,
    finished_at: Optional[float] = None,
    elapsed_sec: Optional[float] = None,
    error: Optional[str] = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """UPDATE job_history SET status = ?, result = ?, started_at = ?, finished_at = ?, elapsed_sec = ?, error = ?
               WHERE job_id = ?""",
            (
                status,
                json.dumps(result, ensure_ascii=False) if result else None,
                started_at,
                finished_at,
                elapsed_sec,
                error,
                job_id,
            ),
        )


def get_job_by_id(job_id: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT j.*, u.username FROM job_history j LEFT JOIN users u ON j.user_id = u.id WHERE j.job_id = ?",
            (job_id,),
        ).fetchone()
    if not row:
        return None
    return _row_to_job_dict(row)


def list_jobs(
    user_id: Optional[int] = None,
    status: Optional[str] = None,
    pool_name: Optional[str] = None,
    start_date: Optional[float] = None,
    end_date: Optional[float] = None,
    page: int = 1,
    limit: int = 20,
) -> tuple[List[Dict[str, Any]], int]:
    """返回 (items, total)。user_id=None 且为 admin 时查全部。"""
    conditions = ["1=1"]
    args: list = []
    if user_id is not None:
        conditions.append("j.user_id = ?")
        args.append(user_id)
    if status:
        conditions.append("j.status = ?")
        args.append(status)
    if pool_name:
        conditions.append("j.pool_name = ?")
        args.append(pool_name)
    if start_date is not None:
        conditions.append("j.created_at >= ?")
        args.append(start_date)
    if end_date is not None:
        conditions.append("j.created_at <= ?")
        args.append(end_date)

    where = " AND ".join(conditions)
    count_sql = f"SELECT COUNT(*) FROM job_history j WHERE {where}"
    list_sql = f"""SELECT j.*, u.username FROM job_history j
                  LEFT JOIN users u ON j.user_id = u.id
                  WHERE {where} ORDER BY j.created_at DESC LIMIT ? OFFSET ?"""

    with get_conn() as conn:
        total = conn.execute(count_sql, args).fetchone()[0]
        args_ext = args + [limit, (page - 1) * limit]
        rows = conn.execute(list_sql, args_ext).fetchall()

    items = [_row_to_job_dict(r) for r in rows]
    return items, total


def get_chart_data(
    user_id: Optional[int] = None,
    start_date: Optional[float] = None,
    end_date: Optional[float] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """按鱼池分组，返回 { "鱼池1": [{ "date": "YYYY-MM-DD", "weight_g": float }, ...], ... }。仅 status=done 且 result 含 avg_weight_g。"""
    conditions = ["j.status = 'done'", "j.result IS NOT NULL"]
    args: list = []
    if user_id is not None:
        conditions.append("j.user_id = ?")
        args.append(user_id)
    if start_date is not None:
        conditions.append("j.created_at >= ?")
        args.append(start_date)
    if end_date is not None:
        conditions.append("j.created_at <= ?")
        args.append(end_date)
    where = " AND ".join(conditions)

    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT j.job_id, j.pool_name, j.created_at, j.result FROM job_history j WHERE {where} ORDER BY j.created_at""",
            args,
        ).fetchall()

    by_pool: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        pool = r["pool_name"] or "未分类"
        try:
            res = json.loads(r["result"]) if isinstance(r["result"], str) else r["result"]
            w = res.get("avg_weight_g")
        except Exception:
            w = None
        if w is None or not isinstance(w, (int, float)):
            continue
        from datetime import datetime
        dt = r["created_at"]
        if isinstance(dt, (int, float)):
            date_str = datetime.utcfromtimestamp(dt).strftime("%Y-%m-%d")
        else:
            date_str = str(dt)[:10]
        if pool not in by_pool:
            by_pool[pool] = []
        by_pool[pool].append({"date": date_str, "weight_g": round(float(w), 2)})

    return by_pool


def _row_to_job_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    if d.get("params"):
        try:
            d["params"] = json.loads(d["params"]) if isinstance(d["params"], str) else d["params"]
        except Exception:
            pass
    if d.get("result"):
        try:
            d["result"] = json.loads(d["result"]) if isinstance(d["result"], str) else d["result"]
        except Exception:
            pass
    return d


def list_pools(user_id: int) -> List[Dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, user_id, name, created_at FROM pools WHERE user_id = ? ORDER BY name",
            (user_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def create_pool(user_id: int, name: str) -> int:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO pools (user_id, name, created_at) VALUES (?, ?, ?)",
            (user_id, name.strip(), __now()),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def get_pool(pool_id: int) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT id, user_id, name, created_at FROM pools WHERE id = ?", (pool_id,)).fetchone()
    if not row:
        return None
    return dict(row)


def update_pool(pool_id: int, user_id: int, name: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute(
            "UPDATE pools SET name = ? WHERE id = ? AND user_id = ?",
            (name.strip(), pool_id, user_id),
        )
        return cur.rowcount > 0


def pool_has_history(user_id: int, pool_name: str) -> bool:
    with get_conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM job_history WHERE user_id = ? AND pool_name = ?",
            (user_id, pool_name),
        ).fetchone()[0]
    return n > 0


def delete_pool(pool_id: int, user_id: int) -> Tuple[bool, Optional[str]]:
    """Returns (success, error_message). Fails if pool has history."""
    pool = get_pool(pool_id)
    if not pool or pool["user_id"] != user_id:
        return False, "池不存在或无权操作"
    if pool_has_history(user_id, pool["name"]):
        return False, "该鱼池已有历史记录，无法删除"
    with get_conn() as conn:
        conn.execute("DELETE FROM pools WHERE id = ? AND user_id = ?", (pool_id, user_id))
    return True, None


def delete_jobs(
    user_id: int,
    job_id: Optional[str] = None,
    pool_name: Optional[str] = None,
    start_date: Optional[float] = None,
    end_date: Optional[float] = None,
) -> int:
    """Delete job_history by condition; return deleted count."""
    conditions = ["user_id = ?"]
    args: list = [user_id]
    if job_id:
        conditions.append("job_id = ?")
        args.append(job_id)
    if pool_name:
        conditions.append("pool_name = ?")
        args.append(pool_name)
    if start_date is not None:
        conditions.append("created_at >= ?")
        args.append(start_date)
    if end_date is not None:
        conditions.append("created_at <= ?")
        args.append(end_date)
    where = " AND ".join(conditions)
    with get_conn() as conn:
        cur = conn.execute(f"DELETE FROM job_history WHERE {where}", args)
        return cur.rowcount


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    with get_conn() as conn:
        row = conn.execute("SELECT id, username, password_hash, role, created_at FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return None
    return dict(row)


def create_user(username: str, password_hash: str, role: str = "user") -> int:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (username, password_hash, role, __now()),
        )
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def user_exists(username: str) -> bool:
    return get_user_by_username(username) is not None


def __now() -> float:
    import time
    return time.time()
