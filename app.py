# app.py
from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import os, bcrypt, jwt, secrets, hashlib, uuid
import httpx
from dotenv import load_dotenv

load_dotenv()

# ====== CONFIG ======
SUPABASE_URL = (os.getenv("SUPABASE_URL", "") or "").strip()
SUPABASE_SERVICE_KEY = (os.getenv("SUPABASE_SERVICE_KEY", "") or "").strip()

JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_SUPER_SECRET")
JWT_ALG = "HS256"
JWT_EXPIRES_DAYS = 14


FRONTEND_ORIGIN = (os.getenv("FRONTEND_ORIGIN", "") or "").strip()

ENV = (os.getenv("ENV", "production") or "production").strip().lower()
IS_PROD = ENV == "production"

if not SUPABASE_URL or not SUPABASE_URL.startswith("https://"):
    raise RuntimeError("SUPABASE_URL is missing/invalid. Put it in .env as https://xxxx.supabase.co")

if not SUPABASE_SERVICE_KEY or not SUPABASE_SERVICE_KEY.startswith("ey"):
    raise RuntimeError("SUPABASE_SERVICE_KEY is missing/invalid. Put service_role key in .env")

if not FRONTEND_ORIGIN or FRONTEND_ORIGIN.startswith("http://localhost") or "127.0.0.1" in FRONTEND_ORIGIN:
    raise RuntimeError("FRONTEND_ORIGIN is missing/invalid (must be a real domain, not localhost). Put it in .env")

REST_BASE = f"{SUPABASE_URL}/rest/v1"

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://nawras-frontend.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== ONE SHARED HTTP CLIENT (FAST) ======
sb_client: httpx.AsyncClient | None = None

@app.on_event("startup")
async def startup_event():
    global sb_client
    # ❌ لا تستخدم http2=True (يطلب h2)
    sb_client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=10.0))

@app.on_event("shutdown")
async def shutdown_event():
    global sb_client
    if sb_client is not None:
        await sb_client.aclose()
        sb_client = None

# ====== Schemas ======
class RegisterIn(BaseModel):
    name: str
    email: EmailStr
    password: str
    grade: str | None = None
    school: str | None = None
    path: str | None = None
    field: str | None = None
    phone: str | None = None
    governorate: str | None = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class EmailIn(BaseModel):
    email: EmailStr

class VerifyCodeIn(BaseModel):
    email: EmailStr
    code: str

# ====== Helpers ======
def make_jwt(user_id_uuid: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRES_DAYS)
    payload = {"sub": user_id_uuid, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def set_auth_cookie(resp: Response, token: str):
    resp.set_cookie(
        key="nawras_token",
        value=token,
        httponly=True,
        secure=True,
        samesite="none",
        domain=".onrender.com",  # ⭐⭐⭐ هاي الإضافة
        max_age=JWT_EXPIRES_DAYS * 24 * 3600,
        path="/",
    )


def get_user_from_cookie(req: Request) -> str:
    token = req.cookies.get("nawras_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return data["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

def send_email_code(email: str, code: str):
    # مؤقتاً اطبعيه بالكونسول (بدك خدمة ايميل لاحقاً)
    print(f"✅ Verification code for {email}: {code}")

def sb_headers(return_representation: bool = False):
    h = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if return_representation:
        h["Prefer"] = "return=representation"
    return h

def _ensure_client() -> httpx.AsyncClient:
    if sb_client is None:
        raise HTTPException(status_code=500, detail="HTTP client not initialized")
    return sb_client

async def sb_get(table: str, params: dict):
    client = _ensure_client()
    r = await client.get(f"{REST_BASE}/{table}", headers=sb_headers(), params=params)
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase GET error: {r.status_code} {r.text}")
    return r.json()

async def sb_post(table: str, data: dict, return_rep: bool = True):
    client = _ensure_client()
    r = await client.post(
        f"{REST_BASE}/{table}",
        headers=sb_headers(return_representation=return_rep),
        json=data,
    )
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase POST error: {r.status_code} {r.text}")
    return r.json() if r.text.strip() else []

async def sb_patch(table: str, data: dict, params: dict):
    client = _ensure_client()
    r = await client.patch(
        f"{REST_BASE}/{table}",
        headers=sb_headers(return_representation=True),
        params=params,
        json=data,
    )
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase PATCH error: {r.status_code} {r.text}")
    return r.json() if r.text.strip() else []

async def sb_delete(table: str, params: dict):
    client = _ensure_client()
    r = await client.delete(f"{REST_BASE}/{table}", headers=sb_headers(), params=params)
    if r.status_code >= 400:
        raise HTTPException(status_code=500, detail=f"Supabase DELETE error: {r.status_code} {r.text}")
    return True

# ====== Endpoints ======
@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/register")
async def register(payload: RegisterIn):
    email = payload.email.strip().lower()

    existing = await sb_get("users", {"select": "id_uuid", "email": f"eq.{email}", "limit": 1})
    if existing:
        raise HTTPException(status_code=409, detail="Email already exists")

    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 chars")

    # ملاحظة: rounds=10 أسرع للتجربة، على الإنتاج خليها 12+
    pw_hash = bcrypt.hashpw(
        payload.password.encode("utf-8"),
        bcrypt.gensalt(rounds=10 if not IS_PROD else 12)
    ).decode("utf-8")

    new_uuid = str(uuid.uuid4())

    data = await sb_post("users", {
        "id_uuid": new_uuid,
        "name": payload.name,
        "email": email,
        "password_hash": pw_hash,
        "is_verified": False,
        "grade": payload.grade,
        "school": payload.school,
        "path": payload.path,
        "field": payload.field,
        "phone": payload.phone,
        "governorate": payload.governorate,
    }, return_rep=True)

    user_id = data[0].get("id_uuid") if data else new_uuid
    return {"ok": True, "user_id": user_id}

@app.post("/api/auth/request_email_code")
async def request_email_code(payload: EmailIn):
    email = payload.email.strip().lower()

    user = await sb_get("users", {"select": "id_uuid,is_verified", "email": f"eq.{email}", "limit": 1})
    if not user:
        return {"ok": True}

    if user[0].get("is_verified") is True:
        return {"ok": True}

    code = f"{secrets.randbelow(10**6):06d}"
    code_h = hash_code(code)
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    await sb_delete("email_verifications", {"email": f"eq.{email}"})

    await sb_post("email_verifications", {
        "email": email,
        "code_hash": code_h,
        "expires_at": expires.isoformat(),
    }, return_rep=False)

    send_email_code(email, code)
    return {"ok": True}

@app.post("/api/auth/verify_email_code")
async def verify_email_code(payload: VerifyCodeIn):
    email = payload.email.strip().lower()
    code = payload.code.strip()

    row = await sb_get("email_verifications", {"select": "*", "email": f"eq.{email}", "limit": 1})
    if not row:
        raise HTTPException(status_code=400, detail="No code requested")

    rec = row[0]
    expires_at_str = str(rec["expires_at"]).replace("Z", "+00:00")
    expires_at = datetime.fromisoformat(expires_at_str)

    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Code expired")

    if rec["code_hash"] != hash_code(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    await sb_patch("users", {"is_verified": True}, {"email": f"eq.{email}"})
    await sb_delete("email_verifications", {"email": f"eq.{email}"})
    return {"ok": True}

@app.post("/api/login")
async def login(payload: LoginIn, resp: Response):
    email = payload.email.strip().lower()

    user = await sb_get("users", {"select": "id_uuid,password_hash,is_verified", "email": f"eq.{email}", "limit": 1})
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    u = user[0]
    stored = (u.get("password_hash") or "").encode("utf-8")

    if not stored or not bcrypt.checkpw(payload.password.encode("utf-8"), stored):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if u.get("is_verified") is not True:
        raise HTTPException(status_code=403, detail={"code": "EMAIL_NOT_VERIFIED", "message": "Email not verified"})

    token = make_jwt(u["id_uuid"])
    set_auth_cookie(resp, token)
    return {"ok": True}

@app.post("/api/logout")
def logout(resp: Response):
    resp.delete_cookie("nawras_token", path="/")
    return {"ok": True}

@app.get("/api/me")
async def me(req: Request):
    user_id = get_user_from_cookie(req)

    profile = await sb_get(
        "users",
        {"select": "id_uuid,name,email,grade,school,path,field,is_verified", "id_uuid": f"eq.{user_id}", "limit": 1},
    )

    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile[0]
