from fastapi import FastAPI, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import os, bcrypt, jwt, secrets, hashlib, uuid
import httpx
from dotenv import load_dotenv
from typing import Optional, Dict, Any

# Resend
import resend

load_dotenv()

# ====== CONFIG ======
SUPABASE_URL = (os.getenv("SUPABASE_URL", "") or "").strip()
SUPABASE_SERVICE_KEY = (os.getenv("SUPABASE_SERVICE_KEY", "") or "").strip()

JWT_SECRET = (os.getenv("JWT_SECRET", "CHANGE_ME_SUPER_SECRET") or "CHANGE_ME_SUPER_SECRET").strip()
JWT_ALG = "HS256"
JWT_EXPIRES_DAYS = int(os.getenv("JWT_EXPIRES_DAYS", "14") or "14")

FRONTEND_ORIGIN = (os.getenv("FRONTEND_ORIGIN", "") or "").strip()
ENV = (os.getenv("ENV", "production") or "production").strip().lower()
IS_PROD = ENV == "production"

# Resend env
RESEND_API_KEY = (os.getenv("RESEND_API_KEY", "") or "").strip()
EMAIL_FROM = (os.getenv("EMAIL_FROM", "") or "").strip()
EMAIL_REPLY_TO = (os.getenv("EMAIL_REPLY_TO", "") or "").strip()

if RESEND_API_KEY:
    resend.api_key = RESEND_API_KEY

if not SUPABASE_URL or not SUPABASE_URL.startswith("https://"):
    raise RuntimeError("SUPABASE_URL is missing/invalid. Put it in .env as https://xxxx.supabase.co")

if not SUPABASE_SERVICE_KEY or not SUPABASE_SERVICE_KEY.startswith("ey"):
    raise RuntimeError("SUPABASE_SERVICE_KEY is missing/invalid. Put service_role key in .env")

REST_BASE = f"{SUPABASE_URL}/rest/v1"

app = FastAPI()

# ====== CORS ======
# مهم: صفحة الأسئلة على Vercel غالباً ما بتبعت كوكي، فنعتمد كمان على Bearer Token.
allowed_origins = []
if FRONTEND_ORIGIN:
    allowed_origins.append(FRONTEND_ORIGIN)

# للـ local
allowed_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====== ONE SHARED HTTP CLIENT ======
sb_client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def startup_event():
    global sb_client
    sb_client = httpx.AsyncClient(timeout=httpx.Timeout(25.0, connect=10.0))

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
    grade: Optional[str] = None
    school: Optional[str] = None
    path: Optional[str] = None
    field: Optional[str] = None
    phone: Optional[str] = None
    governorate: Optional[str] = None

class LoginIn(BaseModel):
    email: EmailStr
    password: str

class EmailIn(BaseModel):
    email: EmailStr

class VerifyCodeIn(BaseModel):
    email: EmailStr
    code: str

class ResetRequestIn(BaseModel):
    email: EmailStr

class ResetPasswordIn(BaseModel):
    email: EmailStr
    code: str
    new_password: str

class ChatIn(BaseModel):
    user_id: str
    question: str
    session_id: Optional[str] = None
    log_history: bool = True
    source: Optional[str] = None

# ====== Helpers ======
def require_gmail(email: str):
    e = (email or "").strip().lower()
    if not e.endswith("@gmail.com"):
        raise HTTPException(
            status_code=400,
            detail={"code": "ONLY_GMAIL_ALLOWED", "message": "Only Gmail accounts are allowed"}
        )

def make_jwt(user_id_uuid: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=JWT_EXPIRES_DAYS)
    payload = {"sub": user_id_uuid, "exp": exp}
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALG)

def set_auth_cookie(resp: Response, token: str):
    resp.set_cookie(
        key="nawras_token",
        value=token,
        httponly=True,
        secure=True if IS_PROD else False,
        samesite="none" if IS_PROD else "lax",
        max_age=JWT_EXPIRES_DAYS * 24 * 3600,
        path="/",
    )

def clear_auth_cookie(resp: Response):
    resp.delete_cookie(key="nawras_token", path="/")

def _get_bearer_token(req: Request) -> Optional[str]:
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(" ")
    if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
        return parts[1].strip()
    return None

def get_user_from_request(req: Request) -> str:
    # ✅ يدعم cookie أو Bearer token
    token = req.cookies.get("nawras_token") or _get_bearer_token(req)
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALG])
        return data["sub"]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()

# ====== Resend ======
def send_email(to_email: str, subject: str, html: str, text: Optional[str] = None):
    if not RESEND_API_KEY or not EMAIL_FROM:
        raise HTTPException(status_code=500, detail="Email not configured: missing RESEND_API_KEY or EMAIL_FROM")

    payload: Dict[str, Any] = {"from": EMAIL_FROM, "to": [to_email], "subject": subject, "html": html}
    if text:
        payload["text"] = text
    if EMAIL_REPLY_TO:
        payload["reply_to"] = EMAIL_REPLY_TO

    try:
        resend.Emails.send(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to send email: {e}")

def send_email_code(email: str, code: str):
    subject = "Nawras | كود تفعيل الحساب"
    html = f"""
    <div style="font-family:Arial,sans-serif; line-height:1.9; direction:rtl; text-align:right">
      <h2 style="color:#0284c7; margin:0 0 10px">تفعيل حسابك في نَوْرَس</h2>
      <p>هذا هو <b>كود التفعيل</b> (صالح لمدة 10 دقائق):</p>
      <div style="font-size:28px; letter-spacing:6px; font-weight:700; padding:12px 16px;
                  background:#f0f9ff; border:1px solid #bae6fd; display:inline-block; border-radius:12px;">
        {code}
      </div>
      <p style="margin-top:14px; color:#555">إذا لم تطلبي هذا الكود، تجاهلي الرسالة.</p>
      <p style="font-size:12px; color:#777">Nawras Team</p>
    </div>
    """
    send_email(email, subject, html, f"Nawras verification code: {code} (valid 10 minutes)")

def send_reset_code(email: str, code: str):
    subject = "Nawras | كود استرجاع كلمة المرور"
    html = f"""
    <div style="font-family:Arial,sans-serif; line-height:1.9; direction:rtl; text-align:right">
      <h2 style="color:#0284c7; margin:0 0 10px">استرجاع كلمة المرور</h2>
      <p>هذا هو <b>كود الاسترجاع</b> (صالح لمدة 10 دقائق):</p>
      <div style="font-size:28px; letter-spacing:6px; font-weight:700; padding:12px 16px;
                  background:#f0f9ff; border:1px solid #bae6fd; display:inline-block; border-radius:12px;">
        {code}
      </div>
      <p style="margin-top:14px; color:#555">إذا لم تطلبي هذا الطلب، تجاهلي الرسالة.</p>
      <p style="font-size:12px; color:#777">Nawras Team</p>
    </div>
    """
    send_email(email, subject, html, f"Nawras password reset code: {code} (valid 10 minutes)")

# ====== Supabase REST helpers ======
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
@app.get("/")
def root():
    return {"ok": True, "service": "nawras-backend"}

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/register")
async def register(payload: RegisterIn):
    email = payload.email.strip().lower()
    require_gmail(email)

    existing = await sb_get("users", {"select": "id_uuid,is_verified", "email": f"eq.{email}", "limit": 1})
    if existing:
        if existing[0].get("is_verified") is not True:
            raise HTTPException(
                status_code=409,
                detail={"code": "EMAIL_EXISTS_NOT_VERIFIED", "message": "Email exists but not verified"},
            )
        raise HTTPException(status_code=409, detail="Email already exists")

    if len(payload.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 chars")

    pw_hash = bcrypt.hashpw(
        payload.password.encode("utf-8"),
        bcrypt.gensalt(rounds=12 if IS_PROD else 10)
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
    require_gmail(email)

    user = await sb_get("users", {"select": "id_uuid,is_verified", "email": f"eq.{email}", "limit": 1})
    if not user or user[0].get("is_verified") is True:
        return {"ok": True}

    code = f"{secrets.randbelow(10**6):06d}"
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    await sb_delete("email_verifications", {"email": f"eq.{email}"})
    await sb_post("email_verifications", {
        "email": email,
        "code_hash": hash_code(code),
        "expires_at": expires.isoformat(),
    }, return_rep=False)

    send_email_code(email, code)
    return {"ok": True}

@app.post("/api/auth/verify_email_code")
async def verify_email_code(payload: VerifyCodeIn):
    email = payload.email.strip().lower()
    require_gmail(email)
    code = payload.code.strip()

    row = await sb_get("email_verifications", {"select": "*", "email": f"eq.{email}", "limit": 1})
    if not row:
        raise HTTPException(status_code=400, detail="No code requested")

    rec = row[0]
    expires_at = datetime.fromisoformat(str(rec["expires_at"]).replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Code expired")
    if rec["code_hash"] != hash_code(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    await sb_patch("users", {"is_verified": True}, {"email": f"eq.{email}"})
    await sb_delete("email_verifications", {"email": f"eq.{email}"})
    return {"ok": True}

@app.post("/api/auth/request_password_reset")
async def request_password_reset(payload: ResetRequestIn):
    email = payload.email.strip().lower()
    require_gmail(email)

    user = await sb_get("users", {"select": "id_uuid", "email": f"eq.{email}", "limit": 1})
    if not user:
        return {"ok": True}

    code = f"{secrets.randbelow(10**6):06d}"
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)

    await sb_delete("password_resets", {"email": f"eq.{email}"})
    await sb_post("password_resets", {
        "email": email,
        "code_hash": hash_code(code),
        "expires_at": expires.isoformat(),
    }, return_rep=False)

    send_reset_code(email, code)
    return {"ok": True}

@app.post("/api/auth/reset_password")
async def reset_password(payload: ResetPasswordIn):
    email = payload.email.strip().lower()
    require_gmail(email)
    code = payload.code.strip()

    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 chars")

    row = await sb_get("password_resets", {"select": "*", "email": f"eq.{email}", "limit": 1})
    if not row:
        raise HTTPException(status_code=400, detail="No reset code requested")

    rec = row[0]
    expires_at = datetime.fromisoformat(str(rec["expires_at"]).replace("Z", "+00:00"))
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Code expired")
    if rec["code_hash"] != hash_code(code):
        raise HTTPException(status_code=400, detail="Invalid code")

    pw_hash = bcrypt.hashpw(
        payload.new_password.encode("utf-8"),
        bcrypt.gensalt(rounds=12 if IS_PROD else 10)
    ).decode("utf-8")

    await sb_patch("users", {"password_hash": pw_hash}, {"email": f"eq.{email}"})
    await sb_delete("password_resets", {"email": f"eq.{email}"})
    return {"ok": True}

@app.post("/api/login")
async def login(payload: LoginIn, resp: Response):
    email = payload.email.strip().lower()
    require_gmail(email)

    user = await sb_get("users", {"select": "id_uuid,name,email,password_hash,is_verified", "email": f"eq.{email}", "limit": 1})
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

    # ✅ رجّعي token للفرونت عشان Page الأسئلة تبعت Bearer
    return {
        "ok": True,
        "user_id": u["id_uuid"],
        "name": u.get("name") or "",
        "email": u.get("email") or "",
        "token": token,
    }

@app.post("/api/logout")
def logout(resp: Response):
    clear_auth_cookie(resp)
    return {"ok": True}

@app.get("/api/me")
async def me(req: Request):
    user_id = get_user_from_request(req)
    profile = await sb_get(
        "users",
        {"select": "id_uuid,name,email,grade,school,path,field,is_verified,phone,governorate",
         "id_uuid": f"eq.{user_id}", "limit": 1},
    )
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile[0]

# =========================
# Chat + History (Protected)
# =========================
@app.get("/api/history_db")
async def history_db(req: Request, user_id: str, session_id: Optional[str] = None):
    auth_user = get_user_from_request(req)
    if auth_user != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    params = {
        "select": "id,role,content,ts,session_id,source",
        "user_id": f"eq.{user_id}",
        "order": "ts.asc",
        "limit": 400,
    }
    if session_id:
        params["session_id"] = f"eq.{session_id}"

    return await sb_get("wizard_sessions", params)

@app.post("/api/chat")
async def chat(req: Request, payload: ChatIn):
    auth_user = get_user_from_request(req)
    if auth_user != payload.user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    sid = payload.session_id or str(uuid.uuid4())

    # ✅ مؤقتاً جواب بسيط
    answer = f"وصلني سؤالك: {payload.question}\n(جاري ربط الإجابة الذكية لاحقاً)."

    if payload.log_history:
        now_iso = datetime.now(timezone.utc).isoformat()

        await sb_post("wizard_sessions", {
            "user_id": payload.user_id,
            "session_id": sid,
            "role": "user",
            "content": payload.question,
            "ts": now_iso,
            "source": payload.source or "questions_page",
        }, return_rep=False)

        await sb_post("wizard_sessions", {
            "user_id": payload.user_id,
            "session_id": sid,
            "role": "assistant",
            "content": answer,
            "ts": now_iso,
            "source": payload.source or "questions_page",
        }, return_rep=False)

    return {"ok": True, "session_id": sid, "answer": answer}
