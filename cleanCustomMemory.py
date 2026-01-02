#cleanCustomMemory.py# %% [markdown]
# ## Costum version storing in qdrant 

# %%
# memory_and_rag_pipeline.py

import os
import uuid
#import sqlite3
import psycopg2
from datetime import datetime
from typing import Dict, Optional, List

from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.http import models as rest
import bcrypt


# ----------------------------------------
# CONFIG
# ----------------------------------------
QDRANT_URL = "https://980d3339-dbc8-4075-9542-f929b824df37.europe-west3-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.nz4wTnjVEvAf3JLmOI658h8R3Sw0in-mUvX04tgbvDU")


EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
USER_PROFILE_COLLECTION = "User_Profiles"
MEMORY_COLLECTION = "User_Memories"
RAG_COLLECTION = "Tawjihi_Knowledge_files_collection"

FETCH_K = 10
FINAL_K = 10



STM_TOKEN_LIMIT = 1000

# Configure Gemini API key
import google.generativeai as genai

# --- Configure your API key ---
genai.configure(api_key="AIzaSyAPgxi065xLdcu-e_W9TB8ZYzUJ3hhMW34")

# --- Initialize the model ---
model = genai.GenerativeModel('gemini-2.5-flash')

embedder = SentenceTransformer(EMBEDDING_MODEL)


#RERANKER_MODEL =  "BAAI/bge-reranker-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker = CrossEncoder(RERANKER_MODEL)
#max_length=512,  # قصير لزيادة السرعة




# %%
# ----------------------------------------
# Qdrant client + collections
# ----------------------------------------
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

def ensure_collection(name: str, dim: int):
    if not qdrant_client.collection_exists(name):
        qdrant_client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

# embedder dimension
test_emb = embedder.encode(["test"], convert_to_numpy=True)[0]
VECTOR_DIM = len(test_emb)

#print(VECTOR_DIM)

ensure_collection(MEMORY_COLLECTION, VECTOR_DIM)
ensure_collection(USER_PROFILE_COLLECTION, VECTOR_DIM)


# %%
from supabase import create_client
import bcrypt, uuid
from typing import Optional, Dict

# --- CONFIGURATION ---
SUPABASE_URL = "https://obqomfvcysryuzuqyvsa.supabase.co"
SUPABASE_KEY="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9icW9tZnZjeXNyeXV6dXF5dnNhIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2Mjk2ODM4MSwiZXhwIjoyMDc4NTQ0MzgxfQ.Uk3oeMlDXWCRf0YnP3OVCQS6c-wrvP2ENfs0e4_0XAs"
# use service_role, not anon

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


from threading import Lock

qdrant_sync_lock = Lock()
qdrant_profile_lock = Lock()
qdrant_memory_lock = Lock()


# --- HELPERS ---
def fetch_safe_user_profile(user_id: str) -> Optional[Dict]:
    """Fetch limited user info from Supabase by UUID."""
    res = supabase.table("users").select(
        "uuid, name, grade, school, path, field"
    ).eq("uuid", user_id).execute()

    if not res.data or len(res.data) == 0:
        return None

    row = res.data[0]
    return {
        "user_id": row["uuid"],
        "name": row.get("name"),
        "grade": row.get("grade"),
        "school": row.get("school"),
        "path": row.get("path"),
        "field": row.get("field")
    }


def upsert_sql_user_to_qdrant(user_id: str):
    """Sync user profile from Supabase to Qdrant."""
    with qdrant_profile_lock:  # 🔒 prevent race conditions
        profile = fetch_safe_user_profile(user_id)
        if not profile:
            print(f"⚠️ No profile found for user {user_id}")
            return

    # Create vector
    profile_text = (
        f"الطالب: {profile['name']}, الصف: {profile['grade']}, "
        f"المدرسة: {profile['school']}, المسار: {profile['path']}, الحقل: {profile['field']}"
    )
    vector = embedder.encode(profile_text, convert_to_numpy=True).tolist()

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, profile["user_id"]))

    qdrant_client.upsert(
        collection_name=USER_PROFILE_COLLECTION,
        points=[{
            "id": point_id,
            "vector": vector,
            "payload": profile
        }]
    )
    print(f"🧠 Synced profile for {profile['name']} to Qdrant.")


# --- MAIN REGISTER FUNCTION ---
def register_user(name, email, password, grade=None, school=None, path=None, field=None):
    """Register a new user in Supabase and sync to Qdrant."""
    try:
        # Check if email exists
        existing = supabase.table("users").select("email").eq("email", email).execute()
        if existing.data and len(existing.data) > 0:
            print("⚠️ Email already exists!")
            return None

        # Hash password
        password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        user_uuid = str(uuid.uuid4())

        # Insert new user
        response = supabase.table("users").insert({
            "uuid": user_uuid,
            "name": name,
            "email": email,
            "password_hash": password_hash,
            "grade": grade,
            "school": school,
            "path": path,
            "field": field
        }).execute()

        if not response.data:
            print("⚠️ Failed to insert user (no response data).")
            return None

        # Sync to Qdrant
        upsert_sql_user_to_qdrant(user_uuid)
        print(f"✅ Registered user '{name}' successfully.")
        return user_uuid

    except Exception as e:
        print(f"⚠️ Failed to register user: {e}")
        return None


def verify_user(email: str, password: str) -> Optional[str]:
    """Verify user credentials and return UUID if valid, else None."""
    try:
        res = supabase.table("users").select("uuid, password_hash").eq("email", email).execute()
        if not res.data or len(res.data) == 0:
            return None

        user = res.data[0]
        stored_hash = user["password_hash"].encode("utf-8")
        if bcrypt.checkpw(password.encode("utf-8"), stored_hash):
            return user["uuid"]
        return None
    except Exception as e:
        print(f"⚠️ Login check failed: {e}")
        return None


def get_user_profile(user_id: str) -> Optional[Dict]:
    """
    Retrieve user profile from Qdrant. 
    If missing, automatically fetch from Supabase and sync to Qdrant.
    """
    try:
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(user_id)))
        res = qdrant_client.retrieve(collection_name=USER_PROFILE_COLLECTION, ids=[point_id])
        if res and len(res) > 0 and res[0].payload:
            return res[0].payload
    except Exception as e:
        print(f"⚠️ Qdrant profile retrieval failed: {e}")

    # If not found in Qdrant, fetch from Supabase and upsert
    profile = fetch_safe_user_profile(user_id)
    if profile:
        try:
            upsert_sql_user_to_qdrant(user_id)
            print(f"Profile for user {user_id} synced to Qdrant automatically.")
        except Exception as e:
            print(f"⚠️ Failed to sync profile to Qdrant: {e}")
        return profile
    
    # Profile missing everywhere
    print(f"⚠️ User profile for {user_id} not found in Supabase or Qdrant.")
    return None



# %%
# ----------------------------------------
# Chat Session Manager
# ----------------------------------------
class ChatSession:
    """Keeps track of session identity and title for continuity."""
    def __init__(self, session_id=None, session_title=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.session_title = session_title or datetime.now().strftime("Session %Y-%m-%d %H:%M")

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "session_title": self.session_title
        }


# %%
# Ensure user_id and field are indexed for filtering
try:
    qdrant_client.create_payload_index(
        collection_name=MEMORY_COLLECTION,
        field_name="user_id",
        field_schema="keyword"
    )
except Exception as e:
    print("user_id index may already exist:", e)

try:
    qdrant_client.create_payload_index(
        collection_name=MEMORY_COLLECTION,
        field_name="field",
        field_schema="keyword"
    )
except Exception as e:
    print("field index may already exist:", e)

try:
    qdrant_client.create_payload_index(
        collection_name=MEMORY_COLLECTION,
        field_name="session_id",
        field_schema="keyword"
    )
except Exception as e:
    print("session_id index may already exist:", e)


# %%
# ----------------------------------------
# RAG retriever stub (your existing logic)
# ----------------------------------------
from typing import Dict
import numpy as np


def retrieve_and_rerank(query, fetch_k=FETCH_K, final_k=FINAL_K):
    q_emb = embedder.encode([query], convert_to_numpy=True)[0]
    COLLECTION_NAME = "Tawjihi_Knowledge_files_collection"
    results = qdrant_client.query_points(
    collection_name=COLLECTION_NAME,
    query=q_emb,
    limit=fetch_k
)
    
    hits = results.points if hasattr(results, "points") else results

    if not hits:
        return ""

    chunks = []
    for h in hits:
        payload = h.payload or {}
        chunks.append({
            "text": payload.get("text"),
            "source": payload.get("source"),
            "pos": payload.get("pos")
        })

    # Rerank
    candidate_texts = [c["text"] for c in chunks if c["text"]]
    pairs = [(query, t) for t in candidate_texts]
    scores = reranker.predict(pairs)
    sorted_idx = np.argsort(scores)[::-1][:final_k]
    top_chunks = [candidate_texts[i] for i in sorted_idx]

    # Clean + combine
    unique_texts = []
    for t in top_chunks:
        if t not in unique_texts:
            unique_texts.append(t.strip())

    # Add clear separation between chunks
    combined_text = "\n\n-----------------------------\n\n".join(unique_texts)
    return combined_text



# %%
# ----------------------------------------
# Chat pipeline with memory + RAG + Gemini
# ----------------------------------------

class ChatPipeline:

    def __init__(
        self,
        user_id: str,
        session_id: str = None,
        session_title: str = None
    ):
        self.user_id = user_id

        # 🧩 Try to restore session title if this session already exists in Qdrant
        existing_title = None
        if session_id:
            try:
                results, _ = qdrant_client.scroll(
                    collection_name=MEMORY_COLLECTION,
                    scroll_filter=rest.Filter(
                        must=[
                            rest.FieldCondition(key="session_id", match=rest.MatchValue(value=session_id))
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                    with_vectors=False
                )
                if results:
                    existing_title = results[0].payload.get("session_title")
            except Exception as e:
                print(f"Could not retrieve old session title: {e}")

        # Prefer the existing title from Qdrant if found
        if existing_title:
            session_title = existing_title

        # Now safely initialize the ChatSession object
        self.session = ChatSession(session_id=session_id, session_title=session_title)
        self.session_id = self.session.session_id
        self.session_title = self.session.session_title

        # Load local short-term memory for this session
        self.local_stm = []
        self.load_recent_stm()


    def get_recent_context(self, n_pairs=1):
        """Return the last n user-assistant pairs plus current user message."""
        # Filter only user/assistant messages
        msgs = [m for m in self.local_stm if m["role"] in ("user", "assistant")]

        # Get last n pairs (each pair is 2 messages)
        context = msgs[-(2 * n_pairs + 1):]  # previous n pairs + current user message
        return context


    def reformulate_query(self, user_query: str) -> str:
    
        """Use Gemini Flash 2.5 to rewrite vague/follow-up questions into clear ones."""
        profile = get_user_profile(self.user_id)
        field = profile.get("field", "")
        last_msgs = self.get_recent_context(n_pairs=1)
        context_snippet = (
        "\n".join(f"{m['role']}: {m['content']}" for m in last_msgs)
        if last_msgs else "لا يوجد سياق سابق."
        )
        # print("Context used for reformulation:")
        # print(context_snippet)


        prompt = f"""
        أنت مساعد لغوي ذكي يساعد في تحسين صياغة أسئلة الطلاب لتكون أكثر وضوحًا وفهمًا لنظام الاسترجاع (RAG).

        الطالب
        حقله الدراسي: {field}
        السياق السابق من المحادثة:
        {context_snippet}

        السؤال الأصلي من الطالب:
        {user_query}

        أعد صياغة السؤال بحيث يصبح واضحًا ومفهومًا لنظام الاسترجاع (RAG)،
        مع الحفاظ على المعنى الأصلي. 
        لا تضف أي إجابات أو معلومات إضافية من السياق.
        فقط ضف معلومات الحقل
        قم فقط بإعادة صياغة السؤال.

        """

        reformulated = model.generate_content(prompt)
        return reformulated.text.strip()

    def upsert_conversation_pair(self, user_msg, assistant_msg):
        """Batch upsert a user-assistant message pair into Qdrant."""
        with qdrant_memory_lock:  # 🔒 ensure order & avoid collisions
            try:
                # Combine texts for embedding
                combined_text = f"user: {user_msg['content']}\nassistant: {assistant_msg['content']}"
                vector = embedder.encode(combined_text, convert_to_numpy=True).tolist()

                point_id = str(uuid.uuid4())

                payload = {
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                    "session_title": self.session_title,
                    "user_text": user_msg["content"],
                    "assistant_text": assistant_msg["content"],
                    "timestamp_user": user_msg["timestamp"],
                    "timestamp_assistant": assistant_msg["timestamp"],
                    "text_preview": combined_text[:120]
                }

                qdrant_client.upsert(
                    collection_name=MEMORY_COLLECTION,
                    points=[{
                        "id": point_id,
                        "vector": vector,
                        "payload": payload
                    }]
                )

                #print(f" Stored user-assistant pair for session {self.session_id}.")
            except Exception as e:
                print(f" Failed to store user-assistant pair: {e}")

    
    def generate_session_title(self, first_msg: str) -> str:
        prompt = (
        "أنت مساعد ذكي يقوم بتلخيص المحادثة في عنوان قصير جداً وواضح، لا يتجاوز 5 كلمات، "
        "ويصلح أن يكون عنوان جلسة دردشة.\n"
        "لا تكتب أي شرح أو خيارات أو تنسيقات.\n"
        f"الرسالة الأولى للمستخدم: {first_msg}\n"
        "اكتب العنوان فقط بدون أي إضافات."
        )
        response = model.generate_content(prompt)
        return response.text.strip()


    def store_message(self, role, text):
        timestamp = datetime.now().isoformat()
        msg = {"role": role, "content": text, "timestamp": timestamp}
        self.local_stm.append(msg)

        # 🧠 Trim STM to last N messages
        MAX_STM_MESSAGES = 10
        if len(self.local_stm) > MAX_STM_MESSAGES:
            self.local_stm = self.local_stm[-MAX_STM_MESSAGES:]

        if role == "user" and len(self.local_stm) == 1:
            new_title = self.generate_session_title(text)
            self.session_title = new_title
            self.session.session_title = new_title


    def retrieve_ltm(self, query: str, limit: int = 7) -> List[str]:
        try:
            q_emb = embedder.encode([query], convert_to_numpy=True)[0]
            # Build filter
            query_filter = rest.Filter(
                must=[rest.FieldCondition(key="user_id", match=rest.MatchValue(value=self.user_id))]  
            )

            results = qdrant_client.query_points(
                collection_name=MEMORY_COLLECTION,
                query=q_emb,      
                query_filter=query_filter,
                limit=limit
            )

            points = results.points if hasattr(results, "points") else []

            return [ f"User: {p.payload.get('user_text','')}\nAssistant: {p.payload.get('assistant_text','')}"
            for p in points if p.payload]

        except Exception as e:
            print(f" LTM retrieval failed: {e}")
            return []
    
    def build_context(self, user_query: str) -> str:

        # 1- User profile
        profile = get_user_profile(self.user_id)
        profile_text = ""
        if profile:
            profile_text =  f"الطالب {profile.get('name','')} من {profile.get('field','')}."


        # 2- STM last 3 messages used 
        last_msgs = self.local_stm[-3:]
        stm_text = "\n".join(f"{m['role']}: {m['content']}" for m in last_msgs)

        # 4- RAG
        # Reformulate query for RAG
        reformed_query = self.reformulate_query(user_query)
        rag_text = retrieve_and_rerank(reformed_query)
        #print("Original query:", user_query)
        #print("Reformed query:", reformed_query)
        # print("RAG: -----------------------------------")
        # print(rag_text)


        # 3- LTM 
        #ltm_text = self.retrieve_ltm(user_query)
        ltm_text = self.retrieve_ltm(reformed_query)
        #print("LTM:")
        if ltm_text:
            ltm_text = f"\n\n[تذكّر من المحادثات السابقة]\n{ltm_text}"
        #print(ltm_text)

        parts = [p for p in [profile_text, stm_text, ltm_text, rag_text] if p]
        combined = "\n\n".join(parts)
        #print("lookkkkk\n"+combined)
        return combined

    def load_recent_stm(self, limit=3):
        """Retrieve the most recent few message pairs for STM from Qdrant."""
        try:
            results, _ = qdrant_client.scroll(
                collection_name=MEMORY_COLLECTION,
                scroll_filter=rest.Filter(
                    must=[
                        rest.FieldCondition(key="user_id", match=rest.MatchValue(value=self.user_id)),
                        rest.FieldCondition(key="session_id", match=rest.MatchValue(value=self.session_id))
                    ]
                ),
                limit=100,  # Fetch more to ensure we get the latest few
                with_payload=True,
                with_vectors=False
            )

            stm = []
            for point in results:
                p = point.payload or {}
                if p.get("user_text"):
                    stm.append({"role": "user", "content": p["user_text"], "timestamp": p.get("timestamp_user", "")})
                if p.get("assistant_text"):
                    stm.append({"role": "assistant", "content": p["assistant_text"], "timestamp": p.get("timestamp_assistant", "")})

            stm = sorted(stm, key=lambda x: x["timestamp"])  # ensure chronological
            self.local_stm = stm[-(2 * limit):]  # last user+assistant pairs

            #if self.local_stm:
                #print(f"Loaded {len(self.local_stm)} STM messages (latest).")
                # print(self.local_stm)
            #else:
                #print(f" No STM found for session {self.session_id}.")
        except Exception as e:
            print(f"Failed to load STM: {e}")




    def ask_gemini(self, context: str, question: str) -> str:
        last_msgs = self.get_recent_context(n_pairs=1)
        context_snippet = "\n".join(f"{m['role']}: {m['content']}" for m in last_msgs)
        prompt = f"""
    أنت مساعد ذكي ومستشار لطلاب التوجيهي في الأردن عبر منصة نورس.

    استخدم فقط مصادر نورس للإجابة على السؤال بطريقة دقيقة وواضحة.
    استعمل فقط المعلومات الدقيقة المتعلقة بالسؤال، وتجاهل أي تكرار أو معلومات غير متعلقة.

    التعليمات الإضافية:
    - لا تبدأ الإجابة بأي تحية أو جملة ترحيبية.
    - لا تستخدم أي رموز تنسيق مثل * أو ** أو علامات نجمة أو فواصل زخرفية.
    - استخدم لغة عربية فصحى واضحة وبأسلوب رسمي وموجز.
    - علامة "+" في سياق الحقول تعني الدمج

    - بعد الإجابة، اطرح سؤال متابعة بسيط يساعد المستخدم إن كان يريد معرفة أكثر أو توضيحاً إضافياً، ويكون السؤال متعلّقاً بالإجابة.
    - لا تطرح أي سؤال متابعة إلا إذا كانت المصادر الواردة في المحتوى تحتوي فعلاً على معلومات إضافية يمكن التوسع فيها.
    - إذا لم يكن في المحتوى ما يمكن التوسع فيه، اكتفِ بالإجابة فقط بدون أي سؤال إضافي.
    - لا تطرح أسئلة تتعلق بإجراءات أو معلومات غير مذكورة في المحتوى.
    - ركز على الاجابة ضمن حقل الطالب الا إذا سأل الطالب غير ذلك

    - استخدم أحيانا اسم الطالب بأسلوب المخاطبة في الإجابة بطريقة طبيعية بدون المبالغة أو التكرار.
    - لا تكرر عبارة "يا" أكثر من مرة واحدة في نفس الإجابة.
    - يمكنك استخدام الاسم فقط دون "يا" عندما يكون ذلك أنسب.

    - ركّز فقط على المسار الدراسي الحالي للطالب واغفل أي حقل دراسي آخر موجود في الوثائق أو السياق
    احذر من الخلط بين:
    - "جامعة العلوم والتكنولوجيا الأردنية" (جامعة)
    - "حقل العلوم والتكنولوجيا" (حقل أكاديمي في التوجيهي)

    عند وجود كلمة "حقل" أو "مسار"، المقصود هو المسار الأكاديمي وليس الجامعة.
    عند وجود كلمة "جامعة العلوم والتكنولوجيا"، المقصود هو الجامعة.
    - يمكن في البداية الاجابة بوفقًا لمصادر نورس .

    المحتوى:
    {context}

    
    السياق السابق من المحادثة:
    {context_snippet}

    السؤال:
    {question}
    """

        response = model.generate_content(prompt)
        return response.text.strip()



    def query(self, question: str) -> str:
        # store user query
        self.store_message("user", question)
        user_msg = self.local_stm[-1]  # last stored message

        # build context
        context = self.build_context(question)

        # call LLM
        answer = self.ask_gemini(context, question)

        # store assistant response
        self.store_message("assistant", answer)
        assistant_msg = self.local_stm[-1]

        # Batch upsert both messages together
        self.upsert_conversation_pair(user_msg, assistant_msg)

        return answer


if __name__ == "__main__":
    test_uuid = register_user(
        name="Ayah Test",
        email="ayah_test@example.com",
        password="secure123",
        grade="12",
        school="مدرسة الياسمين",
        path="الأكاديمي",
        field="حقل العلوم والتكنولوجيا"
    )

    if not test_uuid:
        test_uuid = get_user_uuid_by_email("ayah_test@example.com")

    if test_uuid:
        chat = ChatPipeline(user_id=test_uuid)
        print(chat.query("ما المواد التي علي دراستها؟"))
        print(chat.query("نعم"))
