import os
# ==========================================
# 1. Optimasi & Pengaturan Environment
# ==========================================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from contextlib import asynccontextmanager
import time
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from deepface import DeepFace
import numpy as np
import cv2
from pathlib import Path
from psycopg2 import OperationalError, pool

BASE_DIR = Path(__file__).resolve().parent
# DATABASE_URL = os.getenv("DATABASE_URL")
DATABASE_URL = "postgresql://root:1234@127.0.0.1:32768/facedb"
FACE_MATCH_THRESHOLD = 0.62
FACE_VECTOR_DIM = 512
MAX_IMAGE_BYTES = 3 * 1024 * 1024
MAX_IMAGE_SIDE = 960
MIN_BRIGHTNESS = 45
MAX_BRIGHTNESS = 210
MIN_SHARPNESS = float(os.getenv("MIN_SHARPNESS", "20.0"))
DB_MAX_CONN = int(os.getenv("DB_MAX_CONN", "6"))
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "200"))
db_pool: pool.SimpleConnectionPool | None = None

def _get_connection():
    if db_pool is None:
        raise RuntimeError("Database pool belum diinisialisasi.")
    return db_pool.getconn()

def _release_connection(conn):
    if db_pool is not None:
        db_pool.putconn(conn)

def init_db_schema():
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE EXTENSION IF NOT EXISTS vector;
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS face_embeddings (
                    name TEXT PRIMARY KEY,
                    embedding VECTOR(512) NOT NULL,
                    sample_count INTEGER NOT NULL DEFAULT 1,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                ALTER TABLE face_embeddings
                ADD COLUMN IF NOT EXISTS sample_count INTEGER NOT NULL DEFAULT 1;
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS face_embeddings_embedding_hnsw_cosine_idx
                ON face_embeddings
                USING hnsw (embedding vector_cosine_ops);
                """
            )
            conn.commit()
    finally:
        _release_connection(conn)

def _l2_normalize(embedding):
    embedding_array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(embedding_array)
    if norm <= 0.0:
        raise ValueError("Embedding tidak valid untuk normalisasi.")
    return embedding_array / norm

def _vector_literal(embedding) -> str:
    embedding_array = _l2_normalize(embedding)
    if embedding_array.size != FACE_VECTOR_DIM:
        raise ValueError(
            f"Panjang embedding tidak valid. Diharapkan {FACE_VECTOR_DIM}, diterima {embedding_array.size}."
        )
    return "[" + ",".join(f"{float(value):.6f}" for value in embedding_array) + "]"

def _db_vector_to_numpy(db_vector):
    if isinstance(db_vector, str):
        text = db_vector.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        if not text:
            raise ValueError("Embedding database kosong.")
        values = [float(v) for v in text.split(",")]
        return np.asarray(values, dtype=np.float32).reshape(-1)
    return np.asarray(db_vector, dtype=np.float32).reshape(-1)

def upsert_face_embedding(name: str, embedding) -> None:
    new_vector = _l2_normalize(embedding)
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT embedding, sample_count
                FROM face_embeddings
                WHERE name = %s
                FOR UPDATE;
                """,
                (name,),
            )
            existing = cur.fetchone()

            if existing is None:
                cur.execute(
                    """
                    INSERT INTO face_embeddings (name, embedding, sample_count)
                    VALUES (%s, %s::vector, 1);
                    """,
                    (name, _vector_literal(new_vector)),
                )
            else:
                existing_vector = _db_vector_to_numpy(existing[0])
                existing_count = int(existing[1]) if existing[1] is not None else 1
                merged = ((existing_vector * existing_count) + new_vector) / (existing_count + 1)
                cur.execute(
                    """
                    UPDATE face_embeddings
                    SET embedding = %s::vector,
                        sample_count = %s,
                        updated_at = NOW()
                    WHERE name = %s;
                    """,
                    (_vector_literal(merged), existing_count + 1, name),
                )
            conn.commit()
    finally:
        _release_connection(conn)

def find_best_match(embedding):
    vector_literal = _vector_literal(embedding)
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL hnsw.ef_search = %s;", (HNSW_EF_SEARCH,))
            cur.execute(
                """
                SELECT
                    name,
                    (embedding <=> %s::vector) AS distance,
                    sample_count
                FROM face_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT 1;
                """,
                (vector_literal, vector_literal),
            )
            return cur.fetchone()
    finally:
        _release_connection(conn)

def get_all_users_summary():
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT name, sample_count
                FROM face_embeddings
                ORDER BY name ASC;
                """
            )
            return cur.fetchall()
    finally:
        _release_connection(conn)

def count_users():
    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM face_embeddings;")
            result = cur.fetchone()
            return int(result[0]) if result else 0
    finally:
        _release_connection(conn)

# ==========================================
# 2. LIFESPAN (Pre-load AI Models)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    print("Pre-loading AI Models (ArcFace & RetinaFace)...")
    try:
        # Load model Face Recognition (ArcFace)
        DeepFace.build_model("ArcFace")
        
        # PERBAIKAN: Warm-up Face Detector (RetinaFace) menggunakan dummy image
        print("Warming up RetinaFace detector...")
        dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
        try:
            DeepFace.extract_faces(
                img_path=dummy_img, 
                detector_backend="retinaface", 
                enforce_detection=False 
            )
        except Exception:
            pass
            
        print("Models berhasil dimuat! Server siap.")
    except Exception as e:
        print(f"Error saat memuat model: {e}")

    print("Menghubungkan ke PostgreSQL...")
    last_error = None
    for attempt in range(1, 11):
        try:
            db_pool = pool.SimpleConnectionPool(minconn=1, maxconn=DB_MAX_CONN, dsn=DATABASE_URL)
            init_db_schema()
            print("PostgreSQL berhasil terhubung dan schema siap.")
            last_error = None
            break
        except OperationalError as e:
            last_error = e
            print(f"Gagal koneksi DB (percobaan {attempt}/10): {e}")
            time.sleep(2)

    if last_error is not None:
        raise RuntimeError(f"Tidak dapat terhubung ke PostgreSQL: {last_error}")

    yield
    if db_pool is not None:
        db_pool.closeall()

app = FastAPI(title="Sistem Face ID Optimasi", lifespan=lifespan)
MODELS_DIR = BASE_DIR / "models"

if MODELS_DIR.exists():
    app.mount("/models", StaticFiles(directory=MODELS_DIR), name="models")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 5. FUNGSI HELPER (Sinkron)
# ==========================================
def decode_bytes_to_image(image_bytes: bytes):
    if not image_bytes:
        raise ValueError("File gambar kosong.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"Ukuran gambar melebihi batas {MAX_IMAGE_BYTES // (1024 * 1024)} MB.")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Gambar tidak valid atau format tidak didukung.")

    height, width = img.shape[:2]
    max_side = max(height, width)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / float(max_side)
        img = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return img

def validate_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < MIN_BRIGHTNESS or brightness > MAX_BRIGHTNESS:
        raise ValueError("Pencahayaan wajah kurang ideal. Gunakan cahaya yang cukup dan merata.")
    if sharpness < MIN_SHARPNESS:
        raise ValueError("Gambar terlalu blur. Mohon dekatkan wajah ke kamera dan tahan 1-2 detik.")

async def read_upload_image(image: UploadFile):
    if image.content_type is None or not image.content_type.startswith("image/"):
        raise ValueError("File harus berupa gambar.")
    image_bytes = await image.read()
    img = decode_bytes_to_image(image_bytes)
    validate_image_quality(img)
    return img

def get_face_vector(img):
    """Fungsi berat ini akan dijalankan di thread terpisah"""
    results = DeepFace.represent(
        img_path=img, 
        model_name="ArcFace", 
        enforce_detection=True,
        detector_backend="retinaface",
        normalization="ArcFace",
        align=True
    )
    if len(results) > 1:
        raise ValueError("Terdeteksi lebih dari satu wajah. Pastikan hanya ada satu orang.")
    return results[0]["embedding"]

# ==========================================
# 6. ENDPOINTS API
# ==========================================

@app.get("/", include_in_schema=False)
async def web_index():
    return FileResponse(BASE_DIR / "index.html")

@app.post("/api/v1/register")
async def register_face(
    name: str = Form(...),
    image: UploadFile = File(...),
):
    try:
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Nama tidak boleh kosong.")

        img = await read_upload_image(image)
        vector = await run_in_threadpool(get_face_vector, img)
        await run_in_threadpool(upsert_face_embedding, normalized_name, vector)
        total_users = await run_in_threadpool(count_users)

        return {
            "status": "success",
            "message": f"Wajah {normalized_name} berhasil didaftarkan!",
            "total_users": total_users
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/recognize")
async def recognize_face(
    image: UploadFile = File(...),
):
    try:
        img = await read_upload_image(image)
        query_vector = await run_in_threadpool(get_face_vector, img)
        best_match = await run_in_threadpool(find_best_match, query_vector)
        if best_match is None:
            raise HTTPException(status_code=400, detail="Database kosong.")

        best_match_name = best_match[0]
        min_distance = float(best_match[1])
        sample_count = int(best_match[2]) if best_match[2] is not None else 1
        adaptive_threshold = FACE_MATCH_THRESHOLD if sample_count >= 3 else FACE_MATCH_THRESHOLD - 0.04
        is_match = min_distance <= adaptive_threshold

        if is_match:
            return {
                "status": "success",
                "match_found": True,
                "name": best_match_name,
                "distance": round(min_distance, 4),
                "sample_count": sample_count,
                "message": "Wajah berhasil dikenali."
            }

        return {
            "status": "failed",
            "match_found": False,
            "name": "Unknown",
            "distance": round(min_distance, 4),
            "sample_count": sample_count,
            "message": "Wajah tidak ditemukan pada database."
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/users")
async def get_all_users():
    users = await run_in_threadpool(get_all_users_summary)
    return {
        "total_users": len(users),
        "registered_names": [name for name, _ in users],
        "samples_per_user": {name: sample_count for name, sample_count in users}
    }

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}