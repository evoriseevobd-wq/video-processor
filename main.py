import os
import json
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

JOBS_DIR = Path("/app/jobs")
JOBS_DIR.mkdir(exist_ok=True)

whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    return whisper_model

class UrlRequest(BaseModel):
    youtube_url: str
    job_id: str = None

class JobRequest(BaseModel):
    job_id: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/metadados")
def metadados(req: UrlRequest):
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", req.youtube_url],
        capture_output=True, text=True, timeout=30
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=result.stderr)
    data = json.loads(result.stdout)
    return {
        "titulo": data.get("title"),
        "canal": data.get("uploader"),
        "duracao_segundos": data.get("duration"),
        "duracao_minutos": round(data.get("duration", 0) / 60, 1),
        "video_id": data.get("id")
    }

@app.post("/baixar")
def baixar(req: UrlRequest):
    job_id = req.job_id or str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    output_path = str(job_dir / "video_original.mp4")
    result = subprocess.run(
        ["yt-dlp", "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
         "-o", output_path, req.youtube_url],
        capture_output=True, text=True, timeout=3600
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=result.stderr)
    return {"job_id": job_id, "arquivo": output_path}

@app.post("/extrair-audio")
def extrair_audio(req: JobRequest):
    job_dir = JOBS_DIR / req.job_id
    video_path = job_dir / "video_original.mp4"
    audio_path = job_dir / "audio.wav"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path), "-ar", "16000", "-ac", "1", "-y", str(audio_path)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=result.stderr)
    return {"job_id": req.job_id, "arquivo": str(audio_path)}

@app.post("/transcrever")
def transcrever(req: JobRequest):
    job_dir = JOBS_DIR / req.job_id
    audio_path = job_dir / "audio.wav"
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Áudio não encontrado")
    model = get_whisper_model()
    segments, info = model.transcribe(str(audio_path), beam_size=5, language="pt")
    transcricao = []
    texto_completo = ""
    for segment in segments:
        inicio = format_time(segment.start)
        fim = format_time(segment.end)
        transcricao.append({
            "inicio": inicio,
            "fim": fim,
            "texto": segment.text.strip()
        })
        texto_completo += f"[{inicio} - {fim}] {segment.text.strip()}\n"
    transcricao_path = job_dir / "transcricao.json"
    with open(transcricao_path, "w", encoding="utf-8") as f:
        json.dump({
            "segments": transcricao,
            "texto_formatado": texto_completo
        }, f, ensure_ascii=False, indent=2)
    return {
        "job_id": req.job_id,
        "idioma": info.language,
        "segmentos": len(transcricao),
        "texto_formatado": texto_completo,
        "transcricao": transcricao
    }

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"
