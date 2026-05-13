import os
import json
import uuid
import subprocess
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File as FastAPIFile
from fastapi.responses import FileResponse
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

class CortarRequest(BaseModel):
    job_id: str
    corte_id: str
    inicio: str
    fim: str

class RenderizarRequest(BaseModel):
    job_id: str
    corte_id: str

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/metadados")
def metadados(req: UrlRequest):
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download",
        "--js-runtimes", "node",
        "--cookies", "/app/cookies.txt",
         req.youtube_url],
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
          "--js-runtimes", "node",
          "--cookies", "/app/cookies.txt",
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
    segments, info = model.transcribe(str(audio_path), beam_size=5, language="pt", word_timestamps=True)
    transcricao = []
    texto_completo = ""
    for segment in segments:
        inicio = format_time(segment.start)
        fim = format_time(segment.end)
        words = []
        if hasattr(segment, 'words') and segment.words:
            for word in segment.words:
                words.append({
                    "word": word.word.strip(),
                    "inicio": word.start,
                    "fim": word.end
                })
        transcricao.append({
            "inicio": inicio,
            "fim": fim,
            "texto": segment.text.strip(),
            "words": words
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

@app.post("/cortar")
def cortar(req: CortarRequest):
    job_dir = JOBS_DIR / req.job_id
    video_path = job_dir / "video_original.mp4"
    corte_path = job_dir / f"corte_{req.corte_id}_bruto.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Vídeo não encontrado")
    result = subprocess.run(
        ["ffmpeg", "-i", str(video_path),
         "-ss", req.inicio, "-to", req.fim,
         "-c:v", "libx264", "-c:a", "aac", "-y",
         str(corte_path)],
        capture_output=True, text=True, timeout=600
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=result.stderr)
    return {"job_id": req.job_id, "corte_id": req.corte_id, "arquivo": str(corte_path)}

@app.post("/gerar-legenda")
def gerar_legenda(req: CortarRequest):
    job_dir = JOBS_DIR / req.job_id
    transcricao_path = job_dir / "transcricao.json"
    if not transcricao_path.exists():
        raise HTTPException(status_code=404, detail="Transcrição não encontrada")
    with open(transcricao_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    segments = data["segments"]
    inicio_seg = time_to_seconds(req.inicio)
    fim_seg = time_to_seconds(req.fim)
    corte_segments = [s for s in segments
                      if time_to_seconds(s["inicio"]) >= inicio_seg - 1
                      and time_to_seconds(s["fim"]) <= fim_seg + 1]

    # Gera ASS com destaque por palavra
    ass_header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,28,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,10,10,20,1
Style: Highlight,Arial,28,&H0000FFFF,&H000000FF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,10,10,20,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    ass_events = ""
    for seg in corte_segments:
        seg_inicio = max(0, time_to_seconds(seg["inicio"]) - inicio_seg)
        seg_fim = max(0, time_to_seconds(seg["fim"]) - inicio_seg)
        words = seg.get("words", [])
        if words:
            for word in words:
                w_inicio = max(0, word["inicio"] - inicio_seg)
                w_fim = max(0, word["fim"] - inicio_seg)
                # Linha com palavra destacada
                before = " ".join(w["word"] for w in words if w["fim"] <= word["inicio"])
                after = " ".join(w["word"] for w in words if w["inicio"] >= word["fim"])
                highlighted = word["word"]
                text = ""
                if before:
                    text += f"{{\\rDefault}}{before} "
                text += f"{{\\rHighlight}}{highlighted}"
                if after:
                    text += f"{{\\rDefault}} {after}"
                ass_events += f"Dialogue: 0,{seconds_to_ass(w_inicio)},{seconds_to_ass(w_fim)},Default,,0,0,0,,{text}\n"
        else:
            text = seg["texto"]
            ass_events += f"Dialogue: 0,{seconds_to_ass(seg_inicio)},{seconds_to_ass(seg_fim)},Default,,0,0,0,,{text}\n"

    ass_content = ass_header + ass_events
    ass_path = job_dir / f"corte_{req.corte_id}.ass"
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)

    # Mantém SRT como fallback
    srt_content = ""
    for i, seg in enumerate(corte_segments, 1):
        seg_inicio = max(0, time_to_seconds(seg["inicio"]) - inicio_seg)
        seg_fim = max(0, time_to_seconds(seg["fim"]) - inicio_seg)
        srt_content += f"{i}\n{seconds_to_srt(seg_inicio)} --> {seconds_to_srt(seg_fim)}\n{seg['texto']}\n\n"
    srt_path = job_dir / f"corte_{req.corte_id}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return {"job_id": req.job_id, "corte_id": req.corte_id, "arquivo_ass": str(ass_path), "arquivo_srt": str(srt_path)}

@app.post("/renderizar")
def renderizar(req: RenderizarRequest):
    import cv2
    import numpy as np

    job_dir = JOBS_DIR / req.job_id
    corte_path = job_dir / f"corte_{req.corte_id}_bruto.mp4"
    ass_path = job_dir / f"corte_{req.corte_id}.ass"
    srt_path = job_dir / f"corte_{req.corte_id}.srt"
    output_path = job_dir / f"corte_{req.corte_id}_final.mp4"
    if not corte_path.exists():
        raise HTTPException(status_code=404, detail="Corte bruto não encontrado")

    # Detecta rosto a cada 15 frames
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(str(corte_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    face_centers_x = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 15 == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                x, w = faces[0][0], faces[0][2]
                face_centers_x.append(x + w // 2)
        frame_count += 1
    cap.release()

    # Posição X do zoom — usa centro do rosto ou centro do vídeo
    if face_centers_x:
        avg_x = int(sum(face_centers_x) / len(face_centers_x))
        crop_x = max(0, min(avg_x - 540, width - 1080))
    else:
        crop_x = max(0, (width - 1080) // 2)

    # Duração para CTA
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(corte_path)],
        capture_output=True, text=True
    )
    duration = float(probe.stdout.strip()) if probe.returncode == 0 else 60
    cta_start = max(0, duration - 3)

    cta = (
        f"drawtext=text='Segue para mais conteudo':"
        f"fontsize=26:fontcolor=white:"
        f"x=(w-text_w)/2:y=h-100:"
        f"alpha='if(gte(t,{cta_start}),min(1,(t-{cta_start})/0.5),0)':"
        f"borderw=3:bordercolor=black"
    )

    legend_file = ass_path if ass_path.exists() else srt_path if srt_path.exists() else None

    if legend_file:
        fc = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,gblur=sigma=20[bg];"
            f"[0:v]crop={min(width,height)}:{min(width,height)}:{crop_x}:0,"
            f"scale=1080:1080[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
            f"[base]subtitles={str(legend_file)}:force_style="
            f"'FontSize=8,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
            f"Outline=2,Alignment=2,MarginV=20'[subbed];"
            f"[subbed]{cta}[out]"
        )
    else:
        fc = (
            f"[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
            f"crop=1080:1920,gblur=sigma=20[bg];"
            f"[0:v]crop={min(width,height)}:{min(width,height)}:{crop_x}:0,"
            f"scale=1080:1080[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[base];"
            f"[base]{cta}[out]"
        )

    result = subprocess.run(
        ["ffmpeg", "-i", str(corte_path),
         "-filter_complex", fc,
         "-map", "[out]", "-map", "0:a",
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-c:a", "aac", "-b:a", "128k", "-y",
         str(output_path)],
        capture_output=True, text=True, timeout=1800
    )
    if result.returncode != 0:
        raise HTTPException(status_code=400, detail=result.stderr)
    return {
        "job_id": req.job_id,
        "corte_id": req.corte_id,
        "arquivo_final": str(output_path)
    }

@app.get("/download/{job_id}/{corte_id}")
def download(job_id: str, corte_id: str):
    job_dir = JOBS_DIR / job_id
    arquivo = job_dir / f"corte_{corte_id}_final.mp4"
    if not arquivo.exists():
        raise HTTPException(status_code=404, detail="Arquivo não encontrado")
    return FileResponse(
        path=str(arquivo),
        media_type="video/mp4",
        filename=f"corte_{corte_id}_final.mp4"
    )

@app.post("/atualizar-cookies")
async def atualizar_cookies(file: UploadFile = FastAPIFile(...)):
    conteudo = await file.read()
    with open("/app/cookies.txt", "wb") as f:
        f.write(conteudo)
    return {"status": "cookies atualizados"}

def format_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def time_to_seconds(time_str: str) -> float:
    parts = time_str.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

def seconds_to_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def seconds_to_ass(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds - int(seconds)) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
