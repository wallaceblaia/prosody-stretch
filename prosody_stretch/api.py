"""
REST API for prosody-stretch.

Provides endpoints for audio duration adjustment via HTTP.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import tempfile
import shutil
from pathlib import Path
import os

app = FastAPI(
    title="Prosody-Stretch API",
    description="Natural audio duration adjustment for dubbing synchronization",
    version="0.1.0"
)


class AdjustmentResult(BaseModel):
    """Response model for adjustment results."""
    original_duration: float
    target_duration: float
    final_duration: float
    error_ms: float
    quality_score: float
    strategies_used: list[str]
    warnings: list[str]


class AudioInfo(BaseModel):
    """Response model for audio info."""
    duration: float
    sample_rate: int
    samples: int
    speech_regions: int
    silence_regions: int
    speech_time: float
    silence_time: float


@app.get("/")
async def root():
    """API health check."""
    return {"status": "ok", "service": "prosody-stretch", "version": "0.1.0"}


@app.post("/adjust", response_model=AdjustmentResult)
async def adjust_duration(
    file: UploadFile = File(...),
    target_duration: float = Form(...),
    text: Optional[str] = Form(None),
    quality_threshold: float = Form(0.5)
):
    """
    Adjust audio duration to target.
    
    - **file**: Audio file (WAV, MP3, etc.)
    - **target_duration**: Target duration in seconds
    - **text**: Optional transcription text
    - **quality_threshold**: Minimum quality (0-1)
    
    Returns the adjusted audio file.
    """
    from prosody_stretch import ProsodyStretcher
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        # Process
        audio, sr = AudioAnalyzer.load(tmp_path)
        original = AudioAnalyzer.get_duration(audio, sr)
        
        stretcher = ProsodyStretcher(quality_threshold=quality_threshold)
        result, report = stretcher.adjust_duration_array(
            audio, sr, target_duration=target_duration, text=text
        )
        
        final = AudioAnalyzer.get_duration(result, sr)
        
        # Save result
        output_path = tmp_path.replace(Path(file.filename).suffix, "_adjusted.wav")
        AudioAnalyzer.save(result, output_path, sr)
        
        # Return file with metadata in headers
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{Path(file.filename).stem}_adjusted.wav",
            headers={
                "X-Original-Duration": str(original),
                "X-Target-Duration": str(target_duration),
                "X-Final-Duration": str(final),
                "X-Error-Ms": str(abs(final - target_duration) * 1000),
                "X-Quality-Score": str(report.quality_score),
                "X-Strategies": ",".join(report.strategies_used),
            }
        )
    finally:
        # Cleanup
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/adjust/json", response_model=AdjustmentResult)
async def adjust_duration_json(
    file: UploadFile = File(...),
    target_duration: float = Form(...),
    text: Optional[str] = Form(None),
    quality_threshold: float = Form(0.5)
):
    """
    Adjust audio duration and return JSON report (without audio file).
    """
    from prosody_stretch import ProsodyStretcher
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        audio, sr = AudioAnalyzer.load(tmp_path)
        original = AudioAnalyzer.get_duration(audio, sr)
        
        stretcher = ProsodyStretcher(quality_threshold=quality_threshold)
        result, report = stretcher.adjust_duration_array(
            audio, sr, target_duration=target_duration, text=text
        )
        
        final = AudioAnalyzer.get_duration(result, sr)
        
        return AdjustmentResult(
            original_duration=original,
            target_duration=target_duration,
            final_duration=final,
            error_ms=abs(final - target_duration) * 1000,
            quality_score=report.quality_score,
            strategies_used=report.strategies_used,
            warnings=report.warnings
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/info", response_model=AudioInfo)
async def audio_info(file: UploadFile = File(...)):
    """
    Get audio file information and analysis.
    """
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    from prosody_stretch.analyzer.silence import SilenceDetector
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    
    try:
        audio, sr = AudioAnalyzer.load(tmp_path)
        duration = AudioAnalyzer.get_duration(audio, sr)
        
        detector = SilenceDetector()
        silences = detector.detect(audio, sr)
        speech_regions = detector.get_speech_segments(audio, sr, silences)
        
        total_silence = sum(s.duration for s in silences)
        total_speech = duration - total_silence
        
        return AudioInfo(
            duration=duration,
            sample_rate=sr,
            samples=len(audio),
            speech_regions=len(speech_regions),
            silence_regions=len(silences),
            speech_time=total_speech,
            silence_time=total_silence
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/match")
async def match_duration(
    source: UploadFile = File(...),
    reference: UploadFile = File(...),
    text: Optional[str] = Form(None)
):
    """
    Match source audio duration to reference audio duration.
    """
    from prosody_stretch import ProsodyStretcher
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    
    # Save both files
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_src:
        shutil.copyfileobj(source.file, tmp_src)
        src_path = tmp_src.name
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_ref:
        shutil.copyfileobj(reference.file, tmp_ref)
        ref_path = tmp_ref.name
    
    try:
        source_audio, source_sr = AudioAnalyzer.load(src_path)
        ref_audio, ref_sr = AudioAnalyzer.load(ref_path)
        
        source_dur = AudioAnalyzer.get_duration(source_audio, source_sr)
        ref_dur = AudioAnalyzer.get_duration(ref_audio, ref_sr)
        
        stretcher = ProsodyStretcher()
        result, report = stretcher.match_duration(
            source_audio=source_audio,
            reference_audio=ref_audio,
            source_sr=source_sr,
            reference_sr=ref_sr,
            text=text
        )
        
        final = AudioAnalyzer.get_duration(result, source_sr)
        
        output_path = src_path.replace(".wav", "_matched.wav")
        AudioAnalyzer.save(result, output_path, source_sr)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=f"{Path(source.filename).stem}_matched.wav",
            headers={
                "X-Source-Duration": str(source_dur),
                "X-Reference-Duration": str(ref_dur),
                "X-Final-Duration": str(final),
                "X-Error-Ms": str(abs(final - ref_dur) * 1000),
                "X-Quality-Score": str(report.quality_score),
            }
        )
    finally:
        for p in [src_path, ref_path]:
            if os.path.exists(p):
                os.unlink(p)
