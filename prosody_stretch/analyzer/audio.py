"""Audio loading, saving, and format utilities."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import soundfile as sf


class AudioAnalyzer:
    """Load, save, and analyze audio files with format support."""
    
    # Supported formats
    SUPPORTED_FORMATS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    
    @staticmethod
    def load(
        path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file (supports WAV, MP3, FLAC, OGG, etc.).
        
        Args:
            path: Path to audio file
            sr: Target sample rate (None = keep original)
            mono: Convert to mono if True
            
        Returns:
            (audio_array, sample_rate)
        """
        path = Path(path)
        
        # Check format
        if path.suffix.lower() not in AudioAnalyzer.SUPPORTED_FORMATS:
            # Try anyway, soundfile/librosa might support it
            pass
        
        try:
            # Try soundfile first (faster, supports most formats)
            audio, orig_sr = sf.read(str(path), dtype='float32')
        except Exception:
            # Fallback to librosa for MP3 and other formats
            try:
                import librosa
                audio, orig_sr = librosa.load(str(path), sr=None, mono=False)
                # librosa returns (samples,) for mono or (channels, samples)
                if audio.ndim > 1:
                    audio = audio.T  # Convert to (samples, channels)
            except Exception as e:
                raise ValueError(f"Cannot load audio file: {path}. Error: {e}")
        
        # Convert to mono if needed
        if mono and audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        
        # Resample if needed
        if sr is not None and sr != orig_sr:
            audio = AudioAnalyzer._resample(audio, orig_sr, sr)
            return audio.astype(np.float32), sr
        
        return audio.astype(np.float32), orig_sr
    
    @staticmethod
    def save(
        audio: np.ndarray,
        path: Union[str, Path],
        sr: int,
        format: Optional[str] = None,
        subtype: Optional[str] = None
    ):
        """
        Save audio to file (supports WAV, MP3, FLAC, OGG).
        
        Args:
            audio: Audio array
            path: Output path
            sr: Sample rate
            format: Override format (e.g., 'WAV', 'MP3')
            subtype: Audio subtype (e.g., 'PCM_16', 'PCM_24')
        """
        path = Path(path)
        ext = path.suffix.lower()
        
        # Ensure audio is in correct range
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
        
        if ext == '.mp3':
            AudioAnalyzer._save_mp3(audio, path, sr)
        elif ext in {'.wav', '.flac', '.ogg'}:
            # Use soundfile
            sf.write(str(path), audio, sr, subtype=subtype, format=format)
        else:
            # Default to WAV
            sf.write(str(path), audio, sr)
    
    @staticmethod
    def _save_mp3(audio: np.ndarray, path: Path, sr: int, bitrate: str = '192k'):
        """Save as MP3 using pydub or ffmpeg."""
        try:
            from pydub import AudioSegment
            
            # Convert to 16-bit PCM
            audio_16 = (audio * 32767).astype(np.int16)
            
            # Create AudioSegment
            if audio_16.ndim == 1:
                channels = 1
            else:
                channels = audio_16.shape[1]
                audio_16 = audio_16.flatten()
            
            segment = AudioSegment(
                audio_16.tobytes(),
                frame_rate=sr,
                sample_width=2,
                channels=channels
            )
            segment.export(str(path), format='mp3', bitrate=bitrate)
        except ImportError:
            # Fallback: save as WAV (ffmpeg can convert later)
            wav_path = path.with_suffix('.wav')
            sf.write(str(wav_path), audio, sr)
            raise ImportError(
                f"pydub not installed for MP3 export. Saved as WAV: {wav_path}. "
                "Install with: pip install pydub"
            )
    
    @staticmethod
    def get_duration(audio: np.ndarray, sr: int) -> float:
        """Get audio duration in seconds."""
        if audio.ndim > 1:
            return audio.shape[0] / sr
        return len(audio) / sr
    
    @staticmethod
    def get_info(path: Union[str, Path]) -> dict:
        """
        Get audio file information without loading full audio.
        
        Returns:
            dict with duration, sample_rate, channels, format
        """
        path = Path(path)
        
        try:
            info = sf.info(str(path))
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'format': info.format,
                'subtype': info.subtype,
            }
        except Exception:
            # Fallback: load and analyze
            audio, sr = AudioAnalyzer.load(path, mono=False)
            channels = 1 if audio.ndim == 1 else audio.shape[1]
            return {
                'duration': AudioAnalyzer.get_duration(audio, sr),
                'sample_rate': sr,
                'channels': channels,
                'format': path.suffix.upper().strip('.'),
                'subtype': None,
            }
    
    @staticmethod
    def convert_sample_rate(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """Convert audio to different sample rate."""
        return AudioAnalyzer._resample(audio, orig_sr, target_sr)
    
    @staticmethod
    def convert_channels(
        audio: np.ndarray,
        target_channels: int
    ) -> np.ndarray:
        """
        Convert between mono and stereo.
        
        Args:
            audio: Input audio
            target_channels: 1 for mono, 2 for stereo
            
        Returns:
            Converted audio
        """
        current_channels = 1 if audio.ndim == 1 else audio.shape[1]
        
        if current_channels == target_channels:
            return audio
        
        if target_channels == 1:
            # Stereo to mono
            if audio.ndim > 1:
                return np.mean(audio, axis=1)
            return audio
        elif target_channels == 2:
            # Mono to stereo
            if audio.ndim == 1:
                return np.column_stack([audio, audio])
            return audio
        else:
            raise ValueError(f"Unsupported channel count: {target_channels}")
    
    @staticmethod
    def get_rms_energy(audio: np.ndarray) -> float:
        """Get overall RMS energy."""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target dB level."""
        peak = np.max(np.abs(audio))
        if peak == 0:
            return audio
        
        target_peak = 10 ** (target_db / 20)
        return audio * (target_peak / peak)
    
    @staticmethod
    def extract_segment(
        audio: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """Extract audio segment by time."""
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        return audio[start_sample:end_sample]
    
    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio using scipy or librosa."""
        if orig_sr == target_sr:
            return audio
        
        try:
            # Try librosa (better quality)
            import librosa
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback to scipy
            from scipy import signal
            num_samples = int(len(audio) * target_sr / orig_sr)
            return signal.resample(audio, num_samples)
    
    @staticmethod
    def crossfade(
        audio1: np.ndarray,
        audio2: np.ndarray,
        overlap_samples: int
    ) -> np.ndarray:
        """Crossfade two audio segments."""
        if overlap_samples <= 0:
            return np.concatenate([audio1, audio2])
        
        overlap_samples = min(overlap_samples, len(audio1), len(audio2))
        
        fade_out = np.linspace(1, 0, overlap_samples)
        fade_in = np.linspace(0, 1, overlap_samples)
        
        part1 = audio1[:-overlap_samples]
        part2 = audio2[overlap_samples:]
        
        crossfade = (
            audio1[-overlap_samples:] * fade_out +
            audio2[:overlap_samples] * fade_in
        )
        
        return np.concatenate([part1, crossfade, part2])
    
    @staticmethod
    def create_silence(duration: float, sr: int) -> np.ndarray:
        """Create silence of specified duration."""
        return np.zeros(int(duration * sr), dtype=np.float32)
