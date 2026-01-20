"""
Command Line Interface for prosody-stretch.

Usage:
    prosody-stretch adjust input.wav --target 15.5 -o output.wav
    prosody-stretch match source.wav reference.wav -o output.wav
    prosody-stretch info input.wav
"""

import click
from pathlib import Path
import sys


@click.group()
@click.version_option(version="0.1.1")
def cli():
    """Prosody-Stretch: Natural audio duration adjustment for dubbing."""
    pass


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-t', '--target', type=float, required=True, 
              help='Target duration in seconds (e.g., 15.87)')
@click.option('-o', '--output', type=click.Path(), default=None,
              help='Output file path (default: input_adjusted.wav)')
@click.option('--text', type=str, default=None,
              help='Transcription text (improves quality)')
@click.option('-q', '--quality', type=float, default=0.5,
              help='Minimum quality threshold (0-1, default: 0.5)')
@click.option('-v', '--verbose', is_flag=True, help='Show detailed output')
@click.option('--sr', '--sample-rate', 'sample_rate', type=int, default=None,
              help='Output sample rate (default: preserve original)')
@click.option('--channels', type=click.Choice(['mono', 'stereo', 'preserve']), 
              default='preserve', help='Output channels (default: preserve)')
@click.option('--format', 'output_format', type=click.Choice(['wav', 'mp3']), 
              default=None, help='Output format (default: from extension)')
def adjust(input_file, target, output, text, quality, verbose, sample_rate, channels, output_format):
    """Adjust audio duration to a specific target time."""
    from prosody_stretch import ProsodyStretcher
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    
    input_path = Path(input_file)
    
    # Determine output path and format
    if output is None:
        ext = f'.{output_format}' if output_format else input_path.suffix
        output = input_path.parent / f"{input_path.stem}_adjusted{ext}"
    else:
        output = Path(output)
    
    # Get original info
    original_info = AudioAnalyzer.get_info(input_path)
    orig_sr = original_info['sample_rate']
    orig_channels = original_info['channels']
    
    # Load audio (keep original properties for processing)
    audio, sr = AudioAnalyzer.load(input_path, sr=None, mono=False)
    original = AudioAnalyzer.get_duration(audio, sr)
    
    # Convert to mono for processing (required by algorithms)
    if audio.ndim > 1:
        audio_mono = AudioAnalyzer.convert_channels(audio, 1)
    else:
        audio_mono = audio
    
    if verbose:
        click.echo(f"📁 Input: {input_path}")
        click.echo(f"📊 Original: {original:.3f}s | {orig_sr}Hz | {orig_channels}ch")
        click.echo(f"🎯 Target duration: {target:.3f}s")
    
    # Process
    stretcher = ProsodyStretcher(quality_threshold=quality)
    result, report = stretcher.adjust_duration_array(
        audio_mono, sr, target_duration=target, text=text
    )
    
    # Convert to target format
    output_sr = sample_rate if sample_rate else orig_sr
    if output_sr != sr:
        result = AudioAnalyzer.convert_sample_rate(result, sr, output_sr)
    
    # Convert channels
    if channels == 'stereo':
        result = AudioAnalyzer.convert_channels(result, 2)
    elif channels == 'mono':
        result = AudioAnalyzer.convert_channels(result, 1)
    elif channels == 'preserve' and orig_channels == 2:
        result = AudioAnalyzer.convert_channels(result, 2)
    
    # Save
    AudioAnalyzer.save(result, output, output_sr)
    
    # Report
    final = AudioAnalyzer.get_duration(result, output_sr)
    error = abs(final - target) * 1000
    change = ((final - original) / original) * 100
    
    if verbose:
        out_channels = 2 if (channels == 'stereo' or (channels == 'preserve' and orig_channels == 2)) else 1
        click.echo(f"\n✅ Result: {final:.3f}s ({change:+.1f}%)")
        click.echo(f"📏 Error: {error:.1f}ms")
        click.echo(f"📈 Quality: {report.quality_score:.2f}")
        click.echo(f"🔧 Strategies: {', '.join(report.strategies_used)}")
        click.echo(f"🎵 Output: {output_sr}Hz | {out_channels}ch | {output.suffix}")
        click.echo(f"💾 Saved: {output}")
        if report.warnings:
            for w in report.warnings:
                click.echo(f"⚠️  {w}")
    else:
        # Compact output (2 lines max)
        click.echo(f"✅ {original:.2f}s → {final:.2f}s ({change:+.1f}%) | Quality: {report.quality_score:.2f}")
        click.echo(f"💾 {output}")


@cli.command()
@click.argument('source_file', type=click.Path(exists=True))
@click.argument('reference_file', type=click.Path(exists=True))
@click.option('-o', '--output', type=click.Path(), default=None,
              help='Output file path')
@click.option('--text', type=str, default=None,
              help='Transcription text')
@click.option('-v', '--verbose', is_flag=True, help='Show detailed output')
@click.option('--sr', '--sample-rate', 'sample_rate', type=int, default=None,
              help='Output sample rate (default: preserve original)')
@click.option('--channels', type=click.Choice(['mono', 'stereo', 'preserve']), 
              default='preserve', help='Output channels')
def match(source_file, reference_file, output, text, verbose, sample_rate, channels):
    """Match source audio duration to reference audio duration."""
    from prosody_stretch import ProsodyStretcher
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    
    source_path = Path(source_file)
    ref_path = Path(reference_file)
    
    if output is None:
        output = source_path.parent / f"{source_path.stem}_matched{source_path.suffix}"
    else:
        output = Path(output)
    
    # Get original info
    source_info = AudioAnalyzer.get_info(source_path)
    orig_sr = source_info['sample_rate']
    orig_channels = source_info['channels']
    
    # Load both
    source_audio, source_sr = AudioAnalyzer.load(source_path, mono=False)
    ref_audio, ref_sr = AudioAnalyzer.load(ref_path)
    
    source_dur = AudioAnalyzer.get_duration(source_audio, source_sr)
    ref_dur = AudioAnalyzer.get_duration(ref_audio, ref_sr)
    
    # Convert to mono for processing
    if source_audio.ndim > 1:
        source_mono = AudioAnalyzer.convert_channels(source_audio, 1)
    else:
        source_mono = source_audio
    
    if verbose:
        click.echo(f"📁 Source: {source_path}")
        click.echo(f"📁 Reference: {ref_path}")
        click.echo(f"📊 Source: {source_dur:.3f}s | Reference: {ref_dur:.3f}s")
    
    # Process
    stretcher = ProsodyStretcher()
    result, report = stretcher.match_duration(
        source_audio=source_mono,
        reference_audio=ref_audio,
        source_sr=source_sr,
        reference_sr=ref_sr,
        text=text
    )
    
    # Convert to target format
    output_sr = sample_rate if sample_rate else orig_sr
    if output_sr != source_sr:
        result = AudioAnalyzer.convert_sample_rate(result, source_sr, output_sr)
    
    # Convert channels
    if channels == 'stereo':
        result = AudioAnalyzer.convert_channels(result, 2)
    elif channels == 'mono':
        result = AudioAnalyzer.convert_channels(result, 1)
    elif channels == 'preserve' and orig_channels == 2:
        result = AudioAnalyzer.convert_channels(result, 2)
    
    # Save
    AudioAnalyzer.save(result, output, output_sr)
    
    final = AudioAnalyzer.get_duration(result, output_sr)
    error = abs(final - ref_dur) * 1000
    
    if verbose:
        click.echo(f"\n✅ Result: {final:.3f}s")
        click.echo(f"📏 Match error: {error:.1f}ms")
        click.echo(f"📈 Quality: {report.quality_score:.2f}")
        click.echo(f"💾 Saved: {output}")
    else:
        click.echo(f"✅ {source_dur:.2f}s → {final:.2f}s (ref: {ref_dur:.2f}s) | Error: {error:.0f}ms")
        click.echo(f"💾 {output}")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-v', '--verbose', is_flag=True, help='Show detailed analysis')
def info(input_file, verbose):
    """Show audio file information and analysis."""
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    from prosody_stretch.analyzer.silence import SilenceDetector
    
    input_path = Path(input_file)
    
    # Get file info
    file_info = AudioAnalyzer.get_info(input_path)
    
    audio, sr = AudioAnalyzer.load(input_path)
    duration = AudioAnalyzer.get_duration(audio, sr)
    
    # Analyze silences
    detector = SilenceDetector()
    silences = detector.detect(audio, sr)
    speech_regions = detector.get_speech_segments(audio, sr, silences)
    
    total_silence = sum(s.duration for s in silences)
    total_speech = duration - total_silence
    
    if verbose:
        click.echo(f"📁 File: {input_path}")
        click.echo(f"📊 Duration: {duration:.3f}s")
        click.echo(f"🎵 Sample rate: {file_info['sample_rate']} Hz")
        click.echo(f"🔊 Channels: {file_info['channels']}")
        click.echo(f"📦 Format: {file_info['format']}")
        if file_info.get('subtype'):
            click.echo(f"📐 Subtype: {file_info['subtype']}")
        click.echo(f"\n🔍 Analysis:")
        click.echo(f"   Speech regions: {len(speech_regions)}")
        click.echo(f"   Speech time: {total_speech:.3f}s ({total_speech/duration*100:.1f}%)")
        click.echo(f"   Silence regions: {len(silences)}")
        click.echo(f"   Silence time: {total_silence:.3f}s ({total_silence/duration*100:.1f}%)")
        
        pause_capacity = sum(s.extensible_amount for s in silences)
        click.echo(f"\n📈 Adjustment capacity:")
        click.echo(f"   Via pauses: ±{pause_capacity:.2f}s")
        click.echo(f"   Via stretch: ±{total_speech*0.4:.2f}s")
        click.echo(f"   Total: ~±{(pause_capacity + total_speech*0.4):.2f}s")
    else:
        click.echo(f"📊 {input_path.name}: {duration:.2f}s | {file_info['sample_rate']}Hz | {file_info['channels']}ch")
        max_adj = sum(s.extensible_amount for s in silences) + total_speech * 0.4
        click.echo(f"📈 Adjustment capacity: ~±{max_adj:.1f}s")


@cli.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.argument('output_file', type=click.Path())
@click.option('--sr', '--sample-rate', 'sample_rate', type=int, default=None,
              help='Target sample rate')
@click.option('--channels', type=click.Choice(['mono', 'stereo']), 
              default=None, help='Target channels')
def convert(input_file, output_file, sample_rate, channels):
    """Convert audio file format, sample rate, or channels."""
    from prosody_stretch.analyzer.audio import AudioAnalyzer
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    # Load
    audio, sr = AudioAnalyzer.load(input_path, mono=False)
    orig_info = AudioAnalyzer.get_info(input_path)
    
    # Convert sample rate
    output_sr = sample_rate if sample_rate else sr
    if output_sr != sr:
        audio = AudioAnalyzer.convert_sample_rate(audio, sr, output_sr)
    
    # Convert channels
    if channels == 'mono':
        audio = AudioAnalyzer.convert_channels(audio, 1)
    elif channels == 'stereo':
        audio = AudioAnalyzer.convert_channels(audio, 2)
    
    # Save
    AudioAnalyzer.save(audio, output_path, output_sr)
    
    out_channels = 1 if channels == 'mono' else (2 if channels == 'stereo' else orig_info['channels'])
    click.echo(f"✅ {input_path.name} → {output_path.name}")
    click.echo(f"🎵 {orig_info['sample_rate']}Hz/{orig_info['channels']}ch → {output_sr}Hz/{out_channels}ch")


@cli.command()
@click.option('-h', '--host', default='0.0.0.0', help='Host to bind')
@click.option('-p', '--port', default=8000, help='Port to bind')
def serve(host, port):
    """Start the REST API server."""
    try:
        import uvicorn
        from prosody_stretch.api import app
        click.echo(f"🚀 Starting API server on http://{host}:{port}")
        uvicorn.run(app, host=host, port=port)
    except ImportError:
        click.echo("❌ API dependencies not installed. Run: pip install prosody-stretch[api]")
        sys.exit(1)


def main():
    cli()


if __name__ == '__main__':
    main()
