"""
Bilingual Audiobook Assembly

Combines alternating source/target sentence audio files with pauses and gaps
for language learning audiobooks.

Structure: [Source₀] → [pause] → [Target₀] → [gap] → [Source₁] → [pause] → [Target₁] → ...
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path


def generate_silence(duration: float, sample_rate: int = 24000, output_path: str = None) -> str:
    """
    Generate a silent audio file of specified duration.

    Args:
        duration: Duration in seconds
        sample_rate: Audio sample rate (default 24000 for XTTS compatibility)
        output_path: Optional output path. If None, creates temp file.

    Returns:
        Path to the generated silence file
    """
    if output_path is None:
        # Create temp file with .flac extension
        fd, output_path = tempfile.mkstemp(suffix='.flac')
        os.close(fd)

    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found in PATH")

    cmd = [
        ffmpeg, '-y',
        '-f', 'lavfi',
        '-i', f'anullsrc=r={sample_rate}:cl=mono',
        '-t', str(duration),
        '-c:a', 'flac',
        output_path
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to generate silence: {result.stderr}")

    return output_path


def get_audio_duration(filepath: str) -> float:
    """Get audio file duration using ffprobe."""
    ffprobe = shutil.which('ffprobe')
    if not ffprobe:
        return 0.0

    cmd = [
        ffprobe, '-v', 'quiet',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        filepath
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception:
        pass

    return 0.0


def combine_bilingual_audio(
    sentences_dir: str,
    output_file: str,
    num_sentences: int,
    pause_duration: float = 0.3,
    gap_duration: float = 1.0,
    sample_rate: int = 24000,
    audio_format: str = 'flac'
) -> bool:
    """
    Combine sentence audio files in bilingual pairs with pauses and gaps.

    DEPRECATED: Use combine_dual_voice_audio for new dual-EPUB workflow.

    Structure: [Source₀] → [pause] → [Target₀] → [gap] → [Source₁] → ...

    This version assumes interleaved sentences in ONE directory:
    - 0.flac = source[0], 1.flac = target[0], 2.flac = source[1], etc.

    Args:
        sentences_dir: Directory containing numbered sentence audio files (0.flac, 1.flac, etc.)
        output_file: Path for the combined output file
        num_sentences: Total number of sentence files
        pause_duration: Seconds of silence between source and target
        gap_duration: Seconds of silence between pairs
        sample_rate: Audio sample rate for silence generation
        audio_format: Audio file extension (default: flac)

    Returns:
        True if successful, False otherwise
    """
    print(f"[BILINGUAL] Combining {num_sentences} sentences with pause={pause_duration}s, gap={gap_duration}s")

    # Validate we have an even number of sentences
    if num_sentences % 2 != 0:
        print(f"[BILINGUAL] Warning: Odd number of sentences ({num_sentences}), last one will be skipped")
        num_sentences = num_sentences - 1

    if num_sentences < 2:
        print("[BILINGUAL] Error: Need at least 2 sentences for bilingual mode")
        return False

    # Create temporary silence files
    temp_dir = tempfile.mkdtemp(prefix='bilingual_')
    try:
        pause_file = os.path.join(temp_dir, 'pause.flac')
        gap_file = os.path.join(temp_dir, 'gap.flac')

        print("[BILINGUAL] Generating silence files...")
        generate_silence(pause_duration, sample_rate, pause_file)
        generate_silence(gap_duration, sample_rate, gap_file)

        # Build concat list
        concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
        num_pairs = num_sentences // 2

        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for pair_idx in range(num_pairs):
                source_idx = pair_idx * 2
                target_idx = pair_idx * 2 + 1

                source_file = os.path.join(sentences_dir, f"{source_idx}.{audio_format}")
                target_file = os.path.join(sentences_dir, f"{target_idx}.{audio_format}")

                # Check files exist
                if not os.path.exists(source_file):
                    print(f"[BILINGUAL] Warning: Missing source file {source_file}")
                    continue
                if not os.path.exists(target_file):
                    print(f"[BILINGUAL] Warning: Missing target file {target_file}")
                    continue

                # Use forward slashes for ffmpeg compatibility (works on Windows too)
                source_path = source_file.replace(os.sep, '/')
                target_path = target_file.replace(os.sep, '/')
                pause_path = pause_file.replace(os.sep, '/')
                gap_path = gap_file.replace(os.sep, '/')

                # Write: source → pause → target → gap (except no gap after last pair)
                f.write(f"file '{source_path}'\n")
                f.write(f"file '{pause_path}'\n")
                f.write(f"file '{target_path}'\n")

                if pair_idx < num_pairs - 1:
                    f.write(f"file '{gap_path}'\n")

        # Run ffmpeg concat
        print("[BILINGUAL] Running ffmpeg concat...")
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            print("[BILINGUAL] Error: ffmpeg not found")
            return False

        cmd = [
            ffmpeg, '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c:a', 'flac',
            output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[BILINGUAL] FFmpeg error: {result.stderr}")
            return False

        print(f"[BILINGUAL] Combined audio saved to {output_file}")
        return True

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[BILINGUAL] Warning: Failed to cleanup temp dir: {e}")


def combine_dual_voice_audio(
    source_sentences_dir: str,
    target_sentences_dir: str,
    output_file: str,
    num_pairs: int,
    pause_duration: float = 0.3,
    gap_duration: float = 1.0,
    sample_rate: int = 24000,
    audio_format: str = 'flac'
) -> bool:
    """
    Combine audio from TWO separate sentence directories (dual-voice workflow).

    Structure: [Source₀] → [pause] → [Target₀] → [gap] → [Source₁] → ...

    This version uses TWO directories for proper accent separation:
    - source_sentences_dir/1.flac, 2.flac, 3.flac... (source language, skips 0.flac)
    - target_sentences_dir/1.flac, 2.flac, 3.flac... (target language, skips 0.flac)

    NOTE: Skips audio file 0.flac because bilingual EPUBs prepend "bookforge" as sentence 0,
    which e2a uses as the chapter title. The actual content starts at sentence 1.

    Args:
        source_sentences_dir: Directory with source language sentences (1.flac, 2.flac, ...)
        target_sentences_dir: Directory with target language sentences (1.flac, 2.flac, ...)
        output_file: Path for the combined output file
        num_pairs: Number of sentence pairs (from sentence_pairs.json)
        pause_duration: Seconds of silence between source and target
        gap_duration: Seconds of silence between pairs
        sample_rate: Audio sample rate for silence generation
        audio_format: Audio file extension (default: flac)

    Returns:
        True if successful, False otherwise
    """
    print(f"[BILINGUAL-DUAL] Combining {num_pairs} pairs from dual directories")
    print(f"[BILINGUAL-DUAL] Source: {source_sentences_dir}")
    print(f"[BILINGUAL-DUAL] Target: {target_sentences_dir}")
    print(f"[BILINGUAL-DUAL] pause={pause_duration}s, gap={gap_duration}s")
    print(f"[BILINGUAL-DUAL] Skipping audio file 0 (bookforge chapter marker)")

    if num_pairs < 1:
        print("[BILINGUAL-DUAL] Error: Need at least 1 pair")
        return False

    # Create temporary silence files
    temp_dir = tempfile.mkdtemp(prefix='bilingual_dual_')
    try:
        pause_file = os.path.join(temp_dir, 'pause.flac')
        gap_file = os.path.join(temp_dir, 'gap.flac')

        print("[BILINGUAL-DUAL] Generating silence files...")
        generate_silence(pause_duration, sample_rate, pause_file)
        generate_silence(gap_duration, sample_rate, gap_file)

        # Build concat list
        concat_list_path = os.path.join(temp_dir, 'concat_list.txt')
        valid_pairs = 0

        with open(concat_list_path, 'w', encoding='utf-8') as f:
            for pair_idx in range(num_pairs):
                # Skip audio file 0 (bookforge chapter marker) - actual content starts at file 1
                audio_idx = pair_idx + 1
                source_file = os.path.join(source_sentences_dir, f"{audio_idx}.{audio_format}")
                target_file = os.path.join(target_sentences_dir, f"{audio_idx}.{audio_format}")

                # Check files exist
                if not os.path.exists(source_file):
                    print(f"[BILINGUAL-DUAL] Warning: Missing source file {source_file}")
                    continue
                if not os.path.exists(target_file):
                    print(f"[BILINGUAL-DUAL] Warning: Missing target file {target_file}")
                    continue

                # Use forward slashes for ffmpeg compatibility
                source_path = source_file.replace(os.sep, '/')
                target_path = target_file.replace(os.sep, '/')
                pause_path = pause_file.replace(os.sep, '/')
                gap_path = gap_file.replace(os.sep, '/')

                # Write: source → pause → target → gap (except no gap after last pair)
                f.write(f"file '{source_path}'\n")
                f.write(f"file '{pause_path}'\n")
                f.write(f"file '{target_path}'\n")

                if pair_idx < num_pairs - 1:
                    f.write(f"file '{gap_path}'\n")

                valid_pairs += 1

        if valid_pairs == 0:
            print("[BILINGUAL-DUAL] Error: No valid pairs found")
            return False

        print(f"[BILINGUAL-DUAL] Found {valid_pairs}/{num_pairs} valid pairs")

        # Run ffmpeg concat
        print("[BILINGUAL-DUAL] Running ffmpeg concat...")
        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            print("[BILINGUAL-DUAL] Error: ffmpeg not found")
            return False

        cmd = [
            ffmpeg, '-y',
            '-f', 'concat',
            '-safe', '0',
            '-i', concat_list_path,
            '-c:a', 'flac',
            output_file
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[BILINGUAL-DUAL] FFmpeg error: {result.stderr}")
            return False

        print(f"[BILINGUAL-DUAL] Combined audio saved to {output_file}")
        return True

    finally:
        # Cleanup temp directory
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[BILINGUAL-DUAL] Warning: Failed to cleanup temp dir: {e}")


def build_bilingual_vtt(
    sentences_dir: str,
    sentence_texts: list,
    output_path: str,
    pause_duration: float = 0.3,
    gap_duration: float = 1.0,
    audio_format: str = 'flac'
) -> bool:
    """
    Build a VTT subtitle file for bilingual audio with proper timing.

    DEPRECATED: Use build_dual_voice_vtt for new dual-EPUB workflow.

    Args:
        sentences_dir: Directory containing numbered sentence audio files
        sentence_texts: List of sentence texts in order [source0, target0, source1, target1, ...]
        output_path: Path for the output VTT file
        pause_duration: Pause duration between source and target
        gap_duration: Gap duration between pairs
        audio_format: Audio file extension

    Returns:
        True if successful, False otherwise
    """
    print(f"[BILINGUAL-VTT] Building VTT with {len(sentence_texts)} sentences")

    num_sentences = len(sentence_texts)
    if num_sentences % 2 != 0:
        print(f"[BILINGUAL-VTT] Warning: Odd number of sentences, truncating")
        num_sentences = num_sentences - 1
        sentence_texts = sentence_texts[:num_sentences]

    if num_sentences < 2:
        print("[BILINGUAL-VTT] Error: Need at least 2 sentences")
        return False

    def format_timestamp(seconds: float) -> str:
        """Format seconds as VTT timestamp HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    # Get durations for all sentence files
    print("[BILINGUAL-VTT] Getting audio durations...")
    durations = []
    for i in range(num_sentences):
        filepath = os.path.join(sentences_dir, f"{i}.{audio_format}")
        duration = get_audio_duration(filepath)
        durations.append(duration)
        if duration == 0:
            print(f"[BILINGUAL-VTT] Warning: Zero duration for sentence {i}")

    # Build VTT cues
    vtt_lines = ["WEBVTT", ""]
    current_time = 0.0
    num_pairs = num_sentences // 2

    for pair_idx in range(num_pairs):
        source_idx = pair_idx * 2
        target_idx = pair_idx * 2 + 1

        source_text = sentence_texts[source_idx] if source_idx < len(sentence_texts) else ""
        target_text = sentence_texts[target_idx] if target_idx < len(sentence_texts) else ""

        source_duration = durations[source_idx]
        target_duration = durations[target_idx]

        # Source cue
        source_start = current_time
        source_end = current_time + source_duration
        vtt_lines.append(f"{format_timestamp(source_start)} --> {format_timestamp(source_end)}")
        vtt_lines.append(source_text)
        vtt_lines.append("")

        current_time = source_end + pause_duration

        # Target cue
        target_start = current_time
        target_end = current_time + target_duration
        vtt_lines.append(f"{format_timestamp(target_start)} --> {format_timestamp(target_end)}")
        vtt_lines.append(target_text)
        vtt_lines.append("")

        current_time = target_end + gap_duration

    # Write VTT file
    print(f"[BILINGUAL-VTT] Writing {num_pairs * 2} cues to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_lines))
        return True
    except Exception as e:
        print(f"[BILINGUAL-VTT] Error writing VTT: {e}")
        return False


def build_dual_voice_vtt(
    source_sentences_dir: str,
    target_sentences_dir: str,
    sentence_pairs: list,
    output_path: str,
    pause_duration: float = 0.3,
    gap_duration: float = 1.0,
    audio_format: str = 'flac'
) -> bool:
    """
    Build a VTT subtitle file for dual-voice bilingual audio.

    NOTE: Skips audio file 0.flac because bilingual EPUBs prepend "bookforge" as sentence 0,
    which e2a uses as the chapter title. The actual content starts at audio file 1.

    Args:
        source_sentences_dir: Directory with source language sentences (1.flac, 2.flac, ...)
        target_sentences_dir: Directory with target language sentences (1.flac, 2.flac, ...)
        sentence_pairs: List of dicts with 'source' and 'target' text keys
        output_path: Path for the output VTT file
        pause_duration: Pause duration between source and target
        gap_duration: Gap duration between pairs
        audio_format: Audio file extension

    Returns:
        True if successful, False otherwise
    """
    num_pairs = len(sentence_pairs)
    print(f"[BILINGUAL-VTT-DUAL] Building VTT with {num_pairs} pairs")
    print(f"[BILINGUAL-VTT-DUAL] Skipping audio file 0 (bookforge chapter marker)")

    if num_pairs < 1:
        print("[BILINGUAL-VTT-DUAL] Error: Need at least 1 pair")
        return False

    def format_timestamp(seconds: float) -> str:
        """Format seconds as VTT timestamp HH:MM:SS.mmm"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"

    # Get durations for all sentence files
    print("[BILINGUAL-VTT-DUAL] Getting audio durations...")
    source_durations = []
    target_durations = []

    for i in range(num_pairs):
        # Skip audio file 0 (bookforge chapter marker) - actual content starts at file 1
        audio_idx = i + 1
        source_file = os.path.join(source_sentences_dir, f"{audio_idx}.{audio_format}")
        target_file = os.path.join(target_sentences_dir, f"{audio_idx}.{audio_format}")

        source_dur = get_audio_duration(source_file)
        target_dur = get_audio_duration(target_file)

        source_durations.append(source_dur)
        target_durations.append(target_dur)

        if source_dur == 0:
            print(f"[BILINGUAL-VTT-DUAL] Warning: Zero duration for source audio file {audio_idx}")
        if target_dur == 0:
            print(f"[BILINGUAL-VTT-DUAL] Warning: Zero duration for target audio file {audio_idx}")

    # Build VTT cues
    vtt_lines = ["WEBVTT", ""]
    current_time = 0.0

    for pair_idx in range(num_pairs):
        pair = sentence_pairs[pair_idx]
        source_text = pair.get('source', '')
        target_text = pair.get('target', '')

        source_duration = source_durations[pair_idx]
        target_duration = target_durations[pair_idx]

        # Source cue
        source_start = current_time
        source_end = current_time + source_duration
        vtt_lines.append(f"{format_timestamp(source_start)} --> {format_timestamp(source_end)}")
        vtt_lines.append(source_text)
        vtt_lines.append("")

        current_time = source_end + pause_duration

        # Target cue
        target_start = current_time
        target_end = current_time + target_duration
        vtt_lines.append(f"{format_timestamp(target_start)} --> {format_timestamp(target_end)}")
        vtt_lines.append(target_text)
        vtt_lines.append("")

        current_time = target_end + gap_duration

    # Write VTT file
    print(f"[BILINGUAL-VTT-DUAL] Writing {num_pairs * 2} cues to {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(vtt_lines))
        return True
    except Exception as e:
        print(f"[BILINGUAL-VTT-DUAL] Error writing VTT: {e}")
        return False


def run_dual_voice_assembly(
    source_sentences_dir: str,
    target_sentences_dir: str,
    sentence_pairs_path: str,
    output_dir: str,
    output_name: str,
    pause_duration: float = 0.3,
    gap_duration: float = 1.0,
    audio_format: str = 'flac'
) -> dict:
    """
    Full dual-voice bilingual assembly pipeline.

    Args:
        source_sentences_dir: Directory with source language sentences
        target_sentences_dir: Directory with target language sentences
        sentence_pairs_path: Path to JSON file with sentence pairs
        output_dir: Directory for output files
        output_name: Base name for output files (without extension)
        pause_duration: Pause between source and target
        gap_duration: Gap between pairs
        audio_format: Audio file extension

    Returns:
        Dict with 'success', 'audio_path', 'vtt_path', and optionally 'error'
    """
    import json

    print("[BILINGUAL-ASSEMBLY] Starting dual-voice assembly")
    print(f"[BILINGUAL-ASSEMBLY] Source: {source_sentences_dir}")
    print(f"[BILINGUAL-ASSEMBLY] Target: {target_sentences_dir}")
    print(f"[BILINGUAL-ASSEMBLY] Pairs: {sentence_pairs_path}")
    print(f"[BILINGUAL-ASSEMBLY] Output: {output_dir}/{output_name}")

    try:
        # Load sentence pairs
        with open(sentence_pairs_path, 'r', encoding='utf-8') as f:
            sentence_pairs = json.load(f)

        num_pairs = len(sentence_pairs)
        print(f"[BILINGUAL-ASSEMBLY] Loaded {num_pairs} sentence pairs")

        # Output paths
        os.makedirs(output_dir, exist_ok=True)
        audio_output = os.path.join(output_dir, f"{output_name}.flac")
        vtt_output = os.path.join(output_dir, f"{output_name}.vtt")
        m4b_output = os.path.join(output_dir, f"{output_name}.m4b")

        # Combine audio
        print("[BILINGUAL-ASSEMBLY] Combining audio...")
        audio_success = combine_dual_voice_audio(
            source_sentences_dir=source_sentences_dir,
            target_sentences_dir=target_sentences_dir,
            output_file=audio_output,
            num_pairs=num_pairs,
            pause_duration=pause_duration,
            gap_duration=gap_duration,
            audio_format=audio_format
        )

        if not audio_success:
            return {'success': False, 'error': 'Audio combination failed'}

        # Build VTT
        print("[BILINGUAL-ASSEMBLY] Building VTT...")
        vtt_success = build_dual_voice_vtt(
            source_sentences_dir=source_sentences_dir,
            target_sentences_dir=target_sentences_dir,
            sentence_pairs=sentence_pairs,
            output_path=vtt_output,
            pause_duration=pause_duration,
            gap_duration=gap_duration,
            audio_format=audio_format
        )

        if not vtt_success:
            return {'success': False, 'error': 'VTT generation failed'}

        # Convert to M4B
        print("[BILINGUAL-ASSEMBLY] Converting to M4B...")
        ffmpeg = shutil.which('ffmpeg')
        if ffmpeg:
            cmd = [
                ffmpeg, '-y',
                '-i', audio_output,
                '-c:a', 'aac',
                '-b:a', '128k',
                m4b_output
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[BILINGUAL-ASSEMBLY] Created M4B: {m4b_output}")
                # Remove intermediate FLAC
                try:
                    os.remove(audio_output)
                except:
                    pass
            else:
                print(f"[BILINGUAL-ASSEMBLY] M4B conversion warning: {result.stderr}")
                m4b_output = audio_output  # Fall back to FLAC

        print("[BILINGUAL-ASSEMBLY] Complete!")
        return {
            'success': True,
            'audio_path': m4b_output,
            'vtt_path': vtt_output
        }

    except Exception as e:
        print(f"[BILINGUAL-ASSEMBLY] Error: {e}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# CLI entry point
if __name__ == '__main__':
    import argparse
    import json as json_module

    parser = argparse.ArgumentParser(description='Bilingual audio assembly')
    parser.add_argument('--mode', choices=['dual', 'legacy'], default='dual',
                        help='Assembly mode: dual (separate dirs) or legacy (interleaved)')
    parser.add_argument('--source-dir', required=True,
                        help='Source sentences directory')
    parser.add_argument('--target-dir',
                        help='Target sentences directory (for dual mode)')
    parser.add_argument('--pairs', required=True,
                        help='Path to sentence_pairs.json')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory')
    parser.add_argument('--output-name', required=True,
                        help='Output filename (without extension)')
    parser.add_argument('--pause', type=float, default=0.3,
                        help='Pause between source and target (seconds)')
    parser.add_argument('--gap', type=float, default=1.0,
                        help='Gap between pairs (seconds)')
    parser.add_argument('--format', default='flac',
                        help='Audio file format (default: flac)')

    args = parser.parse_args()

    if args.mode == 'dual':
        if not args.target_dir:
            print("Error: --target-dir required for dual mode")
            exit(1)

        result = run_dual_voice_assembly(
            source_sentences_dir=args.source_dir,
            target_sentences_dir=args.target_dir,
            sentence_pairs_path=args.pairs,
            output_dir=args.output_dir,
            output_name=args.output_name,
            pause_duration=args.pause,
            gap_duration=args.gap,
            audio_format=args.format
        )
    else:
        # Legacy mode - not implemented in CLI yet
        print("Legacy mode not implemented in CLI")
        exit(1)

    # Output result as JSON for Node.js to parse
    print("---JSON_RESULT---")
    print(json_module.dumps(result))
