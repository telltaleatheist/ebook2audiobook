import os, torch, subprocess, shutil, json, numpy as np

from torch import Tensor
from typing import Any, Union
from scipy.io import wavfile as wav
from scipy.signal import find_peaks

from lib.classes.subprocess_pipe import SubprocessPipe

def detect_gender(voice_path:str)->str|None:
    try:
        samplerate, signal = wav.read(voice_path)
        # Ensure mono
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)
        # FFT and positive frequency range
        fft_spectrum = np.abs(np.fft.fft(signal))
        freqs = np.fft.fftfreq(len(fft_spectrum), d=1.0 / samplerate)
        positive_freqs = freqs[: len(freqs) // 2]
        positive_magnitude = fft_spectrum[: len(fft_spectrum) // 2]
        # Peak detection (20% threshold of max amplitude)
        peaks, _ = find_peaks(positive_magnitude, height=np.max(positive_magnitude) * 0.2)
        if len(peaks) == 0:
            return None
        # Detect first strong peak within human voice pitch range (75–300 Hz)
        for peak in peaks:
            freq = positive_freqs[peak]
            if 75.0 <= freq <= 300.0:
                return "female" if freq > 135.0 else "male"
        return None
    except Exception as e:
        error = f"detect_gender() error: {voice_path}: {e}"
        print(error)
        return None

def trim_audio(audio_data: Union[list[float], Tensor], samplerate: int, silence_threshold: float = 0.003, buffer_sec: float = 0.005) -> Tensor:
    # Ensure audio_data is a PyTorch tensor
    if isinstance(audio_data, list):
        audio_data = torch.tensor(audio_data, dtype=torch.float32)
    if isinstance(audio_data, Tensor):
        if audio_data.ndim != 1:
            error = "audio_data must be a 1D tensor (mono audio)."
            raise ValueError(error)
            return torch.tensor([], dtype=torch.float32)  # just for static analyzers
        if audio_data.device.type != "cpu":
            audio_data = audio_data.cpu()
        # Detect non-silent indices
        non_silent_indices = torch.where(audio_data.abs() > silence_threshold)[0]
        if len(non_silent_indices) == 0:
            return torch.tensor([], dtype=audio_data.dtype)  # Preserves dtype
        # Calculate start and end trimming indices with buffer
        start_index = max(non_silent_indices[0].item() - int(buffer_sec * samplerate), 0)
        end_index = min(non_silent_indices[-1].item() + int(buffer_sec * samplerate), audio_data.size(0))
        return audio_data[start_index:end_index]
    error = "audio_data must be a PyTorch tensor or a list of numerical values."
    raise TypeError(error)
    return torch.tensor([], dtype=torch.float32)
    
_LONGPATH_PREFIX = '\\\\?\\'

def _mediainfo_path(path:str)->str:
    """
    Render a path in the form mediainfo can actually open on Windows.

    mediainfo does not opt into long paths: handed anything past the 260-char MAX_PATH
    limit it prints `"media": null` and STILL EXITS 0, so the duration comes back
    missing with no error raised anywhere. The `\\\\?\\` extended-length prefix bypasses
    the limit, and it works on short paths too — so it is applied unconditionally
    rather than past a threshold that nothing in normal use would ever exercise.
    """
    if os.name != 'nt':
        return path
    # `\\?\` switches off path normalization, so normalize BEFORE prefixing: forward
    # slashes, '..' and relative paths are all rejected once the prefix is on.
    abspath = os.path.abspath(path)
    if abspath.startswith(_LONGPATH_PREFIX):
        return abspath
    if abspath.startswith('\\\\'):
        # UNC \\server\share\... → \\?\UNC\server\share\...
        return f'{_LONGPATH_PREFIX}UNC\\{abspath[2:]}'
    return f'{_LONGPATH_PREFIX}{abspath}'

def _mediainfo_key(path:str)->str:
    """Undo _mediainfo_path() so mediainfo's echoed @ref keys the same as the input."""
    if path.startswith(f'{_LONGPATH_PREFIX}UNC\\'):
        path = '\\\\' + path[len(_LONGPATH_PREFIX) + 4:]
    elif path.startswith(_LONGPATH_PREFIX):
        path = path[len(_LONGPATH_PREFIX):]
    return os.path.realpath(path)

def _mediainfo_durations(filepaths:list[str])->dict[str, float]:
    """
    Duration of each file, keyed by realpath. Files mediainfo could not read are
    ABSENT from the result rather than present as 0.0 — callers must decide what a
    missing duration means, because silently treating it as zero is what let a
    17-hour assembly report itself empty.
    """
    mediainfo = shutil.which('mediainfo')
    if mediainfo is None:
        raise RuntimeError('mediainfo is not on PATH; audio durations cannot be measured.')
    # Windows CreateProcess caps the command line at 32767 chars (and POSIX has
    # ARG_MAX), so one mediainfo invocation over a long file list dies there —
    # batch the paths by cumulative command length instead.
    max_cmd_len = 24000
    batches: list[list[str]] = []
    batch: list[str] = []
    batch_len = 0
    for p in filepaths:
        arg = _mediainfo_path(str(p))
        arg_len = len(arg) + 3  # quotes + separator
        if batch and batch_len + arg_len > max_cmd_len:
            batches.append(batch)
            batch = []
            batch_len = 0
        batch.append(arg)
        batch_len += arg_len
    if batch:
        batches.append(batch)
    durations: dict[str, float] = {}
    for batch in batches:
        out = subprocess.check_output([mediainfo, '--Output=JSON', *batch], text=True)
        durations.update(_extract_mediainfo_durations(json.loads(out)))
    return durations

def _extract_mediainfo_durations(data:dict|list)->dict[str, float]:
    durations: dict[str, float] = {}
    if isinstance(data, list):
        media_blocks = data
    else:
        media_blocks = [data]
    for block in media_blocks:
        media = block.get("media")
        if not media:
            continue
        media_list = media if isinstance(media, list) else [media]
        for m in media_list:
            ref = m.get("@ref")
            if not ref:
                continue
            ref = _mediainfo_key(ref)

            for track in m.get("track", []):
                raw = track.get("Duration")
                if not raw:
                    continue
                try:
                    durations[ref] = float(raw)
                    break
                except (TypeError, ValueError):
                    continue
    return durations

def get_audio_duration(filepath:str)->float:
    """
    Duration in seconds. Raises rather than returning 0.0 for an unreadable file: a
    zero here does not look like an error to any caller, it looks like silence, and
    every consumer (the assembly duration guard, VTT cue timing, the loudnorm
    long-book branch) then draws a confidently wrong conclusion from it.
    """
    key = os.path.realpath(str(filepath))
    durations = _mediainfo_durations([filepath])
    if key not in durations:
        raise RuntimeError(
            f'get_audio_duration() mediainfo reported no duration for {filepath}'
        )
    return durations[key]

def get_audiolist_duration(filepaths:list[str])->dict[str, float]:
    """Duration of every path, keyed by realpath. Raises if any file is unreadable."""
    durations = _mediainfo_durations(filepaths)
    keys = [os.path.realpath(str(p)) for p in filepaths]
    missing = [p for p, k in zip(filepaths, keys) if k not in durations]
    if missing:
        raise RuntimeError(
            f'get_audiolist_duration() mediainfo reported no duration for '
            f'{len(missing)} of {len(filepaths)} files, first: {missing[0]}'
        )
    return {k: durations[k] for k in keys}

def normalize_audio(input_file:str, output_file:str, samplerate:int, is_gui_process:bool)->bool:
    filter_complex = (
        'agate=threshold=-25dB:ratio=1.4:attack=10:release=250,'
        'afftdn=nf=-70,'
        'acompressor=threshold=-20dB:ratio=2:attack=80:release=200:makeup=1dB,'
        'loudnorm=I=-14:TP=-3:LRA=7:linear=true,'
        'equalizer=f=150:t=q:w=2:g=1,'
        'equalizer=f=250:t=q:w=2:g=-3,'
        'equalizer=f=3000:t=q:w=2:g=2,'
        'equalizer=f=5500:t=q:w=2:g=-4,'
        'equalizer=f=9000:t=q:w=2:g=-2,'
        'highpass=f=63[audio]'
    )
    cmd = [shutil.which('ffmpeg'), '-hide_banner', '-nostats', '-i', input_file]
    cmd += [
        '-filter_complex', filter_complex,
        '-map', '[audio]',
        '-ar', str(samplerate),
        '-y', output_file
    ]
    proc_pipe = SubprocessPipe(cmd, is_gui_process=is_gui_process, total_duration=get_audio_duration(str(input_file)), msg='Normalize')
    if proc_pipe:
        return True
    else:
        error = f"normalize_audio() error: ffmpeg failed for {input_file}"
        print(error)
        return False

def is_audio_data_valid(audio_data:Any)->bool:
    if audio_data is None:
        return False
    if isinstance(audio_data, torch.Tensor):
        return audio_data.numel() > 0
    if isinstance(audio_data, (list, tuple)):
        return len(audio_data) > 0
    try:
        if isinstance(audio_data, np.ndarray):
            return audio_data.size > 0
    except ImportError:
        pass
    return False