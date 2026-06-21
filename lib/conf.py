import os, tempfile, sys

# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------
min_python_version = (3,10)
max_python_version = (3,12)

# All paths relative to e2a root - works on any OS (Mac, Windows, Linux)
# This ensures consistent paths regardless of where python is invoked from
_e2a_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# E2A_TMP_DIR relocates SESSION storage only (BookForge points it at a big
# scratch volume). Calibre temp and tempfile stay on local disk: the scratch
# may be ExFAT/NTFS, where macOS AppleDouble ._* sidecar files in calibre's
# staging dir crash its EPUB3 upgrade (and small-file temp churn is slow).
_session_scratch = os.environ.get('E2A_TMP_DIR')
_local_tmp = os.path.join(_e2a_root, 'tmp')
tmp_dir = _session_scratch or _local_tmp
os.makedirs(tmp_dir, exist_ok=True)
if not _session_scratch:
    # Legacy behavior: everything (tempfile, calibre, TMPDIR) under <root>/tmp
    tempfile.tempdir = tmp_dir
tmp_expire = 7 # days

# multiprocessing binds AF_UNIX sockets under its temp dir, and macOS caps
# socket paths at 104 bytes. With tempfile/TMPDIR redirected into a deep e2a
# root (e.g. ~/Library/Application Support/bookforge-app/runtime/e2a/tmp in
# the packaged app), Manager() dies with "OSError: AF_UNIX path too long".
# Pin multiprocessing's temp dir (sockets only) to a short system path.
# /tmp is used instead of tempfile.gettempdir() because spawned children
# inherit the TMPDIR override below and would resolve the deep path again.
if sys.platform != 'win32':
    import shutil as _shutil
    import multiprocessing.process as _mp_process
    import multiprocessing.util as _mp_util
    _mp_config = _mp_process.current_process()._config
    if 'tempdir' not in _mp_config:
        _mp_socket_dir = tempfile.mkdtemp(prefix='pymp-', dir='/tmp')
        _mp_config['tempdir'] = _mp_socket_dir
        _mp_util.Finalize(None, _shutil.rmtree, args=[_mp_socket_dir],
                          kwargs={'ignore_errors': True}, exitpriority=-100)

models_dir = os.path.join(_e2a_root, 'models')
ebooks_dir = os.path.join(_e2a_root, 'ebooks')
voices_dir = os.path.join(_e2a_root, 'voices')
tts_dir = os.path.join(models_dir, 'tts')
components_dir = os.path.join(_e2a_root, 'components')

# ---------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------
os.environ['PYTHONUTF8'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['COQUI_TOS_AGREED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['CALIBRE_NO_NATIVE_FILEDIALOGS'] = '1'
# Calibre dirs + TMPDIR stay on the LOCAL tmp even when sessions are
# relocated — see the E2A_TMP_DIR note above.
os.makedirs(_local_tmp, exist_ok=True)
os.environ['CALIBRE_TEMP_DIR'] = _local_tmp
os.environ['CALIBRE_CACHE_DIRECTORY'] = _local_tmp
os.environ['CALIBRE_CONFIG_DIRECTORY'] = _local_tmp
if not _session_scratch:
    os.environ['TMPDIR'] = tmp_dir
os.environ['GRADIO_DEBUG'] = '0'
os.environ['DO_NOT_TRACK'] = 'True'
os.environ['HUGGINGFACE_HUB_CACHE'] = tts_dir
os.environ['HF_HOME'] = tts_dir
os.environ['HF_DATASETS_CACHE'] = tts_dir
os.environ['BARK_CACHE_DIR'] = tts_dir
os.environ['TTS_CACHE'] = tts_dir
os.environ['TORCH_HOME'] = tts_dir
os.environ['TTS_HOME'] = models_dir
os.environ['XDG_CACHE_HOME'] = models_dir
os.environ['MPLCONFIGDIR'] = f'{models_dir}/matplotlib'
os.environ['TESSDATA_PREFIX'] = f'{models_dir}/tessdata'
os.environ['STANZA_RESOURCES_DIR'] = os.path.join(models_dir, 'stanza')
os.environ['ARGOS_TRANSLATE_PACKAGE_PATH'] = os.path.join(models_dir, 'argostranslate')
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
# PYTORCH_NO_CUDA_MEMORY_CACHING breaks CUDA graph capture on Linux
# Only disable caching on Windows where CUDA graphs are disabled anyway
if sys.platform == 'win32':
    os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# CUDA graph settings - only disable on Windows where they cause issues
# On Linux/WSL, keep CUDA graphs enabled for vLLM performance
if sys.platform == 'win32':
    os.environ['TORCH_CUDA_ENABLE_CUDA_GRAPH'] = '0'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
else:
    os.environ['TORCH_CUDA_ENABLE_CUDA_GRAPH'] = '1'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUDA_CACHE_MAXSIZE'] = '2147483648'
os.environ['SUNO_OFFLOAD_CPU'] = 'False'
os.environ['SUNO_USE_SMALL_MODELS'] = 'False'
# espeak-ng: prefer the pip-bundled library/data (espeakng-loader) so phonemizer
# works inside a frozen/relocated env with no system espeak-ng install.
try:
    import espeakng_loader
    os.environ.setdefault('PHONEMIZER_ESPEAK_LIBRARY', str(espeakng_loader.get_library_path()))
    os.environ.setdefault('ESPEAK_DATA_PATH', str(espeakng_loader.get_data_path()))
    # coqui-tts (TTS.tts.utils.text.phonemizers.espeak_wrapper) shells out to the
    # espeak-ng EXECUTABLE rather than the phonemizer ctypes library. On Windows
    # the relocatable env ships only the DLL (via espeakng-loader), so XTTS fails
    # to load ("espeak-ng --voices returned non-zero exit status"). Ship a matching
    # espeak-ng.exe (1.52, same as the loader's data) and put it on PATH so the
    # CLI resolves. Must run BEFORE TTS is imported (worker imports lib.conf first).
    if sys.platform == 'win32':
        _espeak_bin = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ext', 'espeak-ng-win',
        )
        if os.path.isfile(os.path.join(_espeak_bin, 'espeak-ng.exe')):
            os.environ['PATH'] = _espeak_bin + os.pathsep + os.environ.get('PATH', '')
            # The bundled exe is version-matched to the loader's data; point it there.
            os.environ['ESPEAK_DATA_PATH'] = str(espeakng_loader.get_data_path())
except ImportError:
    if sys.platform == 'win32':
        # espeakng_loader not installed (dev checkout). Still prepend the bundled,
        # self-contained espeak-ng.exe (ships its own libespeak-ng.dll) so the CLI
        # resolves to it. A system espeak-ng (e.g. scoop) breaks when run as a
        # subprocess of torch-python (DLL conflict -> empty --version -> coqui
        # espeak_wrapper IndexError on output.split()[3]).
        _espeak_bin = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'ext', 'espeak-ng-win',
        )
        if os.path.isfile(os.path.join(_espeak_bin, 'espeak-ng.exe')):
            os.environ['PATH'] = _espeak_bin + os.pathsep + os.environ.get('PATH', '')
        # Correct scoop data dir is '...\current\espeak-ng-data' (no 'eSpeak NG'
        # subdir). A wrong/nonexistent ESPEAK_DATA_PATH makes espeak-ng emit empty
        # output -> coqui espeak_wrapper IndexError. Only set it if it resolves.
        _edp = os.path.expandvars(r"%USERPROFILE%\scoop\apps\espeak-ng\current\espeak-ng-data")
        if os.path.isdir(_edp):
            os.environ['ESPEAK_DATA_PATH'] = _edp

# ---------------------------------------------------------------------
# Version and runtime config
# ---------------------------------------------------------------------
prog_version = (lambda: open('VERSION.txt').read().strip())()
max_upload_size = '6GB'

NATIVE = 'native'
FULL_DOCKER = 'full_docker'
BUILD_DOCKER = 'build_docker'

debug_mode = False

# ---------------------------------------------------------------------
# Hardware mappings
# ---------------------------------------------------------------------

systems = {
    "LINUX": "linux",
    "MACOS": "darwin",
    "WINDOWS": "win32"
}

devices = {
    "CPU": {"proc": "cpu", "found": True},
    "CUDA": {"proc": "cuda", "found": False},
    "MPS": {"proc": "mps", "found": False},
    "ROCM": {"proc": "rocm", "found": False},
    "XPU": {"proc": "xpu", "found": False},
    "JETSON": {"proc": "jetson", "found": False},
}

default_device = devices['CPU']['proc']
default_chapters_preview = False

default_gpu_wiki = '<a href="https://github.com/DrewThomasson/ebook2audiobook/wiki/GPU-ISSUES">GPU howto wiki</a>'

default_py_major = sys.version_info.major
default_py_minor = sys.version_info.minor

default_pytorch_url = 'https://download.pytorch.org/whl'
default_jetson_url = 'https://www.e-blokos.com/whl/jetson' # TODO: find a permanent website where to upload the jetpack torch

torch_matrix = {
    # CUDA
    "cu118": {"url": default_pytorch_url, "base": "2.7.1"},
    "cu121": {"url": default_pytorch_url, "base": "2.5.1"},
    "cu124": {"url": default_pytorch_url, "base": "2.6.0"},
    "cu126": {"url": default_pytorch_url, "base": "2.7.1"},
    "cu128": {"url": default_pytorch_url, "base": "2.7.1"},
    # ROCm
    "rocm5.7":   {"url": default_pytorch_url, "base": "2.3.1"},
    "rocm6.0":   {"url": default_pytorch_url, "base": "2.4.1"},
    "rocm6.1":   {"url": default_pytorch_url, "base": "2.6.0"},
    "rocm6.2":   {"url": default_pytorch_url, "base": "2.5.1"},
    "rocm6.2.4": {"url": default_pytorch_url, "base": "2.7.1"},
    "rocm6.3":   {"url": default_pytorch_url, "base": "2.7.1"},
    # MPS
    "mps": {"url": default_pytorch_url, "base": "2.7.1"},
    # XPU
    "xpu": {"url": default_pytorch_url, "base": "2.7.1"},
    # JETSON
    "jetson60": {"url": default_jetson_url, "base": "2.4.0"},
    "jetson61": {"url": default_jetson_url, "base": "2.5.0"},
}

cuda_version_range = {"min": (11,8), "max": (12,9)}
rocm_version_range = {"min": (5,7), "max": (6,3)}
mps_version_range = {"min": (0,0), "max": (0,0)}
xpu_version_range = {"min": (0,0), "max": (0,0)}
jetson_version_range = {"min": (6,0), "max": (6,1)}

# ---------------------------------------------------------------------
# Python environment references
# ---------------------------------------------------------------------

python_env_dir = os.path.join(_e2a_root, 'python_env')
requirements_file = os.path.join(_e2a_root, 'requirements.txt')

# ---------------------------------------------------------------------
# Interface configuration
# ---------------------------------------------------------------------

interface_host = '0.0.0.0'
interface_port = 7860
interface_shared_tmp_expire = 3 # in days
interface_concurrency_limit = 1 # or None for unlimited multiple parallele user conversion

interface_component_options = {
    "gr_tab_xtts_params": True,
    "gr_tab_bark_params": True,
    "gr_group_voice_file": True,
    "gr_group_custom_model": True
}

# ---------------------------------------------------------------------
# UI directories
# ---------------------------------------------------------------------

audiobooks_gradio_dir = os.path.join(_e2a_root, 'audiobooks', 'gui', 'gradio')
audiobooks_host_dir = os.path.join(_e2a_root, 'audiobooks', 'gui', 'host')
audiobooks_cli_dir = os.path.join(_e2a_root, 'audiobooks', 'cli')

# ---------------------------------------------------------------------
# files and audio supported formats
# ---------------------------------------------------------------------

ebook_formats = [
    ".epub", ".mobi", ".azw3", ".fb2", ".lrf", ".rb", ".snb", ".tcr", ".pdf",
    ".txt", ".rtf", ".doc", ".docx", ".html", ".odt", ".azw", ".tiff", ".tif",
    ".png", ".jpg", ".jpeg", ".bmp",
]
voice_formats = [
    ".mp4", ".m4b", ".m4a", ".mp3", ".wav", ".aac", ".flac", ".alac", ".ogg",
    ".aiff", ".aif", ".wma", ".dsd", ".opus", ".pcmu", ".pcma", ".gsm",
]
output_formats = [
    "aac", "flac", "mp3", "m4b", "m4a", "mp4", "mov", "ogg", "wav", "webm",
]
default_audio_proc_samplerate = 24000
default_audio_proc_format = 'flac' # or 'mp3', 'aac', 'm4a', 'm4b', 'amr', '3gp', 'alac'. 'wav' format is ok but limited to process files < 4GB
default_output_format = 'm4b'
default_output_channel = 'mono' # mono or stereo
default_output_split = False
default_output_split_hours = '6' # if the final ouput esceed outpout_split_hours * 2 hours the final file will be splitted by outpout_split_hours + the end if any.