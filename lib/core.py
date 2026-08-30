# NOTE!!NOTE!!!NOTE!!NOTE!!!NOTE!!NOTE!!!NOTE!!NOTE!!!
# THE WORD "CHAPTER" IN THE CODE DOES NOT MEAN
# IT'S THE REAL CHAPTER OF THE EBOOK SINCE NO STANDARDS
# ARE DEFINING A CHAPTER ON .EPUB FORMAT. THE WORD "BLOCK"
# IS USED TO PRINT IT OUT TO THE TERMINAL, AND "CHAPTER" TO THE CODE
# WHICH IS LESS GENERIC FOR THE DEVELOPERS

import argparse, asyncio, csv, difflib, fnmatch, hashlib, io, json, math, os, pytesseract, gc
import concurrent.futures
import random, shutil, subprocess, sys, tempfile, threading, time, uvicorn
import traceback, socket, unicodedata, urllib.request, urllib.parse, uuid, zipfile, fitz, multiprocessing
import ebooklib, gradio as gr, psutil, regex as re, requests, stanza, importlib, queue

from typing import Any, Generator, Dict
from PIL import Image, ImageSequence
from tqdm import tqdm
from bs4 import BeautifulSoup, NavigableString, Tag
from collections import Counter
from collections.abc import Mapping, MutableMapping
from datetime import datetime
from ebooklib import epub
from ebooklib.epub import EpubBook
from ebooklib.epub import EpubHtml
from glob import glob
from iso639 import Lang
from markdown import markdown
from multiprocessing import Pool, cpu_count
from multiprocessing import Manager, Event
from multiprocessing.managers import DictProxy, ListProxy, SyncManager
from stanza.pipeline.core import Pipeline, DownloadMethod
from num2words import num2words
from pathlib import Path
from PIL import Image
from pydub import AudioSegment
from pydub.utils import mediainfo
from queue import Queue, Empty
from types import MappingProxyType
from langdetect import detect
from unidecode import unidecode
from phonemizer import phonemize

from lib.classes.subprocess_pipe import SubprocessPipe
from lib.classes.tts_engines.common.orpheus_text import (
    expand_digits as orpheus_expand_digits,
    normalize_scripture as orpheus_normalize_scripture,
)
from lib.classes.vram_detector import VRAMDetector
from lib.classes.voice_extractor import VoiceExtractor
from lib.classes.tts_manager import TTSManager
#from lib.classes.redirect_console import RedirectConsole
#from lib.classes.argos_translator import ArgosTranslator
from lib.classes.tts_engines.common.audio import get_audiolist_duration, get_audio_duration

from lib import *

#import logging
#logging.basicConfig(
#    level=logging.INFO, # DEBUG for more verbosity
#    format="%(asctime)s [%(levelname)s] %(message)s"
#)

context = None
context_tracker = None
active_sessions = None

class DependencyError(Exception):
    def __init__(self, message:str|None):
        super().__init__(message)
        print(message)
        # Automatically handle the exception when it's raised
        self.handle_exception()

    def handle_exception(self)->None:
        # Print the full traceback of the exception
        traceback.print_exc()      
        # Print the exception message
        error = f'Caught DependencyError: {self}'
        print(error)

class SessionTracker:
    def __init__(self):
        self.lock = threading.Lock()

    def start_session(self, session_id:str)->bool:
        with self.lock:
            session = context.get_session(session_id)
            if session['status'] is None:
                session['status'] = 'ready'
                return True
        return False

    def end_session(self, session_id:str, socket_hash:str)->None:
        active_sessions.discard(socket_hash)
        with self.lock:
            context.sessions.pop(session_id, None)

class SessionContext:
    def __init__(self):
        self.manager:Manager = Manager()
        self.sessions:DictProxy[str, DictProxy[str, Any]] = self.manager.dict()
        self.cancellation_events = {}

    def _recursive_proxy(self, data:Any, manager:SyncManager|None)->Any:
        if manager is None:
            manager = self.manager
        if isinstance(data, dict):
            proxy_dict = manager.dict()
            for key, value in data.items():
                proxy_dict[key] = self._recursive_proxy(value, manager)
            return proxy_dict
        elif isinstance(data, list):
            proxy_list = manager.list()
            for item in data:
                proxy_list.append(self._recursive_proxy(item, manager))
            return proxy_list
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            error = f'Unsupported data type: {type(data)}'
            print(error)
            return None

    def set_session(self, session_id:str)->Any:
        self.sessions[session_id] = self._recursive_proxy({
            "id": session_id,
            "script_mode": NATIVE,
            "tab_id": None,
            "is_gui_process": False,
            "free_vram_gb": 0,
            "process_id": None,
            "status": None,
            "event": None,
            "ticker": 0,
            "heartbeat": time.time(),
            "cancellation_requested": False,
            "device": default_device,
            "tts_engine": default_tts_engine,
            "fine_tuned": default_fine_tuned,
            "model_cache": None,
            "model_zs_cache": None,
            "stanza_cache": None,
            "system": None,
            "client": None,
            "language": default_language_code,
            "language_iso1": None,
            "audiobook": None,
            "audiobooks_dir": None,
            "process_dir": None,
            "ebook": None,
            "ebook_list": None,
            "ebook_mode": "single",
            "chapters_preview": default_chapters_preview,
            "chapters_dir": None,
            "sentences_dir": None,
            "epub_path": None,
            "filename_noext": None,
            "voice": None,
            "voice_dir": None,
            "custom_model": None,
            "custom_model_dir": None,
            "xtts_temperature": default_engine_settings[TTS_ENGINES['XTTSv2']]['temperature'],
            #"xtts_codec_temperature": default_engine_settings[TTS_ENGINES['XTTSv2']]['codec_temperature'],
            "xtts_length_penalty": default_engine_settings[TTS_ENGINES['XTTSv2']]['length_penalty'],
            "xtts_num_beams": default_engine_settings[TTS_ENGINES['XTTSv2']]['num_beams'],
            "xtts_repetition_penalty": default_engine_settings[TTS_ENGINES['XTTSv2']]['repetition_penalty'],
            #"xtts_cvvp_weight": default_engine_settings[TTS_ENGINES['XTTSv2']]['cvvp_weight'],
            "xtts_top_k": default_engine_settings[TTS_ENGINES['XTTSv2']]['top_k'],
            "xtts_top_p": default_engine_settings[TTS_ENGINES['XTTSv2']]['top_p'],
            "xtts_speed": default_engine_settings[TTS_ENGINES['XTTSv2']]['speed'],
            #"xtts_gpt_cond_len": default_engine_settings[TTS_ENGINES['XTTSv2']]['gpt_cond_len'],
            #"xtts_gpt_batch_size": default_engine_settings[TTS_ENGINES['XTTSv2']]['gpt_batch_size'],
            "xtts_enable_text_splitting": default_engine_settings[TTS_ENGINES['XTTSv2']]['enable_text_splitting'],
            "bark_text_temp": default_engine_settings[TTS_ENGINES['BARK']]['text_temp'],
            "bark_waveform_temp": default_engine_settings[TTS_ENGINES['BARK']]['waveform_temp'],
            "final_name": None,
            "output_format": default_output_format,
            "output_channel": default_output_channel,
            "output_split": default_output_split,
            "output_split_hours": default_output_split_hours,
            "metadata": {
                "title": None, 
                "creator": None,
                "contributor": None,
                "language": None,
                "identifier": None,
                "publisher": None,
                "date": None,
                "description": None,
                "subject": None,
                "rights": None,
                "format": None,
                "type": None,
                "coverage": None,
                "relation": None,
                "Source": None,
                "Modified": None,
            },
            "chapters": [],
            "cover": None,
            "duration": 0,
            "playback_time": 0,
            "playback_volume": 0
        }, manager=self.manager)
        return self.sessions[session_id]

    def get_session(self, session_id:str)->Any:
        if session_id in self.sessions:
            return self.sessions[session_id]
        return {}

    def find_id_by_hash(self, socket_hash: str) -> str | None:
        for session_id, session in list(self.sessions.items()):
            if socket_hash in session:
                return session_id
        return None

class JSONDictProxyEncoder(json.JSONEncoder):
    def default(self, o:Any)->Any:
        if isinstance(o, DictProxy):
            return dict(o)
        elif isinstance(o, ListProxy):
            return list(o)
        return super().default(o)

def prepare_dirs(src:str, session_id:str)->bool:
    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            resume = False
            os.makedirs(os.path.join(models_dir,'tts'), exist_ok=True)
            os.makedirs(session['session_dir'], exist_ok=True)
            os.makedirs(session['process_dir'], exist_ok=True)
            os.makedirs(session['custom_model_dir'], exist_ok=True)
            os.makedirs(session['voice_dir'], exist_ok=True)
            os.makedirs(session['audiobooks_dir'], exist_ok=True)
            os.makedirs(session['chapters_dir'], exist_ok=True)
            os.makedirs(session['sentences_dir'], exist_ok=True)
            session['ebook'] = os.path.join(session['process_dir'], os.path.basename(src))
            shutil.copy(src, session['ebook'])
            return True
    except Exception as e:
        DependencyError(e)
        return False

def check_programs(prog_name:str, command:str, options:str)->bool:
    try:
        subprocess.run(
            [command, options],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            check=True,
            text=True,
            encoding='utf-8'
        )
        return True
    except FileNotFoundError:
        e = f'''********** Error: {prog_name} is not installed! if your OS calibre package version 
        is not compatible you still can run ebook2audiobook.sh (linux/mac) or ebook2audiobook.cmd (windows) **********'''
        DependencyError(e)
    except subprocess.CalledProcessError:
        e = f'Error: There was an issue running {prog_name}.'
        DependencyError(e)
    return False

def analyze_uploaded_file(zip_path:str, required_files:list[str])->bool:
    try:
        if not os.path.exists(zip_path):
            error = f'The file does not exist: {os.path.basename(zip_path)}'
            print(error)
            return False
        files_in_zip = {}
        empty_files = set()
        with zipfile.ZipFile(zip_path, 'r') as zf:
            for file_info in zf.infolist():
                file_name = file_info.filename
                if file_info.is_dir():
                    continue
                base_name = os.path.basename(file_name)
                files_in_zip[base_name.lower()] = file_info.file_size
                if file_info.file_size == 0:
                    empty_files.add(base_name.lower())
        required_files = [file.lower() for file in required_files]
        missing_files = [f for f in required_files if f not in files_in_zip]
        required_empty_files = [f for f in required_files if f in empty_files]
        if missing_files:
            msg = f'Missing required files: {missing_files}'
            print(msg)
        if required_empty_files:
            msg = f'Required files with 0 KB: {required_empty_files}'
            print(msg)
        return not missing_files and not required_empty_files
    except zipfile.BadZipFile:
        error = 'The file is not a valid ZIP archive.'
        print(error)
        return False
    except Exception as e:
        error = f'An error occurred: {e}'
        print(error)
        return False

def extract_custom_model(file_src:str, session_id, required_files:list)->str|None:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        model_path = None
        model_name = re.sub('.zip', '', os.path.basename(file_src), flags=re.IGNORECASE)
        model_name = get_sanitized(model_name)
        try:
            with zipfile.ZipFile(file_src, 'r') as zip_ref:
                files = zip_ref.namelist()
                files_length = len(files)
                tts_dir = session['tts_engine']
                model_path = os.path.join(session['custom_model_dir'], tts_dir, model_name)
                os.makedirs(model_path, exist_ok=True)
                required_files_lc = set(x.lower() for x in required_files)
                with tqdm(total=files_length, unit='files') as t:
                    for f in files:
                        base_f = os.path.basename(f).lower()
                        if base_f in required_files_lc:
                            out_path = os.path.join(model_path, base_f)
                            with zip_ref.open(f) as src, open(out_path, 'wb') as dst:
                                shutil.copyfileobj(src, dst)
                        t.update(1)
            if model_path is not None:
                msg = f'Extracted files to {model_path}. Normalizing ref.wav…'
                print(msg)
                voice_ref = os.path.join(model_path, 'ref.wav')
                voice_output = os.path.join(model_path, f'{model_name}.wav')
                extractor = VoiceExtractor(session, voice_ref, model_name)
                success, error = extractor.normalize_audio(voice_ref, voice_output, voice_output)
                if success:
                    if os.path.exists(file_src):
                        os.remove(file_src)
                    if os.path.exists(voice_ref):
                        os.remove(voice_ref)
                    return model_path
                error = f'normalize_audio {voice_ref} error: {error}'
                print(error)
            else:
                error = f'An error occured when unzip {file_src}'
        except asyncio.exceptions.CancelledError as e:
            DependencyError(e)
            error = f'extract_custom_model asyncio.exceptions.CancelledError: {e}'
            print(error)
        except Exception as e:
            DependencyError(e)
            error = f'extract_custom_model Exception: {e}'
            print(error)
        if session['is_gui_process']:
            if os.path.exists(file_src):
                os.remove(file_src)
    return None
        
def hash_proxy_dict(proxy_dict)->str:
    try:
        data = dict(proxy_dict)
    except Exception:
        data = {}
    data_str = json.dumps(data, sort_keys=True, default=str)
    return hashlib.md5(data_str.encode("utf-8")).hexdigest()

def compare_checksums(src_path:str, checksum_path:str, hash_algorithm:str='sha256')->tuple[bool, str|None]:
    try:
        hash_func = hashlib.new(hash_algorithm)
        with open(src_path, 'rb') as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        new_checksum = hash_func.hexdigest()
        if not os.path.exists(checksum_path):
            with open(checksum_path, 'w', encoding='utf-8') as f:
                f.write(new_checksum)
            return False, None
        else:
            with open(checksum_path, 'r', encoding='utf-8') as f:
                saved_checksum = f.read().strip()
            if saved_checksum == new_checksum:
                return True, None
            else:
                with open(checksum_path, 'w', encoding='utf-8') as f:
                    f.write(new_checksum)
                    return False, None
    except Exception as e:
        return False, f'compare_checksums() error: {e}'

def compare_dict_keys(d1, d2):
    if not isinstance(d1, Mapping) or not isinstance(d2, Mapping):
        return d1 == d2
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    missing_in_d2 = d1_keys - d2_keys
    missing_in_d1 = d2_keys - d1_keys
    if missing_in_d2 or missing_in_d1:
        return {
            "missing_in_d2": missing_in_d2,
            "missing_in_d1": missing_in_d1,
        }
    for key in d1_keys.intersection(d2_keys):
        nested_result = compare_keys(d1[key], d2[key])
        if nested_result:
            return {key: nested_result}
    return None

def ocr2xhtml(img: Image.Image, lang: str)->str:
    try:
        debug = True
        try:
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME)
            # Handle silent OCR failures (empty or None result)
            if data is None or data.empty:
                error = f'Tesseract returned empty OCR data for language "{lang}".'
                print(error)
                return False
        except (pytesseract.TesseractError, Exception) as e:
            print(f'The OCR {lang} trained model must be downloaded.')
            try:
                tessdata_dir = os.environ['TESSDATA_PREFIX']
                os.makedirs(tessdata_dir, exist_ok=True)
                url = f'https://github.com/tesseract-ocr/tessdata_best/raw/main/{lang}.traineddata'
                dest_path = os.path.join(tessdata_dir, f'{lang}.traineddata')
                msg = f'Downloading {lang}.traineddata into {tessdata_dir}…'
                print(msg)
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    with open(dest_path, 'wb') as f:
                        f.write(response.content)
                    msg = f'Downloaded and installed {lang}.traineddata successfully.'
                    print(msg)
                    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DATAFRAME)
                    if data is None or data.empty:
                        error = f'Tesseract returned empty OCR data even after downloading {lang}.traineddata.'
                        print(error)
                        return False
                else:
                    error = f'Failed to download traineddata for {lang} (HTTP {response.status_code})'
                    print(error)
                    return False
            except Exception as e:
                error = f'Automatic download failed: {e}'
                print(error)
                return False
        data = data.dropna(subset=['text'])
        lines = []
        last_block = None
        for _, row in data.iterrows():
            text = row['text'].strip()
            if not text:
                continue
            block = row['block_num']
            if last_block is not None and block != last_block:
                lines.append('')  # blank line between blocks
            lines.append(text)
            last_block = block
        joined = '\n'.join(lines)
        raw_lines = [l.strip() for l in joined.split('\n')]
        # Normalize line breaks
        merged_lines = []
        buffer = ''
        for i, line in enumerate(raw_lines):
            if not line:
                if buffer:
                    merged_lines.append(buffer.strip())
                    buffer = ''
                continue
            if buffer and not buffer.endswith(('.', '?', '!', ':')) and not line[0].isupper():
                buffer += ' ' + line
            else:
                if buffer:
                    merged_lines.append(buffer.strip())
                buffer = line
        if buffer:
            merged_lines.append(buffer.strip())
        # Detect heading-like lines
        xhtml_parts = []
        debug_dump = []
        for i, p in enumerate(merged_lines):
            is_heading = False
            if p.isupper() and len(p.split()) <= 8:
                is_heading = True
            elif len(p.split()) <= 5 and p.istitle():
                is_heading = True
            elif (i == 0 or (i > 0 and merged_lines[i-1] == '')) and len(p.split()) <= 10:
                is_heading = True
            if is_heading:
                xhtml_parts.append(f'<h2>{p}</h2>')
                debug_dump.append(f'[H2] {p}')
            else:
                xhtml_parts.append(f'<p>{p}</p>')
                debug_dump.append(f'[P ] {p}')
        if debug:
            print('=== OCR DEBUG OUTPUT ===')
            for line in debug_dump:
                print(line)
            print('========================')
        return '\n'.join(xhtml_parts)
    except Exception as e:
        DependencyError(e)
        error = f'ocr2xhtml error: {e}'
        print(error)
        return False

def chapter_provenance_path(process_dir:str)->str:
    # Chapter provenance lives in the session's process_dir, NOT only in the
    # session dict or session-state.json: the prepare, worker and assemble phases
    # can be separate processes (and separate front-ends — bookforge_ext writes its
    # own session-state.json), and all of them share process_dir.
    return os.path.join(process_dir, 'chapter-provenance.json')

def save_chapter_provenance(session_id:str)->bool:
    session = context.get_session(session_id)
    if not session or not session.get('process_dir'):
        print('save_chapter_provenance(): no process_dir; chapter provenance NOT persisted')
        return False
    try:
        path = chapter_provenance_path(session['process_dir'])
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'chapter_docs': list(session.get('chapter_docs', [])),
                'chapter_titles_by_doc': dict(session.get('chapter_titles_by_doc', {})),
                'chapter_titles': list(session.get('chapter_titles', [])),
            }, f, ensure_ascii=False, indent=2)
        print(f"[TOC] Chapter provenance for {len(session.get('chapter_docs', []))} chapters saved to {path}")
        return True
    except Exception as e:
        print(f'save_chapter_provenance() error: {e}')
        return False

def load_chapter_provenance(session_id:str)->bool:
    # Restore chapter provenance written by get_chapters(). If it is absent or
    # unreadable the session keys are left EMPTY on purpose — assembly then uses
    # per-chapter first sentences and says so, instead of pairing TOC titles by
    # position, which is what mislabels chapters.
    session = context.get_session(session_id)
    if not session or not session.get('process_dir'):
        return False
    path = chapter_provenance_path(session['process_dir'])
    if not os.path.exists(path):
        print(f'[TOC] No chapter provenance at {path}')
        return False
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        session['chapter_docs'] = list(data.get('chapter_docs', []))
        session['chapter_titles_by_doc'] = dict(data.get('chapter_titles_by_doc', {}))
        if data.get('chapter_titles') and not session.get('chapter_titles'):
            session['chapter_titles'] = list(data['chapter_titles'])
        print(f"[TOC] Restored provenance for {len(session['chapter_docs'])} chapters from {path}")
        return True
    except Exception as e:
        print(f'[TOC] Chapter provenance {path} unreadable ({e})')
        return False

def load_json_chapters(filepath:str)->list:
    # Raises on a missing or unreadable chapters cache instead of returning [].
    # This is only called on RESUME (checksum matched, so a prior run saved it).
    # Returning [] made the caller silently re-run get_chapters(); the fresh
    # sentence split can differ from the split the existing numbered sentence
    # files were rendered from, and resume-by-file-index then pairs old audio
    # with the WRONG text. A broken resume cache must stop the run, not degrade.
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(
            f'Saved chapters cache {filepath} could not be read ({e}). Refusing to '
            f're-split sentences over the existing session audio — delete the session '
            f'processing directory to start fresh.'
        ) from e

def save_json_chapters(session_id:str, filepath:str)->bool:
    try:
        session = context.get_session(session_id)
        if not session:
            print(f"save_json_chapters error: session not found ({session_id})")
            return False
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(session['chapters'], f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"save_json_chapters() error: {e}")
        return False

def convert2epub(session_id:str)-> bool:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        if session['cancellation_requested']:
            msg = 'Cancel requested'
            print(msg)
            return False
        try:
            title = False
            author = False
            file_input = session['ebook']
            if os.path.getsize(file_input) == 0:
                error = f'Input file is empty: {file_input}'
                print(error)
                return False
            file_ext = os.path.splitext(file_input)[1].lower()
            if file_ext not in ebook_formats:
                error = f'Unsupported file format: {file_ext}'
                print(error)
                return False
            if file_ext == '.epub':
                # The book already IS an EPUB, so its spine and TOC ARE the chapter
                # structure and they are the authority. Converting it anyway was
                # actively destructive on two counts:
                #
                #  - '--page-breaks-before' on every h1..h5, plus Calibre's default of
                #    splitting at page breaks, exploded one 7-chapter book into 78 spine
                #    documents (each opening with Calibre's own id="calibre_pb_N").
                #    get_chapters() makes ONE CHAPTER PER SPINE DOCUMENT, so every
                #    section heading became an m4b chapter marker.
                #  - '--smarten-punctuation' rewrites the book's straight quotes to
                #    typographic ones (arm's -> arm’s), mutating the very text that
                #    Orpheus fine-tunes are trained on. See the book-exact-text note in
                #    get_sentences(): we deliberately do not normalise this text.
                #
                # Non-EPUB inputs still take the full Calibre pass below — the PDF, TXT
                # and image branches flatten to a SINGLE XHTML document, and that
                # heading split is the only thing giving those books chapters at all.
                msg = f'Input is already an EPUB — using it directly, no Calibre pass: {file_input}'
                print(msg)
                session['epub_path'] = file_input
                return True
            util_app = shutil.which('ebook-convert')
            if not util_app:
                error = 'ebook-convert utility is not installed or not found.'
                print(error)
                return False
            if file_ext == '.txt':
                with open(file_input, 'r', encoding='utf-8') as f:
                    text = f.read()
                text = text.replace('\r\n', '\n')
                text = re.sub(r'\n{2,}', f".{TTS_SML['pause']['static']}", text)
                with open(file_input, 'w', encoding='utf-8') as f:
                    f.write(text)
            elif file_ext == '.pdf':
                msg = 'File input is a PDF. flatten it in XHTML…'
                print(msg)
                doc = fitz.open(file_input)
                file_meta = doc.metadata
                filename_no_ext = os.path.splitext(os.path.basename(session['ebook']))[0]
                title = file_meta.get('title') or filename_no_ext
                author = file_meta.get('author') or False
                xhtml_pages = []
                for i, page in enumerate(doc):
                    try:
                        text = page.get_text('xhtml').strip()
                    except Exception as e:
                        print(f'Error extracting text from page {i+1}: {e}')
                        text = ''
                    if not text:
                        msg = f'The page {i+1} seems to be image-based. Using OCR…'
                        print(msg)
                        if session['is_gui_process']:
                            show_alert({"type": "warning", "msg": msg})
                        pix = page.get_pixmap(dpi=300)
                        img = Image.open(io.BytesIO(pix.tobytes('png')))
                        xhtml_content = ocr2xhtml(img, session['language'])
                    else:
                        xhtml_content = text
                    if xhtml_content:
                        xhtml_pages.append(xhtml_content)
                if xhtml_pages:
                    xhtml_body = '\n'.join(xhtml_pages)
                    xhtml_text = (
                        '<?xml version="1.0" encoding="utf-8"?>\n'
                        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
                        '<head>\n'
                        f'<meta charset="utf-8"/>\n<title>{title}</title>\n'
                        '</head>\n'
                        '<body>\n'
                        f'{xhtml_body}\n'
                        '</body>\n'
                        '</html>\n'
                    )
                    file_input = os.path.join(session['process_dir'], f'{filename_no_ext}.xhtml')
                    with open(file_input, 'w', encoding='utf-8') as html_file:
                        html_file.write(xhtml_text)
                else:
                    return False
            elif file_ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
                filename_no_ext = os.path.splitext(os.path.basename(session['ebook']))[0]
                msg = f'File input is an image ({file_ext}). Running OCR…'
                print(msg)
                img = Image.open(file_input)
                xhtml_pages = []
                page_count = 0
                for i, frame in enumerate(ImageSequence.Iterator(img)):
                    page_count += 1
                    frame = frame.convert('RGB')
                    xhtml_content = ocr2xhtml(frame, session['language'])
                    xhtml_pages.append(xhtml_content)
                if xhtml_pages:
                    xhtml_body = '\n'.join(xhtml_pages)
                    xhtml_text = (
                        '<?xml version="1.0" encoding="utf-8"?>\n'
                        '<html xmlns="http://www.w3.org/1999/xhtml">\n'
                        '<head>\n'
                        f'<meta charset="utf-8"/>\n<title>{filename_no_ext}</title>\n'
                        '</head>\n'
                        '<body>\n'
                        f'{xhtml_body}\n'
                        '</body>\n'
                        '</html>\n'
                    )
                    file_input = os.path.join(session['process_dir'], f'{filename_no_ext}.xhtml')
                    with open(file_input, 'w', encoding='utf-8') as html_file:
                        html_file.write(xhtml_text)
                    print(f'OCR completed for {page_count} image page(s).')
                else:
                    return False
            msg = f"Running command: {util_app} {file_input} {session['epub_path']}"
            print(msg)
            cmd = [
                    util_app, file_input, session['epub_path'],
                    '--input-encoding=utf-8',
                    '--output-profile=generic_eink',
                    '--epub-version=3',
                    '--flow-size=0',
                    '--chapter-mark=pagebreak',
                    '--page-breaks-before',
                    "//*[name()='h1' or name()='h2' or name()='h3' or name()='h4' or name()='h5']",
                    '--disable-font-rescaling',
                    '--pretty-print',
                    '--smarten-punctuation',
                    '--verbose'
                ]
            if title:
                cmd += ['--title', title]
            if author:
                cmd += ['--authors', author]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8'
            )
            print(result.stdout)
            return True
        except subprocess.CalledProcessError as e:
            DependencyError(e)
            error = f'convert2epub subprocess.CalledProcessError: {e.stderr}'
            print(error)
            return False
        except FileNotFoundError as e:
            DependencyError(e)
            error = f'convert2epub FileNotFoundError: {e}'
            print(error)
            return False
        except Exception as e:
            DependencyError(e)
            error = f'convert2epub error: {e}'
            print(error)
            return False

def get_ebook_title(epubBook:EpubBook,all_docs:list[Any])->str|None:
    # 1. Try metadata (official EPUB title)
    meta_title = epubBook.get_metadata('DC','title')
    if meta_title and meta_title[0][0].strip():
        return meta_title[0][0].strip()
    # 2. Try <title> in the head of the first XHTML document
    if all_docs:
        html = all_docs[0].get_content().decode('utf-8')
        soup = BeautifulSoup(html,'html.parser')
        title_tag = soup.select_one('head > title')
        if title_tag and title_tag.text.strip():
            return title_tag.text.strip()
        # 3. Try <img alt = '…'> if no visible <title>
        img = soup.find('img',alt = True)
        if img:
            alt = img['alt'].strip()
            if alt and 'cover' not in alt.lower():
                return alt
    return None

def get_cover(epubBook:EpubBook, session_id:str)->bool|str:
    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            if session['cancellation_requested']:
                msg = 'Cancel requested'
                print(msg)
                return False
            cover_image = None
            cover_path = os.path.join(session['process_dir'], session['filename_noext'] + '.jpg')
            for item in epubBook.get_items_of_type(ebooklib.ITEM_COVER):
                cover_image = item.get_content()
                break
            if not cover_image:
                for item in epubBook.get_items_of_type(ebooklib.ITEM_IMAGE):
                    if 'cover' in item.file_name.lower() or 'cover' in item.get_id().lower():
                        cover_image = item.get_content()
                        break
            if cover_image:
                # Open the image from bytes
                image = Image.open(io.BytesIO(cover_image))
                # Convert to RGB if needed (JPEG doesn't support alpha)
                if image.mode in ('RGBA', 'P'):
                    image = image.convert('RGB')
                image.save(cover_path, format = 'JPEG')
                return cover_path
            return True
    except Exception as e:
        DependencyError(e)
        return False

def normalize_doc_key(href:Any)->str|None:
    # Canonical form used to compare a TOC entry href with a spine document name.
    # Drops the fragment (#anchor), percent-decodes, folds separators and any
    # './' noise, and lowercases. Returns None for an empty/absent href.
    if not href:
        return None
    s = str(href).split('#')[0].strip()
    if not s:
        return None
    s = urllib.parse.unquote(s).replace('\\', '/')
    while s.startswith('./'):
        s = s[2:]
    s = s.lstrip('/')
    return s.lower() or None

def flatten_toc(nodes:Any)->list:
    # epub TOCs nest: a node is either a Link-like object (has .title/.href) or a
    # (Section, [children]) tuple/list. The old flat comprehension dropped BOTH
    # the Section itself and every child under it. Depth-first, document order.
    out = []
    try:
        for node in nodes:
            if isinstance(node, (tuple, list)):
                for part in node:
                    if isinstance(part, (tuple, list)):
                        out.extend(flatten_toc(part))
                    elif hasattr(part, 'title'):
                        out.append(part)
            elif hasattr(node, 'title'):
                out.append(node)
    except Exception as e:
        print(f'flatten_toc() error: {e}')
    return out

def get_chapters(session_id:str, epubBook:EpubBook)->list:
    try:
        msg = r'''
*******************************************************************************
NOTE:
The warning "Character xx not found in the vocabulary."
MEANS THE MODEL CANNOT INTERPRET THE CHARACTER AND WILL MAYBE GENERATE
(AS WELL AS WRONG PUNCTUATION POSITION) AN HALLUCINATION TO IMPROVE THIS MODEL,
IT NEEDS TO ADD THIS CHARACTER INTO A NEW TRAINING MODEL.
YOU CAN IMPROVE IT OR ASK TO A TRAINING MODEL EXPERT.
*******************************************************************************
        '''
        print(msg)
        session = context.get_session(session_id)
        if session and session.get('id', False):
            if session['cancellation_requested']:
                msg = 'Cancel requested'
                print(msg)
                return []
            # Step 1: Extract TOC (Table of Contents)
            # toc_list keeps the old flat title list (heading detection below reads it).
            # toc_by_href is the identity-bearing form: normalized document href -> title.
            toc_list = []
            toc_by_href = {}
            try:
                for item in flatten_toc(epubBook.toc):
                    nt = normalize_text(str(item.title), session['language'], session['language_iso1'], session['tts_engine'])
                    if nt is None:
                        continue
                    toc_list.append(nt)
                    href_key = normalize_doc_key(getattr(item, 'href', None))
                    # Several TOC entries can point into the SAME document (anchors);
                    # the first one in reading order names that document.
                    if href_key and href_key not in toc_by_href:
                        toc_by_href[href_key] = nt
                # Store TOC titles in session for use during assembly
                session['chapter_titles'] = toc_list
                print(f'[TOC] Extracted {len(toc_list)} chapter titles from TOC ({len(toc_by_href)} distinct documents)')
            except Exception as toc_error:
                error = f'Error extracting Table of Content: {toc_error}'
                show_alert({"type": "warning", "msg": error})
                session['chapter_titles'] = []
            # Get spine item IDs
            spine_ids = [item[0] for item in epubBook.spine]
            # Filter only spine documents (i.e., reading order)
            all_docs = [
                item for item in epubBook.get_items_of_type(ebooklib.ITEM_DOCUMENT)
                if item.id in spine_ids
            ]
            if not all_docs:
                error = 'No document body found!'
                print(error)
                return []
            title = get_ebook_title(epubBook, all_docs)
            # Resolve TOC titles onto spine documents by IDENTITY (never position).
            # Keyed by the exact doc.get_name() also recorded in session['chapter_docs'],
            # so assembly does a plain dict lookup with no re-normalization.
            titles_by_doc = {}
            toc_by_basename = {}
            for href_key in toc_by_href:
                toc_by_basename.setdefault(href_key.rsplit('/', 1)[-1], []).append(href_key)
            for doc in all_docs:
                doc_key = normalize_doc_key(doc.get_name())
                if not doc_key:
                    continue
                if doc_key in toc_by_href:
                    titles_by_doc[doc.get_name()] = toc_by_href[doc_key]
                else:
                    # Some epubs write TOC hrefs relative to the nav document while
                    # document names carry an OPF-root prefix (e.g. 'OEBPS/'). Match
                    # on basename ONLY when it is unambiguous — still identity, not position.
                    candidates = toc_by_basename.get(doc_key.rsplit('/', 1)[-1], [])
                    if len(candidates) == 1:
                        titles_by_doc[doc.get_name()] = toc_by_href[candidates[0]]
                        print(f"[TOC] Matched document '{doc.get_name()}' to TOC href '{candidates[0]}' by basename")
            session['chapter_titles_by_doc'] = titles_by_doc
            unmatched_docs = [d.get_name() for d in all_docs if d.get_name() not in titles_by_doc]
            print(f'[TOC] Mapped {len(titles_by_doc)}/{len(all_docs)} spine documents to a TOC title')
            if unmatched_docs:
                print(f'[TOC] No TOC entry for: {unmatched_docs} (these use their own first sentence as title)')
            chapters = []
            chapter_docs = []
            stanza_nlp = False
            if session['language'] in year_to_decades_languages:
                try:
                    stanza_model = f"stanza-{session['language_iso1']}"
                    stanza_nlp = loaded_tts.get(stanza_model, False)
                    if stanza_nlp:
                        msg = f"NLP model {stanza_model} loaded!"
                        print(msg)
                    else:
                        use_gpu = True if (
                            (session['device'] == devices['CUDA']['proc'] and not devices['JETSON']['found'] and devices['CUDA']['found']) or
                            (session['device'] == devices['ROCM']['proc'] and devices['ROCM']['found']) or
                            (session['device'] == devices['XPU']['proc'] and devices['XPU']['found'])
                        ) else False
                        stanza_nlp = stanza.Pipeline(session['language_iso1'], processors='tokenize,ner,mwt', use_gpu=use_gpu, download_method=DownloadMethod.REUSE_RESOURCES, dir=os.getenv('STANZA_RESOURCES_DIR'))
                        if stanza_nlp:
                            session['stanza_cache'] = stanza_model
                            loaded_tts[stanza_model] = stanza_nlp
                            msg = f"NLP model {stanza_model} loaded!"
                            print(msg)
                except (ConnectionError, TimeoutError) as e:
                    error = f'Stanza model download connection error: {e}. Retry later'
                    print(error)
                    return []
                except Exception as e:
                    error = f'Stanza model initialization error: {e}'
                    print(error)
                    return []
            is_num2words_compat = get_num2words_compat(session['language_iso1'])
            for doc_idx, doc in enumerate(all_docs):
                sentences_list = filter_chapter(doc_idx, doc, session_id, stanza_nlp, is_num2words_compat)
                if sentences_list is None:
                    # A genuine extraction error on ONE document must not silently
                    # truncate the book — the old `break` returned the chapters
                    # gathered so far as if the whole book had been processed. Fail
                    # the whole conversion loudly so the missing content is surfaced.
                    error = f'Chapter extraction failed at document index {doc_idx}; aborting so the audiobook is not silently truncated'
                    print(error)
                    return []
                elif len(sentences_list) > 0:
                    # chapters and chapter_docs MUST stay index-aligned: append together.
                    # A document yielding no sentences produces no chapter and therefore
                    # consumes no title.
                    chapters.append(sentences_list)
                    chapter_docs.append(doc.get_name())
            session['chapter_docs'] = chapter_docs
            save_chapter_provenance(session_id)
            if len(chapters) == 0:
                error = 'No chapters found! possible reason: file corrupted or need to convert images to text with OCR'
                print(error)
                return []
            return chapters
        return []
    except Exception as e:
        error = f'Error extracting main content pages: {e}'
        DependencyError(error)
        return []

def _edge_chars(tag:Any)->tuple:
    """(first char, last char) of a Tag's VERBATIM text — ('', '') when it has
    none. bs4's .strings yields the source NavigableStrings untouched, so unlike
    get_text(strip=True) this still knows whether the markup put whitespace at
    the tag's edges. Used ONLY to answer that question in _tuple_row."""
    first = last = ''
    for s in tag.strings:
        if s:
            if not first:
                first = s[0]
            last = s[-1]
    return first, last


def _heading_text(tag:Any)->str:
    """A heading's text, with its LINE BREAKS READ AS SPACES (2026-08-28).

    ── What this replaces, and the damage it did ───────────────────────────────

    `get_text(strip=True)` strips every string in the tag and joins them with
    NOTHING. A title typeset across four lines — `God Will Not Protect<br/>
    Children When Parents<br/>…`, or one <span> per line, which is how a print
    title comes back from an OCR/VLM conversion — therefore came out as
    'God Will Not ProtectChildren When ParentsAllow Occult Items InTheir Homes'.
    Owen heard exactly that, fused words and all, in "14 Things Witches Hope You
    Never Find Out": the words were welded in the TEXT, so they were welded in
    the audio and in the transcript cue.

    It was always wrong; it only became audible when headings started being read
    as their own chunk, because a fused title is much harder to miss when it is
    a take of its own than when it was buried in the paragraph behind it.

    ── Why not simply `get_text(' ')` ─────────────────────────────────────────

    Because a space between two pieces is not always right. `<span
    class="dropcap">I</span>ntroduction` is ONE word split by styling, and a
    blanket separator reads it aloud as 'I ntroduction' — the same class of
    defect, in the other direction. _collapse_glue solves this for body text by
    keeping the markup's own whitespace; a heading needs the same judgement.

    So a space is inserted at a piece boundary only where the markup means a new
    word: a <br> (always — that IS a line break), or a boundary where the text
    so far ends in a word character and the next piece opens with a capital or a
    digit. 'Protect' + 'Children' takes one; 'I' + 'ntroduction' does not.

    A SPACE and never a period: these breaks fall INSIDE one sentence — the
    title wrapped — so a period would put a full stop mid-title and stop the
    reader four times in a row. The period a heading needs is the one at its
    END, which filter_chapter already adds.
    """
    out = []
    last_char = ''
    for node in tag.descendants:
        if isinstance(node, Tag):
            if node.name.lower() == 'br':
                out.append(' ')
                last_char = ' '
            continue
        if not isinstance(node, NavigableString):
            continue
        piece = str(node)
        if not piece:
            continue
        if (last_char and not last_char.isspace() and not piece[:1].isspace()
                and last_char.isalnum() and (piece[0].isupper() or piece[0].isdigit())):
            out.append(' ')
        out.append(piece)
        last_char = piece[-1]
    return re.sub(r'\s+', ' ', ''.join(out)).strip()


def _collapse_glue(rows:list)->list:
    """Resolve the ('glue', payload) markers _tuple_row emits where two pieces of
    text ABUT in the markup with NO whitespace between them.

    Two text rows either side of a glue are concatenated with NOTHING between
    them, because that is what the document says: a drop cap
    (`<span class="cic">W</span>e have freedom.`) is one word, and so is
    `<i>Fu</i>ture`. Every other case restores exactly what the walk emitted
    before the marker existed — payload is the [break] that was suppressed, or
    None where the walk emitted nothing — so no other boundary moves.

    This is deliberately NOT a drop-cap detector: it never looks at letter case
    or word length. `<span>A</span>braham Lincoln` joins because the markup has
    no space; `<span>I</span> can hardly say` keeps its space because the markup
    has one; and a plain "A man walked" was never two rows to begin with."""
    out = []
    glued = False
    suppressed = None
    for typ, payload in rows:
        if typ == 'glue':
            glued = True
            suppressed = payload if suppressed is None else suppressed
            continue
        if glued:
            glued = False
            pending, suppressed = suppressed, None
            if typ == 'text' and out and out[-1][0] == 'text':
                out[-1] = ('text', out[-1][1] + payload)
                continue
            if pending is not None:
                # The restore inherits the row's own inline-ness: a suppressed
                # break resolved by a 'sep' row (an empty pagebreak span glued to
                # the text before it) must come back as 'sep', or the restore
                # re-manufactures the block boundary the walk just declined.
                out.append(('sep' if typ == 'sep' else 'break', pending))
        out.append((typ, payload))
    return out


def filter_chapter(idx:int, doc:EpubHtml, session_id:str, stanza_nlp:Pipeline, is_num2words_compat:bool)->list|None:

    def _tuple_row(node:Any, last_text_char:str|None=None)->Generator[tuple[str, Any], None, None]|None:
        try:
            prev_child_had_data = False
            # WHITESPACE FIDELITY (see _collapse_glue): True when the SOURCE markup
            # puts whitespace between the last text character emitted at this level
            # and whatever comes next. The walk used to answer that question with
            # prev_child_had_data alone — "the previous sibling had text" — and so
            # separated EVERY pair of adjacent inline pieces, even ones the document
            # writes with nothing between them. That is what turns a drop cap
            # (`<span class="cic">W</span>e have freedom.`) into the spoken "W e have
            # freedom." and `<i>Fu</i>ture` into "Fu ture".
            ws_pending = False
            for idx, child in enumerate(node.children):
                current_child_had_data = False
                if isinstance(child, NavigableString):
                    raw = str(child)
                    text = child.strip()
                    if text:
                        if prev_child_had_data:
                            if ws_pending or raw[:1].isspace():
                                # A text node reached across whitespace from an
                                # INLINE sibling (`<em>magical</em> world`) is the
                                # same sentence, not a new block: 'sep' keeps the
                                # [break] token but never closes the block, so
                                # break_between_alnum_re can collapse it back to a
                                # space. As 'break' it manufactured a period
                                # ("magical. world") ~140 times in one real book.
                                yield ('sep', sml_token("break"))
                            else:
                                yield ('glue', sml_token("break"))
                        yield ('text', text)
                        last_text_char = text[-1]
                        current_child_had_data = True
                        ws_pending = raw[-1:].isspace()
                    elif raw:
                        # A whitespace-only string is not data, but it IS whitespace:
                        # `<span>W</span> <i>ord</i>` must not glue.
                        ws_pending = True
                elif isinstance(child, Tag):
                    name = child.name.lower()
                    first_char, last_char = _edge_chars(child)
                    lead_ws = ws_pending or first_char.isspace()
                    if name in heading_tags:
                        title = _heading_text(child)
                        if title:
                            if prev_child_had_data:
                                yield ('break', sml_token("break"))
                            yield ('heading', title)
                            last_text_char = title[-1]
                            current_child_had_data = True
                    elif name == 'table':
                        if prev_child_had_data:
                            yield ('break', sml_token("break"))
                        yield ('table', child)
                        current_child_had_data = True
                    else:
                        return_data = False
                        if name in proc_tags:
                            is_header = False
                            if prev_child_had_data and name in break_tags:
                                # A <span> is INLINE: only the markup's own whitespace
                                # separates it from what precedes it. <br>/<p> are boxes
                                # and always separate.
                                if name == 'span' and not lead_ws:
                                    yield ('glue', sml_token("break"))
                                elif name == 'span' and not first_char and not last_char:
                                    # A span with NO text at all cannot be a block —
                                    # it is a marker (EPUB pagebreak spans:
                                    # `<span epub:type="pagebreak" …></span>` mid-
                                    # sentence). NARROW on purpose: a span WITH text
                                    # keeps 'break', because `<span>Lance Wallnau
                                    # </span> <span>Author</span>` is the signature
                                    # case _close_block exists to protect, and the
                                    # broad form welded it into one line.
                                    yield ('sep', sml_token("break"))
                                else:
                                    yield ('break', sml_token("break"))
                            for inner in _tuple_row(child, last_text_char):
                                return_data = True
                                yield inner
                                if len(inner) > 1 and isinstance(inner[1], str) and inner[1]:
                                    last_text_char = inner[1][-1]
                                current_child_had_data = True
                                if inner[0] in ('text', 'heading') and isinstance(inner[1], str) and inner[1]:
                                    is_header = True
                            if return_data:
                                if name in break_tags and name != 'span':
                                    if is_header or (last_text_char and not last_text_char.isalnum() and not last_text_char.isspace()):
                                        yield ('break', sml_token("break"))
                                elif name in heading_tags or name in pause_tags:
                                    yield ('pause', sml_token("pause"))
                        else:
                            # Transparent inline tag (<em>, <i>, <b>, <a>, <sup>…): no
                            # break is emitted here, so the separator these rows get is
                            # the ' ' of the join in filter_chapter. Glue with a None
                            # payload says "restore NOTHING if this cannot be merged",
                            # which is exactly what the walk emitted before.
                            if prev_child_had_data and not lead_ws:
                                yield ('glue', None)
                            yield from _tuple_row(child, last_text_char)
                            current_child_had_data = True
                    # Whitespace state AFTER this tag. Block-level boxes always end the
                    # run; an inline tag carries its own trailing whitespace, and one
                    # that emitted nothing must not clear a space already pending.
                    if (name in heading_tags or name == 'table' or name in pause_tags
                            or (name in break_tags and name != 'span')):
                        ws_pending = True
                    elif current_child_had_data:
                        ws_pending = last_char.isspace()
                    else:
                        ws_pending = ws_pending or last_char.isspace()
                if current_child_had_data:
                    prev_child_had_data = True
        except Exception as e:
            error = f'filter_chapter() _tuple_row() error: {e}'
            DependencyError(error)
            return None

    def _num_repl(m):
        s = m.group(0)
        # leave years alone (already handled above)
        if re.fullmatch(r"\d{4}", s):
            return s
        n = float(s) if '.' in s else int(s)
        if is_num2words_compat:
            return num2words(n, lang=(lang_iso1 or 'en'))
        else:
            return math2words(m, lang, lang_iso1, tts_engine, is_num2words_compat)

    def _date_num_repl(m):
        # Inside a date span, a bare day-of-month (1-31) is read as an ordinal
        # ("March 12" -> "March twelfth"), not a cardinal. Years (4-digit) are
        # already converted above; anything else falls back to _num_repl.
        s = m.group(0)
        if re.fullmatch(r"\d{4}", s):
            return s
        if s.isdigit() and is_num2words_compat:
            n = int(s)
            if 1 <= n <= 31:
                return num2words(n, to='ordinal', lang=(lang_iso1 or 'en'))
        return _num_repl(m)

    try:
        msg = f'----------\nParsing doc {idx}'
        print(msg)
        session = context.get_session(session_id)
        if session and session.get('id', False):
            lang, lang_iso1, tts_engine = session['language'], session['language_iso1'], session['tts_engine']
            # h1..h6, all of them: an <h5>/<h6> outside this list falls into the
            # transparent-inline branch of _tuple_row — no [break] emitted, no
            # period appended — and is welded to the following prose verbatim.
            # Calibre is even told to page-break on h5 (see convert2epub), so
            # sub-sub-headings are real in this corpus.
            heading_tags = [f'h{i}' for i in range(1, 7)]
            break_tags = ['br', 'p', 'span']
            pause_tags = ['div']
            proc_tags = heading_tags + break_tags + pause_tags
            doc_body = doc.get_body_content()
            raw_html = doc_body.decode('utf-8') if isinstance(doc_body, bytes) else doc_body
            soup = BeautifulSoup(raw_html, 'html.parser')
            # ebooklib's get_body_content() returns ONE OF TWO SHAPES, and which one
            # depends on whether <body> carries attributes. It serializes the element
            # and strips the wrapper with a literal `startswith(b'<body>')` check, so:
            #   <body>            -> returns the INNER html, no body tag at all
            #   <body class="x">  -> the check fails, returns the WHOLE element
            # Calibre stamps class="calibre" onto every body, so for as long as every
            # book was force-converted this only ever saw the second shape and
            # `soup.body` always matched. A publisher/BookForge EPUB with a bare <body>
            # gives the first shape, `soup.body` is None, and this read as "no body
            # found" — silently skipping EVERY document and failing prepare with
            # "No chapters found!" on a book whose text was sitting right there
            # (measured: 21,255 characters in the document that reported no body).
            body = soup.body if soup.body is not None else soup
            if not body.get_text(strip=True):
                # Genuinely empty (a nav page, a spacer) — not an error, skip it.
                msg = 'No body found. Skip to next doc…'
                print(msg)
                return []
            # Skip known non-chapter types
            epub_type = body.get('epub:type', '').lower()
            if not epub_type:
                section_tag = soup.find('section')
                if section_tag:
                    epub_type = section_tag.get('epub:type', '').lower()
            excluded = {
                'frontmatter', 'backmatter', 'toc', 'titlepage', 'colophon',
                'acknowledgments', 'dedication', 'glossary', 'index',
                'appendix', 'bibliography', 'copyright-page', 'landmark'
            }
            if any(part in epub_type for part in excluded):
                msg = 'No body part. Skip to next doc…'
                print(msg)
                return []
            # remove scripts/styles
            for tag in soup(['script', 'style']):
                tag.decompose()
            # Resolve the whitespace-fidelity markers before anything reads the
            # rows: what leaves _collapse_glue is the same ('text'|'heading'|
            # 'break'|'pause'|'table') stream the loop below has always consumed.
            tuples_list = _collapse_glue(list(_tuple_row(body)))
            if not tuples_list:
                error = 'No tuples_list from body created!'
                print(error)
                return None
            msg = f'Parsing xhtml markers…'
            print(msg)
            # Tolerant title normalization for dedup: lowercase, collapse
            # whitespace, drop trailing sentence/colon punctuation, and fold
            # common smart punctuation so near-identical titles still match.
            def _norm_title(s):
                s = (s or '').strip()
                s = (s.replace('’', "'").replace('‘', "'")
                       .replace('“', '"').replace('”', '"')
                       .replace('–', '-').replace('—', '-'))
                s = re.sub(r'\s+', ' ', s)
                s = re.sub(r'[.!?…:]+$', '', s)
                return s.lower().strip()
            # Build set of normalized TOC titles for detecting chapter titles
            # that lost their heading tags (e.g. after AI cleanup converted <h2> to <p>)
            toc_titles_normalized = set()
            for t in session.get('chapter_titles', []):
                ct = _norm_title(t)
                if ct:
                    toc_titles_normalized.add(ct)
            if toc_titles_normalized:
                print(f'[HEADING] {len(toc_titles_normalized)} TOC titles loaded for heading detection')
            text_list = []
            handled_tables = set()
            prev_typ = None
            last_heading_normalized = None  # Track last heading to deduplicate body text
            chapter_title_normalized = None  # First heading of the chapter — suppress later echoes anywhere
            sml_statics = {v['static'] for v in TTS_SML.values() if 'static' in v}

            def _close_block(items):
                # A BLOCK BOUNDARY FOLLOWS THE LAST ITEM. A standalone line that
                # carries no terminal punctuation — a signature ("Lance Wallnau",
                # then "Author" on its own line), a label, a title paragraph the
                # TOC never listed — would otherwise weld to the next block into
                # one sentence: the flattened text keeps the boundary only as an
                # SML token, and the "remove any [break] between words" pass
                # below deletes that token whenever WORD characters flank it.
                # The period is the same treatment headings get at append time,
                # and it also makes the token survive that pass (a '.' is not a
                # word character), so the sentence break AND the paragraph pause
                # both stand.
                #
                # GATED ON A WORD CHARACTER, not on "lacks .!?…": a line ending
                # in a closing quote («She said, "Hello."»), a comma, a colon or
                # a dash already says how it ends, and a second mark after it
                # would be read aloud as a stumble. Only a bare letter or digit
                # at the end is a line that never got to say so.
                if items:
                    last = items[-1]
                    if last not in sml_statics and last and last[-1].isalnum():
                        items[-1] = last + '.'

            for typ, payload in tuples_list:
                if typ == 'heading':
                    _close_block(text_list)
                    if not session.get('skip_headings', False):
                        title = payload.strip()
                        norm = _norm_title(title)
                        # Skip a heading that just repeats one we already voiced
                        # (e.g. two consecutive heading tags with the same text).
                        if norm and norm == last_heading_normalized:
                            print(f'[HEADING] Skipping duplicate heading: "{title}"')
                            prev_typ = typ
                            continue
                        # Add period to chapter titles so TTS pauses after them
                        if title and title[-1] not in '.!?…':
                            title += '.'
                        # …and MARK it, so the period is not the only thing the
                        # splitter knows (2026-08-27). A period alone made this a
                        # short punctuated row and nothing more: all three merge
                        # passes in get_sentences then glued 'Prologue.' onto the
                        # paragraph under it, because 9 chars is far below the
                        # 25-char min-chars floor. The marker is what makes a
                        # header its own chunk. Only headings that are actually
                        # VOICED get marked — the skip_headings branch above has
                        # already dropped the rest, and marking is not a toggle.
                        text_list.append(sml_heading(title))
                        last_heading_normalized = norm
                        if chapter_title_normalized is None:
                            chapter_title_normalized = norm
                elif typ in ('break', 'pause'):
                    _close_block(text_list)
                    if prev_typ != typ:
                        text_list.append(sml_token(typ))
                    # Don't clear last_heading — breaks often sit between heading and duplicate text
                elif typ == 'sep':
                    # An inline whitespace separator: the [break] token is kept
                    # (so a pagebreak span still pauses where nothing flanks it)
                    # but the block is NOT closed — no manufactured period, and
                    # break_between_alnum_re downstream turns the token back into
                    # the plain space the markup meant when words flank it.
                    if prev_typ not in ('break', 'pause', 'sep'):
                        text_list.append(sml_token('break'))
                elif typ == 'table':
                    _close_block(text_list)
                    last_heading_normalized = None
                    table = payload
                    if table in handled_tables:
                        prev_typ = typ
                        continue
                    handled_tables.add(table)
                    rows = table.find_all('tr')
                    if not rows:
                        prev_typ = typ
                        continue
                    headers = [c.get_text(strip=True) for c in rows[0].find_all(['td', 'th'])]
                    for row in rows[1:]:
                        cells = [c.get_text(strip=True).replace('\xa0', ' ') for c in row.find_all('td')]
                        if not cells:
                            continue
                        if len(cells) == len(headers) and headers:
                            line = ' — '.join(f'{h}: {c}' for h, c in zip(headers, cells))
                        else:
                            line = ' — '.join(cells)
                        if line:
                            text_list.append(line.strip())
                else:
                    text = payload.strip()
                    if text:
                        text_check = _norm_title(text)
                        # Deduplicate: skip body text that repeats the heading we
                        # just added OR the chapter title (catches a non-adjacent
                        # echo, e.g. a styled title paragraph after a blank line).
                        if text_check and (text_check == last_heading_normalized or text_check == chapter_title_normalized):
                            print(f'[HEADING] Skipping duplicate body text: "{text}"')
                            last_heading_normalized = None
                            prev_typ = typ
                            continue
                        last_heading_normalized = None
                        # Check if this text matches a TOC chapter title whose heading tag was lost
                        if text_check in toc_titles_normalized:
                            if text[-1] not in '.!?…':
                                text += '.'
                            print(f'[HEADING] Detected chapter title from TOC match: "{text}"')
                            # A title recovered from the TOC is a heading in
                            # everything but markup, so it is marked like one and
                            # reads as its own chunk too (2026-08-27). NOTE it is
                            # NOT gated on skip_headings, and never was: this row
                            # is body text that happens to match a TOC entry, and
                            # skip_headings only ever suppressed real h1-h6 tags.
                            # Marking it does not change what is spoken, only how
                            # it is chunked.
                            text_list.append(sml_heading(text))
                        else:
                            text_list.append(text)
                prev_typ = typ
            # The document's end is a block boundary too — a chapter whose last
            # line is a bare signature closes the same way one mid-chapter does.
            _close_block(text_list)
            msg = f'Flattening as raw text…'
            print(msg)
            # Voxtral and Orpheus both read better multi-sentence: packing 2-3
            # sentences per generation keeps timbre/prosody coherent across a passage
            # instead of making every sentence an independent "take". (The stray
            # syllable/gibberish at sentence starts once blamed on Orpheus
            # multi-sentence packing was actually a prompt-framing bug — a stray BOS
            # in the vLLM prompt — since fixed in orpheus.py _format_prompt_ids.
            # Packing is safe again.) Orpheus caps WELL below Voxtral: fine-tuned
            # voices are trained on short clips and start emitting the end-of-audio
            # token EARLY once a chunk runs well past ~300 chars / ~20s of audio,
            # silently dropping the trailing text (measured on a real book: the p90
            # speaking rate blows up to an impossible ~21 ch/s at 400-449 chars — the
            # model finished with a clean EOS well under MAX_AUDIO_TOKENS, so the
            # token-cap re-split never fired; 450-char packing failed whisper-diff on
            # every model tested). The cap stays inside the reliable zone for EOS-safe
            # (≤20s/2048-recipe) voices; orpheus.py's duration-vs-text guard catches
            # any chunk that still truncates. Same ORPHEUS_MAX_CHARS env as the
            # packing cap below so both passes agree; invalid value raises (NO FALLBACK).
            if tts_engine == 'voxtral':
                max_chars = language_mapping[lang]['max_chars'] * 2  # ~500 chars
            elif tts_engine == 'orpheus':
                _mc = os.environ.get('ORPHEUS_MAX_CHARS')
                max_chars = int(_mc) if _mc else 350
            else:
                max_chars = int(language_mapping[lang]['max_chars'] / 1.5)
            clean_list = []
            i = 0
            while i < len(text_list):
                current = text_list[i]
                if current in {v['static'] for v in TTS_SML.values() if "static" in v}:
                    if clean_list:
                        prev = clean_list[-1]
                        if prev in {v['static'] for v in TTS_SML.values() if "static" in v}:
                            i += 1
                            continue
                    clean_list.append(current)
                    i += 1
                    continue
                clean_list.append(current)
                i += 1
            text = ' '.join(clean_list)
            if not re.search(r"[^\W_]", text):
                error = 'No valid text found!'
                print(error)
                return None
            # clean SML tags badly coded
            bool, text = normalize_sml_tags(text)
            if bool is False:
                print(text)
                if session['is_gui_process']:
                    show_alert({"type": "warning", "msg": text})
                return None
            # remove any [break] between words or cutting words
            # Skip break removal in sentence_per_paragraph mode - we need those breaks!
            break_token = re.escape(sml_token('break'))
            strip_break_spaces_re = re.compile(rf'\s*{break_token}\s*')
            text = strip_break_spaces_re.sub(sml_token('break'), text)

            # In sentence_per_paragraph mode, split on breaks NOW before escape_sml replaces them
            # Each paragraph becomes one sentence, preserving the original structure
            if session.get('sentence_per_paragraph', False):
                msg = 'Sentence-per-paragraph mode: preserving paragraph boundaries…'
                print(msg)
                paragraphs = text.split(sml_token('break'))
                sentences = [p.strip() for p in paragraphs if p.strip()]
                if len(sentences) == 0:
                    error = 'No sentences found!'
                    print(error)
                    return None
                print(f'[sentence_per_paragraph] Extracted {len(sentences)} paragraphs as sentences')
                return sentences

            break_between_alnum_re = re.compile(rf'(?<=[\w]){break_token}(?=[\w])', flags=re.UNICODE)
            text = break_between_alnum_re.sub(' ', text)
            # escape all SML tags to not be touched by any text treatment
            text, sml_blocks = escape_sml(text)
            if tts_engine == TTS_ENGINES['ORPHEUS']:
                # Orpheus fine-tunes are trained on book-exact text, so the whole
                # date/year/roman/clock/math pipeline is SKIPPED — digits like
                # '5,000', '1930s', '7th' and romans like 'Henry VIII' stay as
                # printed. The ONLY lexical transforms the fine-tunes know
                # (scripture refs + bare-integer expansion, orpheus_text.py) run
                # at the ENGINE boundary (_clean_sentence_for_tts), NOT here: the
                # sentences stored below — and the m4b transcript built from
                # them — therefore read like the book, while get_sentences packs
                # against the TRANSFORMED length so the 350-char cap bounds what
                # the model actually reads. Orpheus is English-only by design;
                # any other language is an XTTS job — fail loudly rather than
                # anglicize.
                if lang != 'eng':
                    error = f"Orpheus is English-only (got '{lang}') — route this language to another engine (XTTS)."
                    print(error)
                    return None
                msg = 'Orpheus: book-exact sentences (lexical transforms deferred to the engine)…'
                print(msg)
            elif stanza_nlp:
                msg = 'Converting dates and years to words…'
                print(msg)
                re_ordinal = re.compile(
                    r'(?<!\w)(0?[1-9]|[12][0-9]|3[01])(?:\s|\u00A0)*(?:st|nd|rd|th)(?!\w)',
                    re.IGNORECASE
                )
                re_num = re.compile(r'(?<!\w)[-+]?\d+(?:\.\d+)?(?!\w)')
                text = unicodedata.normalize('NFKC', text).replace('\u00A0', ' ')
                if re_num.search(text) and re_ordinal.search(text):
                    date_spans = get_date_entities(text, stanza_nlp)
                    if date_spans is None:
                        # NER crashed (error already printed by get_date_entities);
                        # do not silently fall through to the "no dates" branch.
                        error = 'Date entity recognition failed!'
                        print(error)
                        return None
                    if date_spans:
                        result = []
                        last_pos = 0
                        for start, end, date_text in date_spans:
                            result.append(text[last_pos:start])
                            # 1) convert 4-digit years (your original behavior)
                            processed = re.sub(
                                r"\b\d{4}\b",
                                lambda m: year2words(m.group(), lang, lang_iso1, is_num2words_compat),
                                date_text
                            )
                            # 2) convert ordinal days like "16th"/"16 th"->"sixteenth"
                            if is_num2words_compat:
                                processed = re_ordinal.sub(
                                    lambda m: num2words(int(m.group(1)), to='ordinal', lang=(lang_iso1 or 'en')),
                                    processed
                                )
                            else:
                                processed = re_ordinal.sub(
                                    lambda m: math2words(m.group(), lang, lang_iso1, tts_engine, is_num2words_compat),
                                    processed
                                )
                            # 3) day-of-month -> ordinal; other numbers -> _num_repl
                            processed = re_num.sub(_date_num_repl, processed)
                            result.append(processed)
                            last_pos = end
                        result.append(text[last_pos:])
                        text = ' '.join(result)
                    else:
                        if is_num2words_compat:
                            text = re_ordinal.sub(
                                lambda m: num2words(int(m.group(1)), to='ordinal', lang=(lang_iso1 or 'en')),
                                text
                            )
                        else:
                            text = re_ordinal.sub(
                                lambda m: math2words(m.group(1), lang, lang_iso1, tts_engine, is_num2words_compat),
                                text
                            )
                        text = re.sub(
                            r"\b\d{4}\b",
                            lambda m: year2words(m.group(), lang, lang_iso1, is_num2words_compat),
                            text
                        )
            if tts_engine != TTS_ENGINES['ORPHEUS']:
                # year-form ALL remaining 4-digit years (1000-2099), even outside a
                # stanza-detected date span ("October 1933" -> "nineteen thirty-three").
                text = re.sub(r"(?<!\d)(1\d{3}|20\d{2})(?!\d)",
                              lambda m: year2words(m.group(), lang, lang_iso1, is_num2words_compat), text)
                msg = 'Convert romans to numbers…'
                print(msg)
                text = roman2number(text)
                msg = 'Convert time to words…'
                print(msg)
                text = clock2words(text, lang, lang_iso1, tts_engine, is_num2words_compat)
                msg = 'Convert numbers, maths signs to words…'
                print(msg)
                text = math2words(text, lang, lang_iso1, tts_engine, is_num2words_compat)
            msg = 'Normalize text…'
            print(msg)
            text = normalize_text(text, lang, lang_iso1, tts_engine)

            msg = f'Get sentences…'
            print(msg)
            sentences = get_sentences(text, session_id, sml_blocks)
            # get_sentences returns None on a genuine failure and [] for empty text.
            # The old `if sentences and len(sentences)==0` could never be true and
            # did not guard None, so a None fell through to the list comprehension
            # below → TypeError → swallowed → chapter dropped. Handle None explicitly.
            if sentences is None:
                error = 'Failed to split chapter text into sentences'
                print(error)
                return None
            sentences = [restore_sml(s, sml_blocks) for s in sentences]
            return sentences
        return None
    except Exception as e:
        error = f'filter_chapter() error: {e}'
        DependencyError(error)
        return None

def _normalize_for_dup(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace — the cheap normal form
    for near-duplicate sentence detection in the Orpheus packer (see
    _is_near_duplicate). Uses the module's `regex as re`."""
    s = re.sub(r'[^\w\s]', ' ', s.lower())
    return re.sub(r'\s+', ' ', s).strip()


def _is_near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    """True when a and b are near-identical prose — the pattern that primes
    Orpheus into a repetition loop (real case: "Kershaw didn't use it in his
    book." / "Trevor-Roper didn't use it in his book."). One spoken sentence is
    hundreds of audio tokens, so such a loop sits entirely OUTSIDE MLX's ~20-token
    repetition-penalty window; keeping the priming pair out of one generation is
    the cheap structural fix.

    Only meaningful for real sentences: both sides must be >= 4 words (short
    dialogue like "Yes."/"No." repeats naturally and is NOT a loop primer) and
    within a [0.6, 1.67] length ratio before the pricier SequenceMatcher ratio is
    even computed. stdlib difflib only."""
    wa, wb = _normalize_for_dup(a).split(), _normalize_for_dup(b).split()
    if len(wa) < 4 or len(wb) < 4:
        return False
    la, lb = len(wa), len(wb)
    if min(la, lb) / max(la, lb) < 0.6:   # length ratio outside [0.6, 1.67]
        return False
    # Ratio over the WORD sequences, not the raw char stream: the repetition
    # primer differs only by a subject ("Kershaw" vs "Trevor-Roper"), and a
    # different-length name prefix drags a char-level ratio under 0.8 even though
    # the sentences are clearly the same template. Word-level ratio scores that
    # real case at ~0.82 while unrelated similar-length prose stays far below.
    return difflib.SequenceMatcher(None, wa, wb).ratio() >= threshold


def _split_into_sentences_for_dup(text: str) -> list:
    """Split a packed prose chunk back into its component sentences for the
    near-duplicate scan. Mirrors the boundary the Orpheus packer's _split_to_cap
    uses (terminal .!?… plus trailing quotes/brackets)."""
    parts, last = [], 0
    for m in re.finditer(r'[.!?…]["\'”’)\]]*\s+', text):
        parts.append(text[last:m.end()].strip())
        last = m.end()
    if last < len(text):
        parts.append(text[last:].strip())
    return [p for p in parts if p]


def _twin_anchor_grams(s: str) -> set:
    """The qualifying 4-grams of a sentence, normalized. A qualifying 4-gram is
    4 consecutive normalized words totalling >= 14 chars — long enough that two
    copies inside one generation are a real TWIN ANCHOR (the skip-ahead primer,
    measured 2026-08-29 on Gods People: attention resolves to the wrong copy and
    silently deletes the words between — 'which made him unacceptable to X, or
    …, which made him unacceptable to Y' lost its middle 85 chars), while
    stock collocations ('at the end of') stay below the length bar."""
    w = _normalize_for_dup(s).split()
    return {tuple(w[i:i + 4]) for i in range(len(w) - 3)
            if sum(len(t) for t in w[i:i + 4]) >= 14}


def _split_intra_twin(sent: str) -> list:
    """Split ONE sentence that contains the same qualifying 4-gram twice, at a
    comma/semicolon/dash between the two copies — the only safe cut points a
    mid-sentence boundary has. Each piece must keep its own copy of the anchor
    and be >= 25 chars, else the sentence is returned unsplit (safe no-op).
    Only the FIRST twin is targeted; the recursion handles pathological text
    with several."""
    grams = _twin_anchor_grams(sent)
    words = _normalize_for_dup(sent).split()
    twin = None
    seen = {}
    for i in range(len(words) - 3):
        g = tuple(words[i:i + 4])
        if g not in grams:
            continue
        if g in seen and i - seen[g] >= 4:   # non-overlapping copies only
            twin = g
            break
        seen.setdefault(g, i)
    if twin is None:
        return [sent]
    phrase = ' '.join(twin)
    for m in re.finditer(r'[,;—–]\s+', sent):
        left, right = sent[:m.end()].strip(), sent[m.end():].strip()
        if len(left) < 25 or len(right) < 25:
            continue
        if phrase in _normalize_for_dup(left) and phrase in _normalize_for_dup(right):
            return [left] + _split_intra_twin(right)
    return [sent]


def _split_near_dup_chunk(chunk: str) -> list:
    """Split one packed Orpheus chunk so no single generation contains a
    repetition primer: a near-duplicate sentence pair (the LOOP primer; see
    _is_near_duplicate) or a repeated qualifying 4-gram (the SKIP primer; see
    _twin_anchor_grams — the same fragility mirrored forward). Walk the chunk's
    sentences, starting a NEW sub-chunk whenever the next sentence
    near-duplicates one already in the current sub-chunk OR shares a qualifying
    4-gram with it; a twin inside a single sentence is pre-split at a comma
    between the copies (_split_intra_twin). Returns [chunk] UNCHANGED (exact
    original text) when nothing splits, so non-repetitive prose keeps its
    packing boundaries byte-for-byte."""
    sents = _split_into_sentences_for_dup(chunk)
    if not sents:
        return [chunk]
    pieces = []
    for s in sents:
        pieces.extend(_split_intra_twin(s))
    if len(pieces) < 2:
        return [chunk]
    groups = [[pieces[0]]]
    group_grams = _twin_anchor_grams(pieces[0])
    for s in pieces[1:]:
        sg = _twin_anchor_grams(s)
        if any(_is_near_duplicate(m, s) for m in groups[-1]) or (sg & group_grams):
            groups.append([s])
            group_grams = sg
        else:
            groups[-1].append(s)
            group_grams |= sg
    if len(groups) < 2:
        return [chunk]
    return [' '.join(g) for g in groups]


def _apply_near_dup_split(chunks: list) -> list:
    """Run _split_near_dup_chunk over a packed Orpheus chunk list (a no-op for
    non-repetitive prose). Splitting only ever shortens a chunk, so no result can
    exceed the char/sentence budget the packer already enforced."""
    out = []
    for c in chunks:
        out.extend(_split_near_dup_chunk(c))
    return out


def _strip_escaped_sml(s:str)->str:
    return ''.join(c for c in s if ord(c) < sml_escape_tag)


def _has_word_chars(s:str)->bool:
    """True when a row has anything to SAY: at least one word character once the
    escaped SML tokens are gone. Unicode-aware by default in Python 3, so a CJK
    or Cyrillic row answers yes exactly as an English one does.

    Callers pass the row through the engine's own text transform first (see
    has_words in get_sentences) so the question is asked of what the MODEL will
    read, not of what is on screen."""
    return bool(re.search(r'\w', _strip_escaped_sml(s)))


def _drop_wordless_rows(rows:list, has_words)->list:
    """Remove every row that would reach TTS as text with NO word character in
    it (2026-08-29).

    ── The bug ────────────────────────────────────────────────────────────────

    Hugh Howey's "Shift" carries a decorated section header on every silo
    chapter: `<h2>• Silo 1 •</h2>`. conf_lang maps the bullet to a period, so
    filter_chapter marked and emitted the row as `[heading]. Silo 1 ..` — and
    PASS 1 then split it at that FIRST period, leaving `[heading].` behind as a
    row of its own. A heading is exempt from the min-chars floor in both
    directions, so nothing could merge it away, and 30 chunks whose entire text
    was '.' were handed to Orpheus in one book. The model has no period sound to
    make, so it improvised: [ORPHEUS][SHORT_CHUNK_OVERRUN] sentence=509 chars=1
    seconds=1.621 allowed=1.090 ratio=1.49 text='.' — 1.6s of non-speech shipped
    into the audiobook.

    ── Why DROP and not merge ─────────────────────────────────────────────────

    A row with no word character is decoration — a scene-break rule, a bullet, a
    dingbat — and it carries no audio content at all. Merging it into a
    neighbour would not add anything spoken; it would prepend orphan punctuation
    to that neighbour's prompt, which is the exact shape ('Silo 1 ..') Owen
    already called a defect in b499c33f. So the row goes, and the log says so.

    THIS IS NOT THE FLOOR'S 'too short' TEST and it does not care about length: a
    250-character rule of asterisks is just as wordless as a lone period. It is
    also not a fallback — nothing is substituted for the row, and nothing spoken
    is lost, because there was nothing spoken in it.

    SML-ONLY ROWS ARE NOT TOUCHED. A row whose core is empty (a bare [break]) is
    not wordless-with-text: the engines never send it to the model at all —
    orpheus.py's convert() writes silence for it — so it is a real pause and it
    stays. Only a row that has surviving TEXT with no word in that text is
    dropped, and the SML tokens riding on it go with it (counted in the log, the
    same ratified trade-off the merge passes make)."""
    out = []
    for row in rows:
        core = _split_sml_edges(row)[1]
        if core and not has_words(row):
            tokens = sum(1 for c in row if ord(c) >= sml_escape_tag)
            # Same 'get_sentences() <pass>: <what changed>' shape as the floor's
            # own lines, but NOT labelled as the floor: this also runs on the
            # ideogram path, which has no floor, and it is not a length decision.
            print(f'get_sentences() wordless row: dropped, nothing to speak, '
                  f'{tokens} pause token(s) with it: {_strip_escaped_sml(core)!r}')
            continue
        out.append(row)
    return out


def _sentence_min_chars()->int:
    """Minimum engine-read length (in chars) a text row may have before
    _apply_min_chars_floor merges it into a neighbour. SENTENCE_MIN_CHARS
    overrides the 25-char default and 0 disables the pass; an invalid value
    raises (NO FALLBACK), same handling style as ORPHEUS_MAX_CHARS."""
    _mn = os.environ.get('SENTENCE_MIN_CHARS')
    min_chars = int(_mn) if _mn else 25
    if min_chars < 0:
        raise ValueError(f'SENTENCE_MIN_CHARS must be >= 0, got {min_chars}')
    return min_chars


def _heading_min_words()->int:
    """Minimum word count a section heading may have and still stand as its own
    chunk. Below it the heading is merged FORWARD into the next row (2026-08-29,
    Owen's ruling): Orpheus can fail to voice an ultra-short prompt at all — the
    SNAC frames are coarse enough that a one-word chunk ('No.', 'II.') can hit
    EOS before any speech exists — and an unread chapter title is worse than one
    that flows into its first paragraph. HEADING_MIN_WORDS overrides the
    3-word default and 0 disables the pass; an invalid value raises (NO
    FALLBACK), same handling style as SENTENCE_MIN_CHARS."""
    _mw = os.environ.get('HEADING_MIN_WORDS')
    min_words = int(_mw) if _mw else 3
    if min_words < 0:
        raise ValueError(f'HEADING_MIN_WORDS must be >= 0, got {min_words}')
    return min_words


def _word_count(core:str)->int:
    """Words in a row's core as the reader would count them: whitespace-split
    tokens that carry at least one word character, SML stripped. Counted on the
    display text — the digit/number transforms have already run by the time any
    merge pass sees a row, so display and engine text agree on word count."""
    return sum(1 for t in _strip_escaped_sml(core).split() if re.search(r'\w', t))


def _merge_short_headings_forward(rows:list, clean_len, max_chars:int, is_heading, min_words:int)->list:
    """Merge every heading of fewer than min_words words FORWARD into the next
    text row, so no ultra-short prompt is ever handed to the engine as its own
    chunk (2026-08-29).

    ── Why forward, and why headings at all ────────────────────────────────────

    Orpheus has a known failure mode on very short prompts: the model can emit
    EOS before any voiced audio exists, so a chunk like 'No.' or 'II.' may not
    be READ AT ALL. Ultra-short chunks are almost exclusively chapter titles —
    every other short row is already merged by the min-chars floor; headings
    were the one exemption — and a chapter title belongs to the text UNDER it,
    so it merges into the next row, never the previous one. This deliberately
    narrows the 2026-08-27 heading isolation: a title of min_words or more still
    stands alone; a shorter one trades its isolation (and its bold VTT cue) for
    the guarantee of being spoken. Being read correctly beats standing alone.

    ── Mechanics ───────────────────────────────────────────────────────────────

    The merged row is the TARGET row with the heading's text prepended after the
    target's leading SML tokens — so the target's own lead pause still plays
    before the combined text, and when the target is ITSELF a heading (stacked
    chapter-number headings: '16.' then '2110.') the target's [heading] marker
    survives and the combined row remains a heading. The demoted heading's own
    edge tokens are DROPPED and counted, the same ratified trade-off every merge
    pass makes; its [heading] marker sits on that dropped lead, which is what
    demotes it. The loop re-examines the merged row, so stacked short headings
    coalesce until the combined title reaches min_words or the next row is body
    text.

    A merge that would break max_chars, and a heading with no following text row
    in its chapter, keep the heading isolated (logged) — the render-side re-roll
    backstop still guards those.

    Runs BEFORE the min-chars floor's own loop and independently of it: the
    floor is about prompts too short to sound natural, this is about prompts too
    short to be voiced at all, and disabling one must not disable the other."""
    if min_words <= 0:
        return rows
    out = list(rows)
    i = 0
    while i < len(out):
        if not is_heading(out[i]):
            i += 1
            continue
        lead, core, trail = _split_sml_edges(out[i])
        if not core or _word_count(core) >= min_words:
            i += 1
            continue
        # Forward to the next row carrying text; SML-only rows between are join
        # fuel, same as the floor's forward merge.
        j = i + 1
        while j < len(out) and not _split_sml_edges(out[j])[1]:
            j += 1
        if j >= len(out):
            print(f'get_sentences() short-heading merge: no following text in this chapter, heading kept: {_strip_escaped_sml(core)!r}')
            i += 1
            continue
        t_lead, t_core, t_trail = _split_sml_edges(out[j])
        merged = f'{t_lead}{core} {t_core}{t_trail}'
        if clean_len(merged) > max_chars:
            print(f'get_sentences() short-heading merge: merge would break max_chars, heading kept: {_strip_escaped_sml(core)!r}')
            i += 1
            continue
        dropped = len(lead) + len(trail) + sum(
            len(_split_sml_edges(r)[0]) + len(_split_sml_edges(r)[2]) for r in out[i + 1:j]
        )
        out[i:j + 1] = [merged]
        print(f'get_sentences() short-heading merge: merged {_word_count(core)}-word heading forward, '
              f'{dropped} join pause token(s) dropped: {_strip_escaped_sml(core)!r}')
        # No i += 1: the merged row may itself be a still-short heading (stacked
        # titles), and the next pass over it is what coalesces the stack.
    return out


def _split_sml_edges(row:str)->tuple:
    """Split a row into (leading SML tokens, plain core, trailing SML tokens).
    An escaped SML token is ONE char with ord >= sml_escape_tag — escape_sml
    replaces each <break>/<pause>/… block with chr(sml_escape_tag + i) — so the
    edges are found by walking off the SML chars and whitespace at both ends.
    The core is returned verbatim, mid-row tokens included, so callers can refuse
    it: orpheus.py's _classify_gap realizes only the LEADING and TRAILING tokens
    of a row as silence, and a token left mid-row is stripped before TTS with its
    pause silently discarded."""
    i, j = 0, len(row)
    while i < j and (row[i].isspace() or ord(row[i]) >= sml_escape_tag):
        i += 1
    while j > i and (row[j - 1].isspace() or ord(row[j - 1]) >= sml_escape_tag):
        j -= 1
    lead = ''.join(c for c in row[:i] if ord(c) >= sml_escape_tag)
    trail = ''.join(c for c in row[j:] if ord(c) >= sml_escape_tag)
    return lead, row[i:j].strip(), trail


def _has_escaped_sml(s:str)->bool:
    return any(ord(c) >= sml_escape_tag for c in s)


def _heading_row_test(sml_blocks:list[str]):
    """Build THE predicate every merge pass asks before it glues a row to a
    neighbour: "is this row a section heading?" (2026-08-27)

    filter_chapter marks a heading with the [heading] token on the row's leading
    edge, and escape_sml has since replaced that token with ONE char whose INDEX
    into sml_blocks is its whole identity — inside get_sentences an escaped token
    is otherwise opaque, which is why sml_blocks has to come in with the text.
    So the question reduces to "does this row carry a char that stands for
    [heading]", and the answer is precomputed once per chapter.

    ONE predicate, built here and passed to all three merge passes
    (_apply_min_chars_floor, the Orpheus PASS 5 packer, the Voxtral packer).
    Three hand-rolled edge checks would drift apart, and the existing
    _has_escaped_sml/_plain tests cannot answer this: they look at a row's CORE,
    and the marker sits on the LEAD, where they are blind to it.

    The whole row is searched rather than just its lead, so a row that carries
    the marker anywhere — however it got there — is one no pass may merge."""
    marks = set()
    for i, block in enumerate(sml_blocks):
        m = SML_TAG_PATTERN.fullmatch(block)
        if m and m.group('tag') == 'heading':
            marks.add(chr(sml_escape_tag + i))

    def _is_heading_row(row:str)->bool:
        return bool(marks) and any(c in marks for c in row)

    return _is_heading_row


def _apply_min_chars_floor(rows:list, clean_len, max_chars:int, min_chars:int, is_heading, has_words)->list:
    """Merge every row whose engine-read text is shorter than min_chars into a
    neighbour, so a one-word paragraph ('No.') is never handed to TTS as its own
    ultra-short prompt. FORWARD first (the tiny row leads into the sentence that
    follows it); BACKWARD only when the forward merge would break max_chars.

    HEADINGS ARE EXEMPT, IN BOTH DIRECTIONS (2026-08-27). A section header is
    read as its own chunk, so it is never merged away and nothing is ever merged
    into it — this pass was the main reason headers used to be spoken as part of
    the paragraph under them, since almost every header ('Prologue.', 'II.',
    'Chapter 8: State of Confusion.') is shorter than the 25-char floor. This is
    the one exemption that IGNORES length: a two-character heading still stands
    alone. is_heading is the shared predicate from _heading_row_test.

    NARROWED 2026-08-29: the exemption is for headings of HEADING_MIN_WORDS
    words or more. A shorter heading is merged FORWARD by
    _merge_short_headings_forward before this loop runs — Orpheus can fail to
    voice an ultra-short prompt at all, and an unread chapter title is worse
    than one that flows into its first paragraph.

    THE EXEMPTION IS FOR HEADINGS WITH WORDS IN THEM (2026-08-29).
    _drop_wordless_rows runs FIRST, so by the time the exemption is consulted no
    row — heading or not — is still wordless. 'II.' stands alone; the '.' left over
    from a `<h2>• Silo 1 •</h2>` never reaches this pass to be exempted. That
    order is the fix: the exemption made a wordless heading unmergeable, which is
    how 30 chunks reading '.' shipped in one book (see _drop_wordless_rows).

    THIS PASS IS THE LAST WORD ON WORDLESSNESS for every engine, because it runs
    last on every return path. Neither packer can undo it — both only ever
    CONCATENATE rows, and concatenation cannot remove a word character — and both
    run before it. _apply_near_dup_split is the only pass after it, and it cannot
    reintroduce the case either: it splits a chunk only at a near-duplicate
    sentence pair, and _is_near_duplicate requires FOUR words on both sides, so a
    wordless fragment can never open a new sub-chunk.

    RATIFIED TRADE-OFF: the SML tokens sitting AT THE JOIN — the tiny row's
    trailing token, any SML-only rows between the two, and the neighbour's
    leading token — are DROPPED and the pause they encode is lost. They cannot be
    carried into the merged row: a token in the middle of a row is stripped
    before TTS anyway (only edge tokens are realized as silence), so keeping it
    would lose the same pause while hiding the loss. The pause INTO the merged
    row survives on its leading edge and the pause out of it on its trailing
    edge; every dropped join is logged.

    Rows with no plain text (SML-only) are never themselves 'tiny': they are join
    fuel when consumed between two merged rows, and left untouched otherwise.

    clean_len measures what the engine will actually read (SML stripped, and for
    Orpheus through the digit/scripture transform), so it must be passed in by
    the caller that owns that transform."""
    # BEFORE the early return: a wordless row must never ship even when the
    # length floor is switched off (SENTENCE_MIN_CHARS=0). The two rules are
    # independent — one is about a row being too SHORT, the other about it having
    # nothing to say at all.
    rows = _drop_wordless_rows(rows, has_words)
    # Also before the early return, and reading its own knob: a heading too
    # short to be VOICED (see _merge_short_headings_forward) is a different
    # defect from a row too short to sound natural, and SENTENCE_MIN_CHARS=0
    # must not switch this off.
    rows = _merge_short_headings_forward(rows, clean_len, max_chars, is_heading, _heading_min_words())
    if min_chars <= 0:
        return rows
    out = list(rows)
    i = 0
    while i < len(out):
        lead, core, trail = _split_sml_edges(out[i])
        if is_heading(out[i]):
            # Its own chunk, whatever its length. Reported only when the floor
            # would otherwise have eaten it, so the log says what changed.
            if core and clean_len(out[i]) < min_chars:
                print(f'get_sentences() min-chars floor: heading kept as its own row: {_strip_escaped_sml(core)!r}')
            i += 1
            continue
        if not core or clean_len(out[i]) >= min_chars:
            i += 1
            continue
        if _has_escaped_sml(core):
            print(f'get_sentences() min-chars floor: mid-row SML token, cannot merge short row: {_strip_escaped_sml(out[i])!r}')
            i += 1
            continue
        # FORWARD — the next row carrying text; every row stepped over on the way
        # is SML-only and is consumed as join fuel.
        j = i + 1
        while j < len(out) and not _split_sml_edges(out[j])[1]:
            j += 1
        if j < len(out):
            next_lead, next_core, next_trail = _split_sml_edges(out[j])
            # …and never merge a short row INTO a heading: the header would stop
            # being the chunk's whole content, which is the point of marking it.
            if not _has_escaped_sml(next_core) and not is_heading(out[j]):
                merged = f'{lead}{core} {next_core}{next_trail}'
                if clean_len(merged) <= max_chars:
                    dropped = len(trail) + len(next_lead) + sum(
                        len(_split_sml_edges(r)[0]) + len(_split_sml_edges(r)[2]) for r in out[i + 1:j]
                    )
                    out[i:j + 1] = [merged]
                    print(f'get_sentences() min-chars floor: merged short row forward, {dropped} join pause token(s) dropped: {_strip_escaped_sml(core)!r}')
                    continue
        # BACKWARD — symmetric: prev's trailing token and this row's leading token
        # are the join and are dropped; this row's trailing token is hoisted to the
        # merged row's trailing edge so the pause out of it still plays.
        k = i - 1
        while k >= 0 and not _split_sml_edges(out[k])[1]:
            k -= 1
        if k >= 0:
            prev_lead, prev_core, prev_trail = _split_sml_edges(out[k])
            # Symmetric refusal: a heading is not a landing site backwards either.
            if not _has_escaped_sml(prev_core) and not is_heading(out[k]):
                merged = f'{prev_lead}{prev_core} {core}{trail}'
                if clean_len(merged) <= max_chars:
                    dropped = len(prev_trail) + len(lead) + sum(
                        len(_split_sml_edges(r)[0]) + len(_split_sml_edges(r)[2]) for r in out[k + 1:i]
                    )
                    out[k:i + 1] = [merged]
                    print(f'get_sentences() min-chars floor: merged short row backward, {dropped} join pause token(s) dropped: {_strip_escaped_sml(core)!r}')
                    i = k
                    continue
        print(f'get_sentences() min-chars floor: NO merge fits under {max_chars} chars, short row kept: {_strip_escaped_sml(core)!r}')
        i += 1
    return out


def get_sentences(text:str, session_id:str, sml_blocks:list[str])->list|None:
    # sml_blocks is escape_sml's block table for THIS text, and it comes in
    # because an escaped token is otherwise an anonymous char here: the merge
    # passes below have to be able to ask which token a char stands for, to keep
    # a section heading out of every merge (2026-08-27, _heading_row_test).

    def split_inclusive(text:str, pattern:re.Pattern[str])->list[str]:
        result = []
        last_end = 0
        for match in pattern.finditer(text):
            result.append(text[last_end:match.end()].strip())
            last_end = match.end()
        if last_end < len(text):
            tail = text[last_end:].strip()
            if tail:
                result.append(tail)
        return result

    def split_sentence_on_sml(sentence:str)->list[str]:
        parts:list[str] = []
        last = 0
        for m in SML_TAG_PATTERN.finditer(sentence):
            start, end = m.span()
            if start > last:
                text = sentence[last:start]
                if text:
                    parts.append(text)
            parts.append(m.group(0))
            last = end
        if last < len(sentence):
            tail = sentence[last:]
            if tail:
                parts.append(tail)
        return parts

    strip_escaped_sml = _strip_escaped_sml

    def clean_len(s:str)->int:
        # Length of what the ENGINE will actually read: SML tokens don't count,
        # and for Orpheus the row is measured through tts_form (book-exact ->
        # model transform; '1923' is 4 chars on screen, 37 spoken as 'one
        # thousand nine hundred twenty three'). Every packing/split decision in
        # the passes below uses this, so the 350-char cap bounds the MODEL's
        # text, not the display text.
        return len(strip_escaped_sml(tts_form(s)))

    def has_words(s:str)->bool:
        # Does the ENGINE get anything to say from this row? Same reading as
        # clean_len — SML stripped, Orpheus's transform applied — so the answer
        # is about the model's text and not the display text.
        return _has_word_chars(tts_form(s))

    def _plain(s:str)->bool:
        # A row is "plain prose" only when it carries NO escaped SML token: the
        # escaped tag chars have ord >= sml_escape_tag, so strip_escaped_sml
        # shortens the string exactly when a token is present. Used by PASS 2 and
        # PASS 4 to refuse a merge that would BURY a [break]/[pause]/… mid-row,
        # where orpheus.py strips it before TTS and its pause is silently lost.
        # (PASS 5 no longer uses it: that pack merges at the EDGES with
        # _split_sml_edges and drops the join's tokens explicitly, counted.)
        # NOTE: compares strip_escaped_sml directly, NOT clean_len — clean_len is
        # transform-aware for Orpheus and would misread any digit-bearing row as
        # "carries SML".
        return len(strip_escaped_sml(s)) == len(s)

    def is_latin_only(s:str)->bool:
        s = strip_escaped_sml(s)
        s = re.sub(r'[^\w\s]', '', s, flags=re.UNICODE)
        has_latin = bool(re.search(r'[A-Za-z]', s))
        has_nonlatin = bool(re.search(r'[^\x00-\x7F]', s))
        return has_latin and not has_nonlatin

    def split_at_space_limit(s:str)->list[str]:
        out = []
        rest = s.strip()
        while rest and len(strip_escaped_sml(rest)) > max_chars:
            cut = rest[:max_chars + 1]
            idx = cut.rfind(' ')
            if idx == -1:
                out.append(rest[:max_chars].strip())
                rest = rest[max_chars:].strip()
            else:
                out.append(rest[:idx].strip())
                rest = rest[idx + 1:].strip()
        if rest:
            out.append(rest.strip())
        return out

    def segment_ideogramms(text:str)->list[str]:
        result = []
        try:
            if lang in ['yue','yue-Hant','yue-Hans','zh-yue','cantonese']:
                import pycantonese as pc
                result.extend([t for t in pc.segment(text) if t.strip()])
            elif lang == 'zho':
                import jieba
                jieba.dt.cache_file = os.path.join(models_dir, 'jieba.cache')
                result.extend([t for t in jieba.cut(text) if t.strip()])
            elif lang == 'jpn':
                import nagisa
                result.extend(nagisa.tagging(text).words)
            elif lang == 'kor':
                from soynlp.tokenizer import LTokenizer
                ltokenizer = LTokenizer()
                result.extend([t for t in ltokenizer.tokenize(text) if t.strip()])
            elif lang in ['tha','lao','mya','khm']:
                from pythainlp.tokenize import word_tokenize
                result.extend([t for t in word_tokenize(text, engine='newmm') if t.strip()])
            else:
                result.append(text.strip())
            return result
        except Exception as e:
            DependencyError(e)
            return [text]

    def join_ideogramms(idg_list:list[str])->str:
        try:
            buffer = ''
            prev_latin = False
            prev_nonlatin = False
            for token in idg_list:
                cur_starts_latin = bool(re.match(r'[A-Za-z0-9]', token))
                cur_starts_nonlatin = bool(re.match(r'[^\x00-\x7F]', token))
                if buffer:
                    if (prev_latin and (cur_starts_latin or cur_starts_nonlatin)) or (prev_nonlatin and cur_starts_latin):
                        buffer += ' '
                    elif len(buffer) + len(token) > max_chars:
                        yield buffer
                        buffer = ''
                buffer += token
                prev_latin = bool(re.search(r'[A-Za-z0-9]$', token))
                prev_nonlatin = bool(re.search(r'[^\x00-\x7F]$', token))
            if buffer:
                yield buffer
        except Exception as e:
            DependencyError(e)
            if buffer:
                yield buffer

    try:
        session = context.get_session(session_id)
        if not session:
            return None
        lang, tts_engine = session['language'], session['tts_engine']
        # Base chunk size splits at soft punctuation only if needed, keeping full
        # sentences together for better prosody. Voxtral (long-form) doubles it
        # (~500 chars); Orpheus packs a few sentences per generation (350 chars).
        # Orpheus is capped well below Voxtral because fine-tuned voices (trained on
        # short clips) emit end-of-audio EARLY well past ~300 chars / ~20s, silently
        # dropping trailing text — a clean EOS under MAX_AUDIO_TOKENS, so the
        # token-cap re-split can't see it (measured p90 speaking-rate blowup to
        # ~21 ch/s at 400+ chars). orpheus.py's duration-vs-text guard is the
        # backstop. Both then greedily pack adjacent sentences below (PASS 5).
        max_chars = language_mapping[lang]['max_chars']
        if tts_engine == 'voxtral':
            max_chars *= 2
        elif tts_engine == 'orpheus':
            # Orpheus packs to a cap of 350 chars (raised from 200 on 2026-07-12 for
            # prosody). The 200-char era was calibrated against rohan-v2, later PROVEN
            # to be a broken TRAINING recipe (38s clips / max_seq_length 4096 → 19%
            # runaway); EOS-safe voices (≤20s/2048 — deathstalker_v3, owen, thirdreich)
            # went 0/126 on the very chunks that broke it, and 300-char packing passed
            # whisper-diff completeness on every ≤20s model. 450 fails everywhere —
            # keep the cap ≤~350; the duration-vs-text + token-cap guards in orpheus.py
            # catch stragglers. ORPHEUS_MAX_CHARS overrides; invalid value raises (NO
            # FALLBACK). The old language_mapping*1.2 formula is retired for Orpheus.
            _mc = os.environ.get('ORPHEUS_MAX_CHARS')
            max_chars = int(_mc) if _mc else 350

        # Floor for the last pass before each return (_apply_min_chars_floor):
        # rows shorter than this get merged into a neighbour rather than becoming
        # their own starved TTS prompt.
        min_chars = _sentence_min_chars()

        # The ONE heading test, shared by every merge pass below (2026-08-27).
        is_heading = _heading_row_test(sml_blocks)

        # Orpheus rows are stored/displayed BOOK-EXACT; the scripture+digit
        # expansion happens at the engine boundary (_clean_sentence_for_tts).
        # tts_form is that same transform, used HERE only so clean_len measures
        # the text the model will read. Deterministic and cached; identity for
        # every other engine (their text is already fully words by this point).
        if tts_engine == 'orpheus':
            _tts_form_cache:dict[str,str] = {}
            def tts_form(s:str)->str:
                r = _tts_form_cache.get(s)
                if r is None:
                    r = orpheus_expand_digits(orpheus_normalize_scripture(s))
                    _tts_form_cache[s] = r
                return r
        else:
            def tts_form(s:str)->str:
                return s

        assert not SML_TAG_PATTERN.search(text)

        # A row ends at whitespace, at end of text — or at a PARAGRAPH BOUNDARY.
        #
        # escape_sml has replaced each <break>/<pause>/… with ONE char of
        # ord >= sml_escape_tag (0xE000), and that char is not \s. So a paragraph
        # that ends 'There is hope.' followed immediately by the next paragraph
        # read as 'hope.<break>So when family…' — one unsplittable row with the
        # token BURIED in the middle, where orpheus.py strips it and its pause is
        # silently lost, and where PASS 5 refuses to pack the row at all
        # (see _plain).
        #
        # Latent until Orpheus stopped being routed through the punctuation-
        # spacing rule above: that rule had been inserting the very space this
        # lookahead now recognises. Measured on Killing America the day it went
        # in — 1357 rows became 1725, because packing stopped happening, and
        # every buried break lost its pause. The book's own spacing is still
        # never touched; this only teaches the splitter that a paragraph break
        # ends a row, which it always did.
        #
        # The range matches _strip_escaped_sml's own test (ord >= sml_escape_tag)
        # rather than assuming the tokens stay inside the Private Use Area.
        tok_class = rf'[{chr(sml_escape_tag)}-\U0010FFFF]'
        row_end = rf'(?=\s|$|{tok_class})'

        # A ROW ALSO ENDS IMMEDIATELY BEFORE A TOKEN, PUNCTUATION OR NOT.
        #
        # row_end lets a row end at a mark that is FOLLOWED by a token. It cannot
        # help text that reaches a paragraph boundary without a mark at all, and
        # two common shapes do exactly that:
        #
        #   HEADINGS. 'Chapter 8: State of Confusion' ends in a letter, so the row
        #   ran on through the heading's own token into the prose and came out as
        #   'Chapter 8: State of Confusion. Some consider Franklin D. Roosevelt…' —
        #   the title welded to the first sentence. True of a real <h2> as much as
        #   of a styled <p>, so it was broken for every book.
        #
        #   THE ABBREVIATION GUARD. The dot after 'D.C.' or 'etc.' is deliberately
        #   not a sentence end (so 'Mr. Darcy' never breaks mid-name), so a
        #   paragraph ending in one swallowed the next paragraph's token. Measured
        #   on Killing America: 69 of the 97 short one-sentence chunks (71%).
        #
        # Both leave the token BURIED mid-row, where PASS 5 refuses to pack and
        # orpheus.py strips it before TTS. Terminating on whichever comes first —
        # a sentence end or a token — is what makes "no token is ever buried" true
        # by construction rather than by luck.
        #
        # ONE lazy scan with two terminators rather than an alternation of two
        # patterns: alternation is ordered, so a token-first branch would swallow
        # whole paragraphs (no sentence splitting inside them) and a marks-first
        # branch would never reach the token. Laziness is what picks the EARLIER
        # of the two. The leading `{tok_class}*` lets a row start on its own
        # token(s) without the terminator matching empty at position zero.

        # A sentence that ends inside quotes ends AT THE CLOSING QUOTE, not at the
        # mark. Without this the lookahead lands on the '"' of '"Are you sure?"',
        # refuses to break, and the row runs on THROUGH the paragraph token that
        # follows — burying it mid-row, where PASS 5 refuses to pack the row at all
        # and it becomes its own one-sentence chunk.
        #
        # This is why DIALOGUE fragments while exposition does not: every turn
        # ending in ?" or !" or ." blocks its own split. Measured on Deathstalker
        # Honor — 6328 of 15067 rows (42%) carried a buried token, 78% of runs were
        # a single row, and chunks filled 48% of the cap against Killing America's
        # 74%. Hard marks only: a soft mark inside quotes ('"No," she said') is
        # mid-sentence and must not end a row.
        closing_run = r'["\'’”»)\]]*'

        # PASS 1 — hard punctuation
        if tts_engine == 'orpheus':
            # Abbreviations stay unexpanded for Orpheus (book-exact text), so
            # the splitter must not treat their dot as a sentence end ('He asked
            # Mr. Darcy…' must not break after 'Mr.' — a break there is a bogus
            # sentence gap at assembly). Guard the dot with lookbehinds for the
            # known English abbreviation stems (the same table normalize_text
            # expands for the other engines) plus any single letter (initials,
            # 'C.I.A.', 'e.g.'). Cost of the guard: a dot that ends BOTH an
            # abbreviation and a real sentence ('…joined the C.I.A. The next
            # year…') no longer splits — two sentences ride one row, which
            # Orpheus reads fine — far lesser evil than a mid-name break.
            stems = set()
            for k in abbreviations_mapping.get('eng', {}):
                stem = (k[:-1] if k.endswith('.') else k).split('.')[-1].strip()
                if len(stem) >= 2:
                    stems.add(stem)
            guards = ''.join(f'(?<!\\b{re.escape(s)})' for s in sorted(stems))
            guarded_dot = rf'(?<!\b[A-Za-z]){guards}\.'
            others = [re.escape(p) for p in punctuation_split_hard_set if p != '.']
            hard_pattern = re.compile(
                rf"{tok_class}*.*?(?:(?:{'|'.join([guarded_dot] + others)}){closing_run}{row_end}|(?={tok_class}))",
                re.DOTALL
            )
        else:
            hard_pattern = re.compile(
                rf"{tok_class}*.*?(?:(?:{'|'.join(map(re.escape, punctuation_split_hard_set))}){closing_run}{row_end}|(?={tok_class}))",
                re.DOTALL
            )
        hard_list = split_inclusive(text, hard_pattern)
        if not hard_list:
            hard_list = [text.strip()]
        hard_list = [s.strip() for s in hard_list if s.strip()]

        # PASS 2 — soft punctuation
        soft_pattern = re.compile(
            rf"(.*?(?:{'|'.join(map(re.escape, punctuation_split_soft_set))})){row_end}",
            re.DOTALL
        )
        soft_list = []
        i = 0
        n = len(hard_list)
        while i < n:
            s = hard_list[i].strip()
            if not s:
                i += 1
                continue
            if i + 1 < n:
                next_s = hard_list[i + 1].strip()
                next_clean = strip_escaped_sml(next_s)
                # _plain BOTH SIDES: this mini-merge used to glue an almost-empty
                # hard row onto its predecessor on length alone, which could bury
                # an SML token mid-row — where it is stripped before TTS and its
                # pause silently lost. Token-carrying fragments now fall through
                # to the min-chars floor pass, which merges at the EDGES.
                if (next_clean and sum(c.isalnum() for c in next_clean) < 3
                        and _plain(s) and _plain(next_s)):
                    s = f"{s} {next_s}"
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            if clean_len(s) <= max_chars:
                soft_list.append(s)
                continue
            parts = split_inclusive(s, soft_pattern)
            if parts:
                valid = False
                for p in parts:
                    if clean_len(p.strip()) <= max_chars:
                        valid = True
                        break
                if valid:
                    soft_list.extend([p.strip() for p in parts if p.strip()])
                else:
                    soft_list.append(s)
            else:
                soft_list.append(s)

        # PASS 3 — space split (last resort)
        last_list = []
        for s in soft_list:
            s = s.strip()
            if not s:
                continue
            rest = s
            while rest:
                current_len = clean_len(rest)
                if current_len <= max_chars:
                    last_list.append(rest.strip())
                    break
                cut = rest[:max_chars + 1]
                idx = cut.rfind(' ')
                if idx > 0:
                    left = rest[:idx].strip()
                    right = rest[idx + 1:].strip()
                else:
                    left = rest[:max_chars].strip()
                    right = rest[max_chars:].strip()
                if not left or right == rest:
                    last_list.append(rest.strip())
                    break
                last_list.append(left)
                rest = right

        # PASS 4 — merge very short rows.
        # BOUNDARY-AWARE (like PASS 5): a row carrying an SML token is NEVER merged
        # in either direction. This pass used to merge purely on length, which
        # produced chunks with a mid-chunk [break] (e.g. "…replaced it with myth.
        # [break]Fiction soon bled…") — orpheus.py strips that token before TTS, so
        # the paragraph pause was SILENTLY discarded. Refuse any merge where either
        # side isn't _plain (no escaped SML); a short token-carrying row stays its
        # own row, which is exactly what preserves the pause.
        final_list = []
        merge_max_chars = int((max_chars / 2) / 3)
        i = 0
        n = len(last_list)
        while i < n:
            cur = last_list[i].strip()
            if not cur:
                i += 1
                continue
            if i == 0:
                final_list.append(cur)
                i += 1
                continue
            cur_len = clean_len(cur)
            if cur_len <= merge_max_chars and _plain(cur):
                j = i + 1
                while j < n:
                    nxt = last_list[j].strip()
                    if not nxt:
                        j += 1
                        continue
                    if _plain(nxt) and cur_len + clean_len(nxt) <= max_chars:
                        cur = cur.rstrip() + ' ' + nxt.lstrip()
                        cur_len = clean_len(cur)
                        j += 1
                        continue
                    break
                if final_list:
                    prev = final_list[-1]
                    if _plain(prev) and clean_len(prev) + cur_len <= max_chars:
                        final_list[-1] = prev.rstrip() + ' ' + cur.lstrip()
                        i = j
                        continue
                final_list.append(cur)
                i = j
                continue
            final_list.append(cur)
            i += 1

        if lang in ['zho', 'jpn', 'kor', 'tha', 'lao', 'mya', 'khm']:
            result = []
            for s in final_list:
                parts = split_sentence_on_sml(s)
                for part in parts:
                    part = part.strip()
                    if not part:
                        continue
                    if SML_TAG_PATTERN.fullmatch(part):
                        result.append(part)
                        continue
                    tokens = segment_ideogramms(part)
                    if isinstance(tokens, list):
                        result.extend([t for t in tokens if t.strip()])
                    else:
                        tokens = tokens.strip()
                        if tokens:
                            result.append(tokens)
            joined = []
            for s in join_ideogramms(result):
                if not is_latin_only(s):
                    joined.append(s)
            # The ideogram path never ran the min-chars floor and still does not,
            # but "never hand the model text with no word in it" is not a floor
            # rule and holds here too. \w is Unicode-aware, so every real CJK
            # token answers yes and only decoration ('。', '※', a rule of dashes)
            # is dropped.
            return _drop_wordless_rows(joined, has_words)
        else:
            if tts_engine == 'voxtral':
                # Voxtral is a long-form model: greedily pack adjacent sentences up to
                # max_chars so each generation spans several sentences. Rendering one
                # short sentence per call makes every sentence an independent "take"
                # with inconsistent timbre/prosody — the packing keeps prosody coherent.
                #
                # A HEADING NEVER SHARES A GENERATION (2026-08-27). This packer
                # was purely length-based and had no SML awareness at all, so a
                # section header was simply the first few characters of whatever
                # chunk it landed in. A marked heading now both starts its own
                # chunk and ends it: the row after a heading opens a fresh chunk
                # rather than being appended to the header's.
                packed = []
                for s in final_list:
                    s = s.strip()
                    if not s:
                        continue
                    if is_heading(s) or (packed and is_heading(packed[-1])):
                        packed.append(s)
                        continue
                    if packed and clean_len(packed[-1]) + 1 + clean_len(s) <= max_chars:
                        packed[-1] = packed[-1].rstrip() + ' ' + s.lstrip()
                    else:
                        packed.append(s)
                return _apply_min_chars_floor(packed, clean_len, max_chars, min_chars, is_heading, has_words)
            if tts_engine == 'orpheus':
                # PASS 5 (Orpheus) — greedily pack adjacent sentences up to max_chars so
                # each generation spans 2-3 sentences: coherent timbre/prosody across a
                # passage instead of a per-sentence "take". (Re-enabled once the
                # sentence-start artifact was traced to a prompt-framing/stray-BOS bug
                # in orpheus.py, not the packing itself.)
                #
                # PACKS ACROSS PARAGRAPH PAUSES (2026-08-10). The pack used to refuse
                # any row carrying an SML token, so every paragraph started a fresh
                # chunk — and since e2a puts a [break] on the LEADING edge of each
                # paragraph's first sentence, that sentence could never join either its
                # predecessor or its own successor and was rendered ALONE. Measured on
                # "Killing America" (540-char deathstalker cap): 53% of chunks were a
                # single sentence and the mean chunk was 200 of 540 available chars,
                # with 83% of the single-sentence chunks carrying a token — one starved
                # "take" per paragraph, which is exactly the prosody break Owen hears.
                #
                # RATIFIED TRADE-OFF (same one _apply_min_chars_floor makes): packing
                # beats the pause. The join's tokens — the accumulating chunk's TRAILING
                # token and the incoming row's LEADING token — are dropped and counted
                # (one summary line per chapter, not one per join). They cannot be
                # carried: orpheus.py's _classify_gap realizes only a row's LEADING and
                # TRAILING token as silence and _clean_sentence_for_tts strips the rest,
                # so a token buried mid-row loses its pause anyway while hiding the loss.
                # The cost is small by construction: for Orpheus _classify_gap collapses
                # the auto [break] and the valueless auto [pause] to the SAME sentence-gap
                # floor every chunk already gets (the paragraph/section tiers were removed
                # 2026-07-17), so a dropped auto token costs no measurable silence.
                # The chunk's own leading edge keeps its leading token and its trailing
                # edge takes the newly absorbed row's trailing token, so the pause INTO
                # and OUT OF the chunk still play and no mid-row token is ever produced.
                # Rows with a token in the MIDDLE, and SML-only rows, are never packed.
                # Sentence boundaries inside a pack are untouched, max_chars is unchanged,
                # and _apply_near_dup_split still runs LAST (anti-runaway outranks packing).
                # (_plain is still used by PASS 4; _split_sml_edges is shared with the
                # min-chars floor pass.)
                # SENTENCE-COUNT CAP — OFF by default (2026-07-12). The cap was added
                # the same day, A/B-calibrated against rohan-v2's per-internal-boundary
                # silence attractor (3 sents: 4/9 runaways, 6: 3/3) — but rohan-v2 was
                # later PROVEN a broken TRAINING recipe (38s/4096). EOS-safe voices
                # (≤20s/2048) went 0/126 on those same multi-sentence chunks, so the
                # char cap alone bounds chunks now. ORPHEUS_MAX_SENTENCES re-imposes a
                # cap for a voice that trips the guards; invalid value raises (NO
                # FALLBACK). When set, it's enforced BOTH ways: the merge won't pack
                # past it, and _split_to_cap splits items that ARRIVE multi-sentence
                # (PASS 4 merges short dialogue fragments upstream — a merge-only cap
                # left 593/2516 chunks at 3-9 sentences). SML tokens stay inside their
                # piece, so no pause is dropped.
                _ms = os.environ.get('ORPHEUS_MAX_SENTENCES')
                max_sents = int(_ms) if _ms else None
                def _nsent(t):
                    return max(1, len(re.findall(r'[.!?…]["\'”’)\]]*(?:\s|$)', t)))
                # BALANCED, not greedy-to-the-brim (Owen, 2026-08-13: "if a
                # paragraph is 900 characters … instead of creating a 500 character
                # chunk and a 100 character chunk, we split it in half and create two
                # 300-character chunks").
                #
                # Greedy filling emits a starved tail: a 600-char run at a 540 cap
                # becomes 540 + 60, and that 60-char chunk is its own generation, with
                # its own take of the voice. Balancing costs NOTHING — the run needs
                # the same number of chunks either way, since k is fixed by the run's
                # total length — so the only question is where the boundaries fall, and
                # even is better than lopsided on both axes that matter: no starved
                # take, and a lower PEAK length, which is the one that flirts with the
                # EOS/token cliff (500 chars already spends ~93% of the 3700-token
                # budget at the measured worst-case rate).
                #
                # Measured on Killing America at deathstalker's real 540 cap: greedy
                # gives 924 chunks with 110 under 150 chars; balanced gives 895 with 12
                # under 150. Chunk COUNT barely moves, which is the point — this buys
                # evenness, not throughput.
                #
                # Boundaries are still only ever SENTENCE boundaries (the rows handed
                # in), so a run whose sentences do not divide evenly stays lopsided and
                # some chunks are a single sentence. That is expected and fine.
                def _group_run(run, limit):
                    # Greedy grouping of one run at `limit`, returning index lists.
                    # clean_len is measured on the merged CORE: lead/trail are SML and
                    # contribute nothing to it, exactly as the old merged check relied on.
                    groups, cur, cur_core = [], [], ''
                    for i, core in enumerate(r[1] for r in run):
                        if not cur:
                            cur, cur_core = [i], core
                            continue
                        merged_core = cur_core.rstrip() + ' ' + core.lstrip()
                        if (clean_len(merged_core) <= limit
                                and (max_sents is None
                                     or _nsent(cur_core) + _nsent(core) <= max_sents)):
                            cur.append(i)
                            cur_core = merged_core
                        else:
                            groups.append(cur)
                            cur, cur_core = [i], core
                    if cur:
                        groups.append(cur)
                    return groups

                def _balanced_groups(run):
                    # k is what greedy needs at the real cap — the floor on chunks for
                    # this run. Then find the SMALLEST limit that still fits in k, which
                    # is the evenest split reachable at sentence boundaries. Binary
                    # search is exact here because feasibility is monotone in the limit.
                    at_cap = _group_run(run, max_chars)
                    if len(at_cap) <= 1:
                        return at_cap
                    k = len(at_cap)
                    lo = max(clean_len(r[1]) for r in run)
                    hi, best = max_chars, max_chars
                    while lo <= hi:
                        mid = (lo + hi) // 2
                        if len(_group_run(run, mid)) <= k:
                            best, hi = mid, mid - 1
                        else:
                            lo = mid + 1
                    return _group_run(run, best)

                # Rows as (original, edges) — edges None for a row that must never be
                # packed into (SML-only, or a token buried mid-row that a merge would
                # have to discard silently). Those break the run they sit in.
                #
                # A HEADING IS THE THIRD KIND (2026-08-27). Breaking the run is
                # exactly the behaviour a header needs and it comes for free: the
                # row is emitted alone, byte for byte, and neither neighbour can
                # reach across it to the other. The classifier could not see this
                # itself — it tests _has_escaped_sml(core), and a heading's marker
                # sits on the row's LEAD, so the packer used to swallow headers
                # happily and drop the marker as a join token.
                items = []
                for s in final_list:
                    s = s.strip()
                    if not s:
                        continue
                    lead, core, trail = _split_sml_edges(s)
                    items.append((s, None if (not core or _has_escaped_sml(core) or is_heading(s))
                                  else (lead, core, trail, s)))

                packed = []
                dropped_join_tokens = 0

                def _emit(run):
                    nonlocal dropped_join_tokens
                    for g in _balanced_groups(run):
                        if len(g) == 1:
                            # Untouched, byte for byte. _split_sml_edges drops the
                            # whitespace around a token, so rebuilding a row that was
                            # never merged would quietly rewrite it.
                            packed.append(run[g[0]][3])
                            continue
                        lead = run[g[0]][0]
                        trail = run[g[-1]][2]
                        core = run[g[0]][1]
                        for i in g[1:]:
                            core = core.rstrip() + ' ' + run[i][1].lstrip()
                        # Every join discards the accumulating chunk's TRAILING token
                        # and the incoming row's LEADING token, counted as before.
                        for i in g[:-1]:
                            dropped_join_tokens += len(run[i][2])
                        for i in g[1:]:
                            dropped_join_tokens += len(run[i][0])
                        packed.append(f'{lead}{core}{trail}')

                run = []
                for s, e in items:
                    if e is None:
                        if run:
                            _emit(run)
                            run = []
                        packed.append(s)
                    else:
                        run.append(e)
                if run:
                    _emit(run)
                if dropped_join_tokens:
                    print(f'get_sentences() Orpheus pack: {dropped_join_tokens} join pause token(s) dropped '
                          f'packing {len(final_list)} rows into {len(packed)} chunks (packing > pause)')
                if max_sents is None:
                    # MIN-CHARS FLOOR runs BEFORE PASS 6: anti-runaway trumps the
                    # floor, so if a near-duplicate re-split breaks a floored merge
                    # back apart, near-dup wins.
                    # PASS 6 — repetition-primed split (anti-runaway). See
                    # _apply_near_dup_split: keeps a near-duplicate sentence pair
                    # out of one generation. No-op for non-repetitive prose.
                    return _apply_near_dup_split(
                        _apply_min_chars_floor(packed, clean_len, max_chars, min_chars, is_heading, has_words)
                    )
                def _split_to_cap(t):
                    parts, last = [], 0
                    for m in re.finditer(r'[.!?…]["\'”’)\]]*\s+', t):
                        parts.append(t[last:m.end()].strip())
                        last = m.end()
                    if last < len(t):
                        parts.append(t[last:].strip())
                    parts = [p for p in parts if p]
                    out = []
                    for p in parts:
                        if (out and _nsent(out[-1]) + _nsent(p) <= max_sents
                                and clean_len(out[-1]) + 1 + clean_len(p) <= max_chars):
                            out[-1] = out[-1] + ' ' + p
                        else:
                            out.append(p)
                    return out or [t]
                capped = []
                for item in packed:
                    if _nsent(item) > max_sents:
                        capped.extend(_split_to_cap(item))
                    else:
                        capped.append(item)
                # MIN-CHARS FLOOR then PASS 6 — repetition-primed split
                # (anti-runaway); see above.
                return _apply_near_dup_split(
                    _apply_min_chars_floor(capped, clean_len, max_chars, min_chars, is_heading, has_words)
                )
            return _apply_min_chars_floor(final_list, clean_len, max_chars, min_chars, is_heading, has_words)
    except Exception as e:
        print(f'get_sentences() error: {e}')
        return None

def get_sanitized(str:str, replacement:str='_')->str:
    str = str.replace('&', 'And')
    forbidden_chars = r'[<>:"/\\|?*\x00-\x1F ()]'
    sanitized = re.sub(r'\s+', replacement, str)
    sanitized = re.sub(forbidden_chars, replacement, sanitized)
    sanitized = sanitized.strip('_')
    return sanitized
    
def get_date_entities(text:str, stanza_nlp:Pipeline)->list[tuple[int,int,str]]|None:
    # Returns a list of DATE spans ([] when none found) or None when the
    # stanza NER pipeline itself crashed — callers must treat None as an error,
    # not as "no dates".
    try:
        doc = stanza_nlp(text)
        date_spans = []
        for ent in doc.ents:
            if ent.type == 'DATE':
                date_spans.append((ent.start_char, ent.end_char, ent.text))
        return date_spans
    except Exception as e:
        error = f'get_date_entities() error: {e}'
        print(error)
        return None

def get_num2words_compat(lang_iso1:str)->bool:
    try:
        test = num2words(1, lang=lang_iso1.replace('zh', 'zh_CN'))
        return True
    except NotImplementedError:
        return False
    except Exception as e:
        return False

def set_formatted_number(text:str, lang:str, lang_iso1:str, is_num2words_compat:bool, max_single_value:int=999_999_999_999_999_999)->str:
    # match up to 18 digits, optional “,…” groups (allowing spaces or NBSP after comma), optional decimal of up to 12 digits
    # handle optional range with dash/en dash/em dash between numbers, and allow trailing punctuation
    number_re = re.compile(
        r'(?<!\w)'
        r'(\d{1,18}(?:,\s*\d{1,18})*(?:\.\d{1,12})?)'      # first number
        r'(?:\s*([-–—])\s*'                                # dash type
        r'(\d{1,18}(?:,\s*\d{1,18})*(?:\.\d{1,12})?))?'    # optional second number
        r'([^\w\s]*)',                                     # optional trailing punctuation
        re.UNICODE
    )

    def normalize_commas(num_str:str)->str:
        # ormalize number string to standard comma format: 1,234,567
        tok = num_str.replace('\u00A0', '').replace(' ', '')
        if '.' in tok:
            integer_part, decimal_part = tok.split('.', 1)
            integer_part = integer_part.replace(',', '')
            integer_part = "{:,}".format(int(integer_part))
            return f'{integer_part}.{decimal_part}'
        else:
            integer_part = tok.replace(',', '')
            return "{:,}".format(int(integer_part))

    def clean_single_num(num_str:str)->str:
        tok = unicodedata.normalize('NFKC', num_str)
        if tok.lower() in ('inf', 'infinity', 'nan'):
            return tok
        clean = tok.replace(',', '').replace('\u00A0', '').replace(' ', '')
        try:
            num = float(clean) if '.' in clean else int(clean)
        except (ValueError, OverflowError):
            return tok
        if not math.isfinite(num) or abs(num) > max_single_value:
            return tok

        # Normalize commas before final output
        tok = normalize_commas(tok)

        if is_num2words_compat:
            new_lang_iso1 = lang_iso1.replace('zh', 'zh_CN')
            return num2words(num, lang=new_lang_iso1)
        else:
            phoneme_map = language_math_phonemes.get(
                lang,
                language_math_phonemes.get(default_language_code, language_math_phonemes['eng'])
            )
            return ' '.join(phoneme_map.get(ch, ch) for ch in str(num))

    def clean_match(match:re.Match)->str:
        first_num = clean_single_num(match.group(1))
        dash_char = match.group(2) or ''
        second_num = clean_single_num(match.group(3)) if match.group(3) else ''
        trailing = match.group(4) or ''
        if second_num:
            return f'{first_num}{dash_char}{second_num}{trailing}'
        else:
            return f'{first_num}{trailing}'

    return number_re.sub(clean_match, text)

def year2words(year_str:str, lang:str, lang_iso1:str, is_num2words_compat:bool)->str|bool:
    try:
        year = int(year_str)
        first_two = int(year_str[:2])
        last_two = int(year_str[2:])
        lang_iso1 = lang_iso1 if lang in language_math_phonemes.keys() else default_language_code
        lang_iso1 = lang_iso1.replace('zh', 'zh_CN')
        # Shapes with no year reading at all: not four digits, or a first pair too
        # low for the pair form to be idiomatic ('1042' is 'ten forty-two' by the
        # rule and 'one thousand forty two' by ear). Those stay cardinal.
        # A year ending 00-09 keeps the OLD cardinal in every language but English,
        # because the readings below ('oh', 'hundred') are English words and a
        # German or French year has its own conventions. Anglicizing them quietly
        # would be worse than the reading this replaces.
        english = lang_iso1.startswith('en')
        if (not year_str.isdigit() or len(year_str) != 4 or first_two < 11
                or (last_two < 10 and not english)):
            if is_num2words_compat:
                return num2words(year, lang=lang_iso1)
            else:
                return ' '.join(language_math_phonemes[lang].get(ch, ch) for ch in year_str)
        if is_num2words_compat:
            # A year ending 00-09 has its own readings, and the old code fell back
            # to the plain cardinal for all of them — which is where every
            # remaining wrong year came from (measured 2026-08-13):
            #   1900 -> 'one thousand, nine hundred'          now 'nineteen hundred'
            #   1905 -> 'one thousand, nine hundred and five' now 'nineteen oh five'
            #   1607 -> 'one thousand, six hundred and seven' now 'sixteen oh seven'
            #   2006 -> 'two thousand and six'                now 'two thousand six'
            # The 2000s are the one century not read as a pair, which is why they
            # are tested before the 'oh' form: 'twenty oh six' is nobody's reading.
            if last_two < 10:
                if first_two == 20:
                    tail = f' {num2words(last_two, lang=lang_iso1)}' if last_two else ''
                    return f'{num2words(first_two * 100, lang=lang_iso1)}{tail}'
                if last_two == 0:
                    return f'{num2words(first_two, lang=lang_iso1)} hundred'
                return (f'{num2words(first_two, lang=lang_iso1)} oh '
                        f'{num2words(last_two, lang=lang_iso1)}')
            return f'{num2words(first_two, lang=lang_iso1)} {num2words(last_two, lang=lang_iso1)}'
        else:
            return ' '.join(language_math_phonemes[lang].get(ch, ch) for ch in first_two) + ' ' + ' '.join(language_math_phonemes[lang].get(ch, ch) for ch in last_two)
    except Exception as e:
        error = f'year2words() error: {e}'
        print(error)
        # MUST return a str, not False: this is a re.sub replacement callable, and
        # returning a bool raises TypeError inside re.sub → the exception is swallowed
        # by filter_chapter's blanket except → the chapter becomes None → get_chapters
        # aborts the rest of the book. Leaving the original token in place is harmless.
        return year_str

def clock2words(text:str, lang:str, lang_iso1:str, tts_engine:str, is_num2words_compat:bool)->str:

    def n2w(n:int)->str:
        key = (n, lang, is_num2words_compat)
        if key in _n2w_cache:
            return _n2w_cache[key]
        if is_num2words_compat:
            word = num2words(n, lang=lang_iso1)
        else:
            word = math2words(n, lang, lang_iso1, tts_engine, is_num2words_compat)
        if not isinstance(word, str):
            word = str(word)
        _n2w_cache[key] = word
        return word

    def repl_num(m:re.Match)->str:
        # Reject enumeration patterns like "(1.2)"
        start, end = m.start(), m.end()
        if (
            start > 0 and end < len(text)
            and text[start - 1] == '('
            and text[end] == ')'
        ):
            return m.group(0)
        # Parse hh[:mm[:ss]]
        try:
            h = int(m.group(1))
            mnt = int(m.group(2))
            sec = m.group(3)
            sec = int(sec) if sec is not None else None
        except Exception:
            return m.group(0)
        # basic validation; if out of range, keep original
        if not (0 <= h <= 23 and 0 <= mnt <= 59 and (sec is None or 0 <= sec <= 59)):
            return m.group(0)
        # If no language clock rules, just say numbers plainly
        if not lc:
            parts = [n2w(h)]
            if mnt != 0:
                parts.append(n2w(mnt))
            if sec is not None and sec > 0:
                parts.append(n2w(sec))
            return ' '.join(parts)
        next_hour = (h + 1) % 24
        special_hours = lc.get('special_hours', {})
        if mnt == 0 and (sec is None or sec == 0):
            if h in special_hours:
                phrase = special_hours[h]
            else:
                phrase = lc['oclock'].format(hour=n2w(h))
        elif mnt == 15:
            phrase = lc['quarter_past'].format(hour=n2w(h))
        elif mnt == 30:
            if lang == 'deu':
                phrase = lc['half_past'].format(next_hour=n2w(next_hour))
            else:
                phrase = lc['half_past'].format(hour=n2w(h))
        elif mnt == 45:
            phrase = lc['quarter_to'].format(next_hour=n2w(next_hour))
        elif mnt < 30:
            phrase = lc['past'].format(hour=n2w(h), minute=n2w(mnt)) if mnt != 0 else lc['oclock'].format(hour=n2w(h))
        else:
            minute_to_hour = 60 - mnt
            phrase = lc['to'].format(
                next_hour=n2w(next_hour),
                minute=n2w(minute_to_hour),
                minute_to_hour=n2w(minute_to_hour)
            )
        if sec is not None and sec > 0:
            second_phrase = lc['second'].format(second=n2w(sec))
            phrase = lc['full'].format(phrase=phrase, second_phrase=second_phrase)
        return phrase

    time_rx = re.compile(
        r'\b([01]?\d|2[0-3]):([0-5]\d)(?::([0-5]\d))?\b'
    )
    lc = language_clock.get(lang) if 'language_clock' in globals() else None
    _n2w_cache = {}
    return time_rx.sub(repl_num, text)

def math2words(text:str, lang:str, lang_iso1:str, tts_engine:str, is_num2words_compat:bool)->str:
    def repl_ambiguous(match:re.Match)->str:
        # handles "num SYMBOL num" and "SYMBOL num"
        if match.group(2) and match.group(2) in ambiguous_replacements:
            return f'{match.group(1)} {ambiguous_replacements[match.group(2)]} {match.group(3)}'
        if match.group(3) and match.group(3) in ambiguous_replacements:
            return f'{ambiguous_replacements[match.group(3)]} {match.group(4)}'
        return match.group(0)

    def _ordinal_to_words(m:re.Match)->str:
        n = int(m.group(1))
        if is_num2words_compat:
            try:
                from num2words import num2words
                return num2words(n, to='ordinal', lang=(lang_iso1 or 'en'))
            except Exception:
                pass
        # If num2words isn't available/compatible, keep original token as-is.
        return m.group(0)

    # Matches any digits + optional space/NBSP + st/nd/rd/th, not glued into words.
    re_ordinal = re.compile(r'(?<!\w)(\d+)(?:\s|\u00A0)*(?:st|nd|rd|th)(?!\w)')
    text = re.sub(r'(\d)\)', r'\1 : ', text)
    text = re_ordinal.sub(_ordinal_to_words, text)
    # Symbol phonemes
    ambiguous_symbols = {"-", "/", "*", "x"}
    phonemes_list = language_math_phonemes.get(lang, language_math_phonemes[default_language_code])
    replacements = {k: v for k, v in phonemes_list.items() if not k.isdigit() and k not in [',', '.']}
    normal_replacements  = {k: v for k, v in replacements.items() if k not in ambiguous_symbols}
    ambiguous_replacements = {k: v for k, v in replacements.items() if k in ambiguous_symbols}
    # Replace unambiguous symbols everywhere
    if normal_replacements:
        sym_pat = r'(' + '|'.join(map(re.escape, normal_replacements.keys())) + r')'
        text = re.sub(sym_pat, lambda m: f' {normal_replacements[m.group(1)]} ', text)
    # Replace ambiguous symbols only in valid equation contexts
    if ambiguous_replacements:
        ambiguous_pattern = (
            r'(?<!\S)'                   # no non-space before
            r'(\d+)\s*([-/*x])\s*(\d+)'  # num SYMBOL num
            r'(?!\S)'                    # no non-space after
            r'|'                         # or
            r'(?<!\S)([-/*x])\s*(\d+)(?!\S)'  # SYMBOL num
        )
        text = re.sub(ambiguous_pattern, repl_ambiguous, text)
    text = set_formatted_number(text, lang, lang_iso1, is_num2words_compat)
    return text

def roman2number(text: str)->str:

    def is_valid_roman(s: str)->bool:
        return bool(valid_roman.fullmatch(s))

    def to_int(s: str)->str:
        s = s.upper()
        i = 0
        result = 0
        while i < len(s):
            for roman, value in roman_numbers_tuples:
                if s[i:i + len(roman)] == roman:
                    result += value
                    i += len(roman)
                    break
            else:
                return s
        return str(result)

    def repl_heading(m: re.Match)->str:
        roman = m.group(1)
        if not is_valid_roman(roman):
            return m.group(0)
        return f"{to_int(roman)}{m.group(2)}{m.group(3)}"

    def repl_standalone(m: re.Match)->str:
        roman = m.group(1)
        if not is_valid_roman(roman):
            return m.group(0)
        return f"{to_int(roman)}{m.group(2)}"

    def repl_word(m: re.Match)->str:
        roman = m.group(1)
        if not is_valid_roman(roman):
            return m.group(0)
        return to_int(roman)

    def repl_chapter_single(m: re.Match)->str:
        word = m.group(1)
        roman = m.group(2)
        if not is_valid_roman(roman):
            return m.group(0)
        return f"{word} {to_int(roman)}"

    valid_roman = re.compile(
        r'^(?=.)M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$',
        re.IGNORECASE
    )
    chapter_words = sorted(
        {w for words in chapter_word_mapping.values() for w in words},
        key=len,
        reverse=True
    )
    chapter_words_re = re.compile(
        rf'\b({"|".join(map(re.escape, chapter_words))})\s+([IVXLCDM]+)\b',
        re.IGNORECASE | re.UNICODE
    )
    text = re.sub(
        r'^(?:\s*)([IVXLCDM]+)([.-])(\s+)',
        repl_heading,
        text,
        flags=re.MULTILINE
    )
    text = re.sub(
        r'^(?:\s*)([IVXLCDM]+)([.-])(?:\s*)$',
        repl_standalone,
        text,
        flags=re.MULTILINE
    )
    text = chapter_words_re.sub(repl_chapter_single, text)
    text = re.sub(
        r'(?<!\S)([IVXLCDM]{2,})(?!\S)',
        repl_word,
        text
    )
    return text
    
def is_latin(s:str)->bool:
    return all((u'a' <= ch.lower() <= 'z') or ch.isdigit() or not ch.isalpha() for ch in s)

def foreign2latin(text:str, base_lang:str)->str:

    def script_of(word:str)->str:
        for ch in word:
            if ch.isalpha():
                name = unicodedata.name(ch, '')
                if 'CYRILLIC' in name:
                    return 'cyrillic'
                if 'LATIN' in name:
                    return 'latin'
                if 'ARABIC' in name:
                    return 'arabic'
                if 'HANGUL' in name:
                    return 'hangul'
                if 'HIRAGANA' in name or 'KATAKANA' in name:
                    return 'japanese'
                if 'CJK' in name or 'IDEOGRAPH' in name:
                    return 'chinese'
        return 'unknown'

    def romanize(word:str)->str:
        scr = script_of(word)
        if scr == 'latin':
            return word
        try:
            if scr == 'chinese':
                from pypinyin import pinyin, Style
                return ''.join(x[0] for x in pinyin(word, style=Style.NORMAL))
            if scr == 'japanese':
                import pykakasi
                k = pykakasi.kakasi()
                k.setMode('H', 'a')
                k.setMode('K', 'a')
                k.setMode('J', 'a')
                k.setMode('r', 'Hepburn')
                return k.getConverter().do(word)
            if scr == 'hangul':
                return unidecode(word)
            if scr == 'arabic':
                return unidecode(phonemize(word, language='ar', backend='espeak'))
            if scr == 'cyrillic':
                return unidecode(phonemize(word, language='ru', backend='espeak'))
            return unidecode(word)
        except Exception:
            return unidecode(word)

    # Protect ALL SML tags using the global grammar
    protected: Dict[str, str] = {}
    for i, m in enumerate(SML_TAG_PATTERN.finditer(text)):
        key: str = f'__TTS_MARKER_{i}__'
        protected[key] = m.group(0)
        text = text.replace(m.group(0), key)
    # Tokenise INCLUDING whitespace, so the original spacing is REBUILT, not guessed.
    #
    # The previous pattern (r"\w+|[^\w\s]") dropped every space, and the rejoin below
    # re-inserted one ONLY between two adjacent word tokens — so any space touching a
    # punctuation mark was destroyed outright:
    #     an "overtly-satanic ... LGBTQIA+ supplier," for so-called "pride month" in
    #  -> an"overtly-satanic ... LGBTQIA+supplier,"for so-called"pride month"in
    # Measured 2026-07-28 on Killing America: this is a no-op semantically for
    # Latin-script books (romanize() returns Latin words unchanged) yet it mangled the
    # spacing of every one of them, and the densest victim — 437 chars carrying four
    # jammed quotes — was the ONE chunk in 1013 that hit the audio-token cap.
    #
    # Whitespace and punctuation now pass through verbatim; only word tokens are
    # romanized. `protected` is still checked first so SML markers are never touched.
    tokens: list[str] = re.findall(r"\w+|\s+|[^\w\s]", text, re.UNICODE)
    buf: list[str] = []
    for t in tokens:
        if t in protected:
            buf.append(t)
        elif re.match(r"^\w+$", t):
            buf.append(romanize(t))
        else:
            buf.append(t)
    out: str = ''.join(buf)
    for k, v in protected.items():
        out = out.replace(k, v)
    return out

def normalize_sml_tags(text:str)->tuple[bool, str]:
    out = []
    stack = []
    last = 0
    for m in SML_TAG_PATTERN.finditer(text):
        start, end = m.span()
        out.append(text[last:start])
        tag = m.group("tag")
        close = bool(m.group("close"))
        value = m.group("value")
        info = TTS_SML.get(tag)
        if not info:
            out.append(m.group(0))
            last = end
            continue
        if info.get("paired"):
            if close:
                if not stack or stack[-1] != tag:
                    error = f'normalize_sml_tags() error: unmatched closing tag [/{tag}]'
                    return False, error
                stack.pop()
                out.append(f"[/{tag}]")
            else:
                stack.append(tag)
                if value is not None:
                    out.append(f"[{tag}:{value.strip()}]")
                else:
                    error = f'normalize_sml_tags() error: paired tag [{tag}] requires a value'
                    return False, error
        else:
            if close:
                error = f'normalize_sml_tags() error: non-paired tag [/{tag}] is invalid'
                return False, error
            out.append(info['static'])
        last = end
    out.append(text[last:])
    if stack:
        error = f"normalize_sml_tags() error: unclosed tag(s): {', '.join(stack)}"
        return False, error
    return True, ''.join(out)

def escape_sml(text:str)->tuple[str, list[str]]:
    sml_blocks:list[str] = []

    def replace(m:re.Match[str])->str:
        sml_blocks.append(m.group(0))
        return chr(sml_escape_tag + len(sml_blocks) - 1)

    return SML_TAG_PATTERN.sub(replace, text), sml_blocks

def restore_sml(text:str, sml_blocks:list[str])->str:
    for i, block in enumerate(sml_blocks):
        text = text.replace(chr(sml_escape_tag + i), block)
    return text

def sml_token(tag:str, value:str|None=None, close:bool=False)->str:
    if close:
        return f"[/{tag}]"
    if value is not None:
        return f"[{tag}:{value}]"
    return f"[{tag}]"

def sml_heading(title:str)->str:
    """Mark a row as a section heading/title (2026-08-27).

    The [heading] marker goes on the row's LEADING edge — see TTS_SML['heading']
    for why it is a leading marker and not a paired wrapper. This marker is the
    ONLY thing that carries the heading identity out of filter_chapter's walker
    and down into get_sentences, where _heading_row_test turns it back into
    "never merge this row into anything". Every engine strips it before TTS
    (SML_UNSPOKEN_PATTERN, and _convert_sml for the XTTS-class engines), so it
    is never spoken, and it is stripped again for VTT cues and m4b chapter
    titles, which are built from these same rows.

    The title arrives already terminated — the caller adds the period that makes
    TTS stop — so this only prefixes the marker."""
    return f"{sml_token('heading')}{title}"

def normalize_text(text:str, lang:str, lang_iso1:str, tts_engine:str)->str:

    def replace(match:re.Match)->str:
        token = match.group(1)
        for k, expansion in mapping.items():
            if token.lower() == k.lower():
                return expansion
        return token  # fallback
            
    # Remove emojis
    emoji_pattern = re.compile(f"[{''.join(emojis_list)}]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    # Orpheus fine-tunes are trained on book-exact text: 'Mr.', 'Mrs.', 'St.'
    # and dotted acronyms ('C.I.A.') appear verbatim in the training transcripts,
    # so expanding or de-dotting them at inference moves AWAY from the training
    # distribution. Both transforms stay for the acoustic engines (XTTS/VITS/…).
    if tts_engine != TTS_ENGINES['ORPHEUS']:
        if lang in abbreviations_mapping:
            mapping = abbreviations_mapping[lang]
            # Sort keys by descending length so longer ones match first
            keys = sorted(mapping.keys(), key=len, reverse=True)
            # Build a regex that only matches whole “words” (tokens) exactly
            pattern = re.compile(
                r'(?<!\w)(' + '|'.join(re.escape(k) for k in keys) + r')(?!\w)',
                flags=re.IGNORECASE
            )
            text = pattern.sub(replace, text)
        # This regex matches sequences like a., c.i.a., f.d.a., m.c., etc…
        # uppercase acronyms
        text = re.sub(r'\b(?:[a-zA-Z]\.){1,}[a-zA-Z]?\b\.?', lambda m: m.group().replace('.', '').upper(), text)
    # romanize foreign words
    if language_mapping[lang]['script'] == 'latin':
        text = foreign2latin(text, lang)
    # Replace multiple newlines ("\n\n", "\r\r", "\n\r", etc.) with a [pause] 1.4sec
    pattern = r'(?:\r\n|\r|\n){2,}'
    text = re.sub(pattern, f" {sml_token('pause')} ", text)
    # Replace single newlines ("\n" or "\r") with spaces
    text = re.sub(r'\r\n|\r|\n', ' ', text)
    # Replace punctuations causing hallucinations
    switch = punctuation_switch
    if tts_engine == TTS_ENGINES['ORPHEUS']:
        # Book-exact overrides. The shared table rewrites em/en dashes to '.'
        # (forcing sentence breaks mid-clause) and parens to commas — both wrong
        # for Orpheus, whose training text folds dashes to hyphens (the corpus
        # extractor's _SMART table: '—' -> ' - ', '–' -> '-') and keeps parens
        # verbatim (parentheticals are prosody the LLM reads natively).
        switch = dict(punctuation_switch)
        switch['–'] = '-'
        switch['—'] = ' - '
        del switch['(']
        del switch[')']
    pattern = f"[{''.join(map(re.escape, switch.keys()))}]"
    text = re.sub(pattern, lambda match: switch.get(match.group(), match.group()), text)
    # remove unwanted chars
    chars_remove_table = str.maketrans({ch: ' ' for ch in chars_remove})
    text = text.translate(chars_remove_table)
    # Quote handling. Orpheus is an LLM-based TTS trained on ordinary prose, where
    # quotation marks are the cue that a span is spoken dialogue — stripping them
    # leaves the model guessing which lines are in a character's mouth. The removal
    # below exists for the older acoustic engines (XTTS/VITS/Bark), which hallucinate
    # on quotes; Orpheus keeps them (already folded to straight quotes above).
    if tts_engine != TTS_ENGINES['ORPHEUS']:
        # A quote that FOLLOWS a letter/digit becomes a comma — it is doing the work
        # a comma would ("she whispered never again before" needs the pause). Any
        # other quote is dropped, but its surrounding whitespace MUST survive: the
        # old code collapsed spaces around every quote FIRST, so dropping one fused
        # the words across it ('"Get out!" he shouted' -> 'Get out!he shouted').
        text = re.sub(r'(?<=[\p{L}\p{N}])\s*"\s*(?=[\p{L}\p{N}])', ', ', text)
        text = re.sub(r'\s*"\s*', ' ', text)
    # Replace multiple and spaces with single space
    text = re.sub(r'\s+', ' ', text)
    # Replace ok by 'Okay' — acoustic engines only; Orpheus reads book-exact text
    if tts_engine != TTS_ENGINES['ORPHEUS']:
        text = re.sub(r'\bok\b', 'Okay', text, flags=re.IGNORECASE)
    # Reduce a RUN of consecutive punctuation to its last mark, followed by one space.
    #
    # The trailing space is withheld in two cases where it does real damage:
    #
    #  * before a CLOSING quote or bracket. The mark belongs to the sentence being
    #    closed, so 'Get out!" he said' must not become 'Get out! " he said' — that
    #    orphans the quote and is nowhere in any narrator's training transcripts.
    #  * inside a DOTTED ACRONYM ('D.C.', 'C.I.A.'). The Orpheus branch above
    #    deliberately skips de-dotting so these read book-exact; splitting them to
    #    'D. C.' here silently undid that, and changes how they are spoken.
    #  * INSIDE A NUMBER. A mark flanked on both sides by digits is not punctuation
    #    at all — it is a thousands separator, a decimal point, or the divider in a
    #    clock time or a chapter:verse reference. Spacing it broke every one of
    #    them: '10,000' -> '10, 000', '9.1-magnitude' -> '9. 1-magnitude',
    #    '7:59 a.m.' -> '7: 59 a.m.'. Orpheus feels this hardest because its
    #    book-exact branch is the one that still HAS digits by the time this runs
    #    (the acoustic engines have already had math2words turn them into words),
    #    and because normalize_scripture matches '\d+:\d+' with no space — a
    #    spaced colon put every scripture reference out of its reach as well.
    #    Measured on Killing America, 2026-08-13: 47 split thousands separators,
    #    94 split colons and 3 split decimals in one narration.
    #
    # The first two were latent until 2026-07-28, masked by foreign2latin having
    # already stripped the spaces this rule then re-inserted.
    closing = '"\'’”»)]}'

    def _collapse(m: re.Match) -> str:
        run = m.group(0).strip()
        # ADJACENT marks are the author's punctuation, not junk: '...' is a hesitation,
        # '?!' is an intonation, '!!' is emphasis. Collapsing them to the last mark
        # threw all three away — and '...' -> '.' is the worst, because it converts a
        # mid-clause pause into a SENTENCE BOUNDARY ('He paused. then spoke.').
        # Marks separated by WHITESPACE (', ,' / '. .') are genuine junk and still
        # collapse. The splitter does not need the collapse either way: it requires
        # (?=\s|$) after a mark, so 'He paused... then' already splits after 'paused...'.
        mark = run if (run and not any(c.isspace() for c in run)) else m.group(2)
        src = m.string
        nxt = src[m.end():m.end() + 1]
        prv = src[m.start() - 1:m.start()] if m.start() > 0 else ''
        if nxt == '' or nxt in closing:
            return mark
        # A whitespace-free mark with a digit on each side belongs to the number,
        # not to the sentence. Whitespace anywhere in the run means the author
        # already ended something there ('It cost him. 3 people died'), so that
        # case still collapses normally.
        if run == mark and prv.isdigit() and nxt.isdigit():
            return mark
        if mark == '.' and nxt.isalpha():
            # Inside an acronym the dot is followed by a LONE letter that itself
            # carries a dot ('D.C.', 'C.I.A.'). At the acronym's END the next word is
            # ordinary ('D.C. in'), and that space must survive — checking only the
            # letter BEFORE the dot cannot tell those apart and swallowed it.
            before = src[:m.start()].rstrip()
            lone_before = (len(before) >= 1 and before[-1].isalpha()
                           and (len(before) < 2 or not before[-2].isalpha()))
            if lone_before and re.match(r'[^\W\d_]\.', src[m.end():]):
                return mark
        return mark + ' '

    # ORPHEUS IS NOT ROUTED THROUGH THIS RULE AT ALL.
    #
    # Owen, 2026-08-13: "we should just ignore the splitting logic that adds spaces
    # after punctuation. we should assume the punctuation spacing will be correct
    # when the book goes in… dont delete it, just dont route orpheus text through
    # that pipeline."
    #
    # The rule is XTTS-era: an acoustic model fed 'wars.Heritage' as one token
    # mispronounces it, so the pipeline pried every mark apart and re-spaced it. An
    # LLM TTS reads 'when she looked,she was disturbed' correctly, so there is
    # nothing to buy — and a space the book never wrote is a token sequence the
    # fine-tune never saw. Measured on Killing America (378k chars of book-exact
    # text): of ~1,500 insertions, 645 split a DOMAIN NAME ('Heritage.org' ->
    # 'Heritage. org'), 306 split a URL ('https://www.' -> 'https: //www.'), 104
    # split a scripture reference and 138 split a number. Ordinary prose is
    # untouched either way, because the book already spaces its own commas.
    #
    # SKIPPED WHOLESALE rather than made a no-op inside _collapse: the pattern
    # swallows the whitespace AROUND a mark as well as the mark, so a version that
    # merely withheld the trailing space would still delete the space the book
    # wrote before one ('Ibid. , "RULE' -> 'Ibid.,"RULE'). Not running it is the
    # only way "the book's spacing is what the model sees" is actually true.
    #
    # The splitter does not need it: it splits on a mark followed by whitespace,
    # and every sentence boundary in real prose already has that whitespace. What
    # stops splitting is a URL, which should never have been split.
    if tts_engine != TTS_ENGINES['ORPHEUS']:
        pattern = '|'.join(map(re.escape, punctuation_split_hard_set))
        text = re.sub(rf'(\s*({pattern})\s*)+', _collapse, text).strip()
        pattern = '|'.join(map(re.escape, punctuation_split_soft_set))
        text = re.sub(rf'(\s*({pattern})\s*)+', _collapse, text).strip()
    # Pattern 1: Add a space between UTF-8 characters and numbers.
    # NOT for Orpheus: its kept-digit forms ('1930s', '7th', '76ers') appear
    # exactly like that in the training transcripts — spacing them ('1930 s')
    # would mangle the very tokens the book-exact branch preserves.
    if tts_engine != TTS_ENGINES['ORPHEUS']:
        text = re.sub(r'(?<=[\p{L}])(?=\d)|(?<=\d)(?=[\p{L}])', ' ', text)
    # Replace special chars with words — acoustic engines only. Orpheus training
    # text keeps the raw glyphs ('40%', 'Q&A', '@'), so the fine-tunes read them
    # natively; substituting words here would leave the transcript saying
    # 'forty percent' where the book says '40%'.
    if tts_engine != TTS_ENGINES['ORPHEUS']:
        specialchars = specialchars_mapping.get(lang, specialchars_mapping.get(default_language_code, specialchars_mapping['eng']))
        specialchars_table = {ord(char): f" {word} " for char, word in specialchars.items()}
        text = text.translate(specialchars_table)
    text = ' '.join(text.split())
    return text

def convert_chapters2audio(session_id:str)->bool:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        try:
            if session['cancellation_requested']:
                msg = 'Cancel requested'
                print(msg)
                return False
            tts_manager = TTSManager(session)
            resume_chapter = 0
            missing_chapters = []
            chapter_re = re.compile(r'^(\d+)\.' + re.escape(default_audio_proc_format) + r'$')
            existing_chapters = [f for f in os.listdir(session['chapters_dir']) if chapter_re.match(f)]
            existing_numbers = sorted(int(chapter_re.match(f).group(1)) for f in existing_chapters)
            if existing_numbers:
                expected = set(range(0, max(existing_numbers) + 1))
                missing_chapters = sorted(expected - set(existing_numbers))
                if not missing_chapters:
                    missing_chapters = [max(existing_numbers) + 1]
                resume_chapter = existing_numbers[-1]
            resume_sentence = 0
            missing_sentences = []
            sentence_re = re.compile(r'^(\d+)\.' + re.escape(default_audio_proc_format) + r'$')
            existing_sentences = [f for f in os.listdir(session['sentences_dir']) if sentence_re.match(f)]
            existing_numbers = sorted(int(sentence_re.match(f).group(1)) for f in existing_sentences)
            if existing_numbers:
                expected = set(range(0, max(existing_numbers) + 1))
                missing_sentences = sorted(expected - set(existing_numbers))
                if not missing_sentences:
                    missing_sentences = [max(existing_numbers) + 1]
                resume_sentence = existing_numbers[-1]
            total_chapters = len(session['chapters'])
            if total_chapters == 0:
                error = 'No chapterrs found!'
                print(error)
                return False
            total_iterations = sum(len(session['chapters'][x]) for x in range(total_chapters))
            total_sentences = sum(sum(1 for row in chapter) for chapter in session['chapters'])
            if total_sentences == 0:
                error = 'No sentences found!'
                print(error)
                return False
            msg = f"--------------------------------------------------\nA total of {total_chapters} {'block' if total_chapters <= 1 else 'blocks'} and {total_sentences} {'sentence' if total_sentences <= 1 else 'sentences'}.\n--------------------------------------------------"
            print(msg)
            if session['is_gui_process']:
                progress_bar = gr.Progress(track_tqdm=False)
            final_sentences = []
            ebook_name = Path(session['ebook']).name if session.get('ebook') else 'audio'

            # Worker mode: get range limits
            worker_mode = session.get('worker_mode', False)
            worker_start = session.get('sentence_start', 0) if worker_mode else 0
            worker_end = session.get('sentence_end', float('inf')) if worker_mode else float('inf')

            # For worker mode, tqdm shows progress within the worker's assigned range only
            if worker_mode:
                worker_range_size = worker_end - worker_start + 1
                tqdm_total = worker_range_size
            else:
                tqdm_total = total_iterations

            with tqdm(total=tqdm_total, desc='0.00%', bar_format='{desc}: {n_fmt}/{total_fmt} ', unit='step', initial=0) as t:
                idx_target = 0
                for c in range(0, total_chapters):
                    chapter_idx = c
                    chapter_audio_file = f'{chapter_idx}.{default_audio_proc_format}'
                    sentences = session['chapters'][c]
                    start = idx_target
                    # Worker mode: check if this chapter overlaps with our range
                    chapter_end_idx = start + len(sentences) - 1
                    chapter_in_worker_range = not worker_mode or (chapter_end_idx >= worker_start and start <= worker_end)

                    if chapter_in_worker_range:
                        if c in missing_chapters:
                            msg = f'********* Recovering missing block {c} *********'
                            print(msg)
                        elif resume_chapter == c and c > 0:
                            msg = f'********* Resuming from block {resume_chapter} *********'
                            print(msg)
                        msg = f'Block {chapter_idx} containing {len(sentences)} sentences…'
                        print(msg)
                    for idx, sentence in enumerate(sentences):
                        if session['cancellation_requested']:
                            msg = 'Cancel requested'
                            print(msg)
                            return False
                        sentence = sentence.strip()
                        if any(c.isalnum() for c in sentence):
                            is_sml = bool(SML_TAG_PATTERN.fullmatch(sentence))
                            if (not is_sml) or (idx == len(sentences) - 1):
                                final_sentences.append(sentence)

                            # Worker mode: efficiently skip sentences outside assigned range
                            in_worker_range = worker_start <= idx_target <= worker_end

                            # Worker mode: fast-forward through sentences before our range
                            if worker_mode and idx_target < worker_start:
                                idx_target += 1
                                # Don't update tqdm - it's sized for our range only
                                continue  # Skip processing, printing, etc.

                            # Worker mode: stop after our range ends
                            if worker_mode and idx_target > worker_end:
                                return True  # Done with our range

                            if in_worker_range and (idx_target in missing_sentences or idx_target >= resume_sentence):
                                if idx_target in missing_sentences:
                                    msg = f'********* Recovering missing sentence {idx_target} *********'
                                    print(msg)
                                elif resume_sentence == idx_target and resume_sentence > 0:
                                    msg = f'********* Resuming from sentence {resume_sentence} ********'
                                    print(msg)
                                success = tts_manager.convert_sentence2audio(idx_target, sentence) if sentence else True
                                if not success:
                                    return False
                            idx_target += 1
                        total_progress = (t.n + 1) / tqdm_total
                        if session['is_gui_process']:
                            progress_bar(progress=total_progress, desc=f'{ebook_name} - {sentence}')
                        percent = total_progress * 100
                        t.set_description(f"{percent:.2f}%")
                        msg = f' : {sentence}'
                        print(msg)
                        t.update(1)
                    end = idx_target - 1
                    if chapter_in_worker_range:
                        msg = f'End of Block {chapter_idx}'
                        print(msg)
                    # Worker mode: skip chapter combining - assembly phase handles this
                    if not worker_mode:
                        if chapter_idx in missing_chapters or idx_target >= resume_sentence:
                            if combine_audio_sentences(session_id, chapter_audio_file, int(start), int(end)):
                                msg = f'Combining block {chapter_idx} to audio, sentence {start} to {end}'
                                print(msg)
                            else:
                                msg = 'combine_audio_sentences() failed!'
                                print(msg)
                                return False
            # Worker mode: skip VTT creation - assembly phase handles this
            if session.get('worker_mode', False):
                return True
            return tts_manager.create_sentences2vtt(final_sentences)
        except Exception as e:
            DependencyError(e)
            error = f'convert_chapters2audio() error: {e}'
            print(error)
            return False

def _read_flac_streaminfo_block(filepath:str)->bytes:
    # The mandatory STREAMINFO metadata block (stdlib only): 'fLaC' magic, then the
    # FIRST metadata block must be STREAMINFO (type 0) per the FLAC spec.
    with open(filepath, 'rb') as f:
        magic = f.read(4)
        if magic != b'fLaC':
            raise ValueError(f'Not a FLAC file (bad magic): {filepath}')
        header = f.read(4)
        if len(header) < 4 or (header[0] & 0x7F) != 0:
            raise ValueError(f'First FLAC metadata block is not STREAMINFO: {filepath}')
        info = f.read(34)
        if len(info) < 34:
            raise ValueError(f'Truncated FLAC STREAMINFO block: {filepath}')
    return info

def read_flac_streaminfo(filepath:str)->tuple[int, int]:
    # Returns (max_blocksize, samplerate).
    info = _read_flac_streaminfo_block(filepath)
    max_blocksize = int.from_bytes(info[2:4], 'big')
    samplerate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
    return max_blocksize, samplerate

def read_flac_duration(filepath:str)->float:
    """
    Exact duration from the FLAC STREAMINFO header — total_samples / samplerate.

    This replaces one ffprobe subprocess (or one full pydub decode) PER FILE. On a
    20-hour book that was 2740 process spawns for the VTT and a complete PCM decode
    of every chapter for the chapter markers; here it is a 42-byte read each.

    STREAMINFO packs, after the block/frame sizes: 20 bits samplerate, 3 bits
    channels, 5 bits bits-per-sample, 36 bits total_samples — bytes 10..17.

    Raises rather than returning 0.0 for an unreadable or unsized stream: a zero
    does not look like an error to any caller, it looks like silence, and every
    consumer (VTT cue timing, chapter marker offsets, the assembly duration guard)
    then draws a confidently wrong conclusion from it.
    """
    info = _read_flac_streaminfo_block(filepath)
    samplerate = (info[10] << 12) | (info[11] << 4) | (info[12] >> 4)
    total_samples = (
        ((info[13] & 0x0F) << 32) | (info[14] << 24) |
        (info[15] << 16) | (info[16] << 8) | info[17]
    )
    if samplerate == 0:
        raise ValueError(f'FLAC STREAMINFO declares samplerate 0: {filepath}')
    if total_samples == 0:
        # 0 means "unknown" in the spec (a stream written without a final rewrite).
        # e2a always writes complete files, so this means the file is damaged.
        raise ValueError(
            f'FLAC STREAMINFO declares total_samples 0 (unknown length) — the file is '
            f'incomplete or was never finalized: {filepath}'
        )
    return total_samples / samplerate

def audio_duration_seconds(filepath:str)->float:
    """Duration of a processed-audio file. Uses the FLAC header when possible."""
    if str(filepath).lower().endswith('.flac'):
        return read_flac_duration(filepath)
    return get_audio_duration(filepath)

def parallel_export_unsupported_reason(session:dict, source_duration:float)->str|None:
    """
    Why the per-chapter parallel encode may NOT stand in for the single serial encode,
    or None when it may. Every condition is about producing a BYTE-EQUIVALENT-IN-INTENT
    result, not about convenience — if any holds we take the serial path.

      - MP4-family output only. Each chunk is a self-contained .m4a whose edit list
        records its encoder delay, which is why concatenating them is gapless
        (measured: 0.179 ms drift across 77 joins on a 20.07 h book). MP3 needs LAME
        gapless tags and Opus/Vorbis carry pre-skip per stream; neither survives a
        naive concat, so they keep the serial encode.
      - No pre-loudnorm filters. FINAL_DENOISE / post_render_filter are applied
        per-stream; running them per chunk would restart the filter's internal state
        78 times instead of once.
      - Not split into parts. The split branch of combine_audio_chapters() merges each
        part's chapters into one intermediate FLAC and hands that to export_audio();
        it never reaches the parallel encoder at all.
      - Longer than the 2 h loudnorm cutoff. Below it the serial path applies
        loudnorm=…:linear=true, which MEASURES THE WHOLE FILE and cannot be computed
        per chunk without changing the result. Above it the serial path already skips
        loudnorm, so the two paths agree — and a sub-2 h book encodes serially in a
        few minutes anyway.

    This lives at module scope, not inside combine_audio_chapters(), because
    assemble_audiobook() has to ask the SAME question BEFORE it decides whether a
    chapter's sentences still need concatenating into a chapter FLAC. Two copies of
    this policy that drifted apart would leave assembly with neither a chapter FLAC
    nor a usable pre-encoded chapter, so there is exactly one copy.
    """
    out_fmt = session['output_format']
    if out_fmt not in ('m4b', 'm4a', 'mp4', 'mov'):
        return f'output format {out_fmt} is not MP4-family (a naive concat is not gapless there)'
    if os.environ.get('FINAL_DENOISE', '0') == '1':
        return 'FINAL_DENOISE is active and must run once over the whole stream'
    if session.get('post_render_filter'):
        return 'a post_render_filter is active and must run once over the whole stream'
    if session.get('output_split'):
        return 'the output is split into parts, which merges chapters before encoding'
    if not source_duration:
        return 'the total duration is unknown'
    if source_duration <= 7200:
        return (
            f'the book is {source_duration:.0f}s, at or under the 7200s cutoff below which '
            f'loudnorm(linear=true) measures the whole file and cannot be split'
        )
    return None

def load_encoded_chapters(session:dict, expected_chapter_nums:list[int])->dict[int, dict]:
    """
    The chapters BookForge already encoded to AAC while the GPU was still rendering,
    handed over as <N>.m4a in session['encoded_chapters_dir'] (N = e2a's 1-indexed
    chapter number, the same number as <N>.flac in chapters_dir). The set may be
    partial or empty; a chapter that is present is one BookForge has already checked
    against its own staleness stamp, so assembly's job is to USE it, not re-derive it.

    Returns {chapter_num: {'path': str, 'duration': float}}, empty when the flag is
    not in play. Idempotent and side-effect free — both assemble_audiobook() (to
    decide which sentence concats to skip) and combine_audio_chapters() (to place
    them in the final concat) call it, and they must see the identical set.

    Every failure here ABORTS rather than quietly rebuilding the chapter from its
    sentences. BookForge vouched for these files; a silent rebuild would repair the
    symptom of a corrupt hand-off and destroy the evidence of the bug that caused it,
    while shipping an audiobook whose chapters came from somewhere the caller did not
    intend.
    """
    encoded_dir = session.get('encoded_chapters_dir')
    if not encoded_dir:
        return {}
    if not os.path.isdir(encoded_dir):
        raise RuntimeError(
            f'load_encoded_chapters(): --encoded_chapters_dir is not a directory: {encoded_dir}'
        )
    expected = set(expected_chapter_nums)
    entries = {}
    for name in sorted(os.listdir(encoded_dir)):
        if not name.lower().endswith('.m4a'):
            continue
        stem = name[:-len('.m4a')]
        path = os.path.join(encoded_dir, name)
        if not stem.isdigit():
            raise RuntimeError(
                f'load_encoded_chapters(): {path} is not named <chapter_number>.m4a. The file '
                f'name IS the chapter mapping — a name we cannot resolve to a chapter number '
                f'would place its audio at an arbitrary point in the book.'
            )
        num = int(stem)
        if num not in expected:
            raise RuntimeError(
                f'load_encoded_chapters(): {path} claims chapter {num}, which is not one of the '
                f'{len(expected)} chapters this assembly is building '
                f'({min(expected) if expected else "-"}..{max(expected) if expected else "-"}). '
                f'Refusing to assemble: the directory and the session disagree about what the '
                f'book is.'
            )
        if os.path.getsize(path) == 0:
            raise RuntimeError(
                f'load_encoded_chapters(): pre-encoded chapter {num} is 0 bytes: {path}'
            )
        entries[num] = {'path': path, 'duration': None}
    if not entries:
        return {}
    # One batched mediainfo call for the whole set: get_audiolist_duration() raises
    # (naming the file) if any of them is unreadable, which is exactly the outcome we
    # want for a chapter we are about to copy verbatim into the audiobook.
    paths = [entries[n]['path'] for n in sorted(entries)]
    durations = get_audiolist_duration(paths)
    for num in sorted(entries):
        duration = durations[os.path.realpath(entries[num]['path'])]
        if duration <= 0:
            raise RuntimeError(
                f'load_encoded_chapters(): pre-encoded chapter {num} reports a duration of '
                f'{duration}s: {entries[num]["path"]}'
            )
        entries[num]['duration'] = duration
    return entries

def combine_audio_sentences(session_id:str, file:str, start:int, end:int)->bool:
    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            chapter_audio_file = os.path.join(session['chapters_dir'], file)
            start = int(start)
            end = int(end)
            base = session['sentences_dir']
            ext = f".{default_audio_proc_format}"
            start_i = int(start)
            end_i = int(end)
            exists = os.path.exists
            join = os.path.join
            missing = []
            selected_files = []
            for i in range(start_i, end_i + 1):
                path = join(base, f"{i}{ext}")
                if exists(path):
                    selected_files.append(path)
                else:
                    missing.append(i)
            if missing:
                error = f"Missing sentence files: {missing}"
                print(error)
                return False
            if not selected_files:
                error = 'No audio files found in the specified range.'
                print(error)
                return False
            # ffmpeg's concat demuxer silently drops every FLAC frame whose
            # blocksize exceeds the FIRST list entry's STREAMINFO max-blocksize
            # (and still exits 0), so a mixed-encoder sentence set must never
            # reach it. Refuse non-homogeneous max-blocksize or samplerate.
            if ext == '.flac':
                blocksizes = {}
                samplerates = {}
                for path in selected_files:
                    max_blocksize, samplerate = read_flac_streaminfo(path)
                    blocksizes.setdefault(max_blocksize, []).append(path)
                    samplerates.setdefault(samplerate, []).append(path)
                if len(blocksizes) > 1:
                    error = 'combine_audio_sentences() FLAC max-blocksize is not homogeneous — ffmpeg concat would silently drop frames:'
                    print(error)
                    for max_blocksize in sorted(blocksizes):
                        paths = blocksizes[max_blocksize]
                        msg = f'  max-blocksize {max_blocksize}: {len(paths)} files (e.g. {os.path.basename(paths[0])})'
                        print(msg)
                    return False
                if len(samplerates) > 1:
                    error = 'combine_audio_sentences() FLAC samplerate is not homogeneous — ffmpeg concat would corrupt timing:'
                    print(error)
                    for samplerate in sorted(samplerates):
                        paths = samplerates[samplerate]
                        msg = f'  samplerate {samplerate}: {len(paths)} files (e.g. {os.path.basename(paths[0])})'
                        print(msg)
                    return False
            concat_dir = session['process_dir']
            concat_list = os.path.join(concat_dir, 'concat_list_sentences.txt')
            with open(concat_list, 'w') as f:
                for path in selected_files:
                    if session['cancellation_requested']:
                        msg = 'Cancel requested'
                        print(msg)
                        return False
                    f.write(f"file '{path.replace(os.sep, '/')}'\n")
            result = assemble_audio_chunks(concat_list, chapter_audio_file, session.get('is_gui_process'))
            if not result:
                error = 'combine_audio_sentences() FFmpeg concat failed.'
                print(error)
                return False
            msg = f'********* Combined block audio file saved in {chapter_audio_file}'
            print(msg)
            return True
    except Exception as e:
        DependencyError(e)
    return False

# Opt-in final-pass denoise (FINAL_DENOISE=1), applied in export_audio()'s final
# encode. Tuned for the faint ~-65 dBFS room-hiss bed that hiss-bed-trained voice
# models (e.g. Orpheus) reproduce; tn=1 tracks the actual noise so it also smooths
# the hiss/dead-silence alternation across sentence gaps. Values may be adjusted
# after listening tests.
final_denoise_filter = 'afftdn=nr=12:nf=-50:tn=1'

def combine_audio_chapters(session_id:str)->list[str]|None:

    def chapter_source_path(filename:str)->str:
        # Where chapter <filename> actually lives. Normally chapters_dir/<N>.flac,
        # the FLAC this assembly just concatenated from sentences — but a chapter
        # BookForge pre-encoded has no FLAC at all (assemble_audiobook() skipped its
        # concat on purpose) and is served from encoded_chapters_dir/<N>.m4a instead.
        # Everything downstream — the completeness check, the durations behind the
        # chapter markers and the export guard, the concat list — must resolve the
        # chapter through here or it will look for a file that was never built.
        chapter_num = int(re.search(r'\d+', filename).group())
        entry = encoded_chapters.get(chapter_num)
        if entry:
            return entry['path']
        return os.path.join(session['chapters_dir'], filename)

    def generate_ffmpeg_metadata(part_chapters:list[tuple[str,str]], output_metadata_path:str, default_audio_proc_format:str)->str|bool:
        try:
            out_fmt = session['output_format']
            is_mp4_like = out_fmt in ['mp4', 'm4a', 'm4b', 'mov']
            is_vorbis = out_fmt in ['ogg', 'webm']
            is_mp3 = out_fmt == 'mp3'
            def tag(key):
                return key.upper() if is_vorbis else key
            ffmpeg_metadata = ';FFMETADATA1\n'
            if session['metadata'].get('title'):
                ffmpeg_metadata += f"{tag('title')}={session['metadata']['title']}\n"
            if session['metadata'].get('creator'):
                ffmpeg_metadata += f"{tag('artist')}={session['metadata']['creator']}\n"
            if session['metadata'].get('language'):
                ffmpeg_metadata += f"{tag('language')}={session['metadata']['language']}\n"
            if session['metadata'].get('description'):
                ffmpeg_metadata += f"{tag('description')}={session['metadata']['description']}\n"
            if session['metadata'].get('publisher') and (is_mp4_like or is_mp3):
                ffmpeg_metadata += f"{tag('publisher')}={session['metadata']['publisher']}\n"
            # Only stamp a year we can actually derive. Defaulting an unknown or
            # unparseable publish date to the CURRENT year silently brands the book
            # with a wrong publication year; leave the tag off instead.
            year = None
            if session['metadata'].get('published'):
                try:
                    if '.' in session['metadata']['published']:
                        year = datetime.strptime(session['metadata']['published'], '%Y-%m-%dT%H:%M:%S.%f%z').year
                    else:
                        year = datetime.strptime(session['metadata']['published'], '%Y-%m-%dT%H:%M:%S%z').year
                except Exception:
                    year = None
            if year is not None:
                if is_vorbis:
                    ffmpeg_metadata += f"{tag('date')}={year}\n"
                else:
                    ffmpeg_metadata += f"{tag('year')}={year}\n"
            if session['metadata'].get('identifiers') and isinstance(session['metadata']['identifiers'], dict):
                if is_mp3 or is_mp4_like:
                    isbn = session['metadata']['identifiers'].get('isbn')
                    if isbn:
                        ffmpeg_metadata += f"{tag('isbn')}={isbn}\n"
                    asin = session['metadata']['identifiers'].get('mobi-asin')
                    if asin:
                        ffmpeg_metadata += f"{tag('asin')}={asin}\n"
            start_time = 0
            for filename, chapter_title in part_chapters:
                if session['cancellation_requested']:
                    msg = 'Cancel requested'
                    print(msg)
                    return False
                filepath = chapter_source_path(filename)
                # Read the length from the container header. AudioSegment.from_file()
                # used to decode the ENTIRE chapter to PCM in memory just to call len()
                # on it — 27s and multiple GB of churn on a 20-hour book, to learn a
                # number that is sitting in the first 42 bytes of the file.
                # For a pre-encoded chapter this reads the .m4a, which is the right
                # source anyway: it is the exact stream that lands in the audiobook,
                # so the marker tiles against what the listener will actually hear.
                duration_ms = int(round(audio_duration_seconds(filepath) * 1000))
                clean_title = re.sub(r'(^#)|[=\\]|(-$)', lambda m: '\\' + (m.group(1) or m.group(0)), sanitize_meta_chapter_title(chapter_title))
                ffmpeg_metadata += '[CHAPTER]\nTIMEBASE=1/1000\n'
                ffmpeg_metadata += f'START={start_time}\nEND={start_time + duration_ms}\n'
                ffmpeg_metadata += f"{tag('title')}={clean_title}\n"
                start_time += duration_ms
            with open(output_metadata_path, 'w', encoding='utf-8') as f:
                f.write(ffmpeg_metadata)
            return output_metadata_path
        except Exception as e:
            error = f'generate_ffmpeg_metadata() Error: Failed to process {txt_file} → {out_file}: {e}'
            print(error)
            return False

    def export_audio(ffmpeg_combined_audio:str, ffmpeg_metadata_file:str, ffmpeg_final_file:str)->bool:
        try:
            if session['cancellation_requested']:
                msg = 'Cancel requested'
                print(msg)
                return False
            cover_path = None
            final_denoise = os.environ.get('FINAL_DENOISE', '0') == '1'
            if final_denoise:
                msg = f'[assembly] FINAL_DENOISE active: {final_denoise_filter}'
                print(msg)
            # BookForge per-voice corrective filter (notch/EQ/expander for the artifacts
            # a fine-tune reproduces), from --post_render_filter. Applied in the SAME
            # single encode as loudnorm — never a separate re-encode pass.
            post_render_filter = session.get('post_render_filter')
            if post_render_filter:
                msg = f'[assembly] post_render_filter active: {post_render_filter}'
                print(msg)
            # Filters that must run BEFORE loudnorm (so the target loudness holds after
            # the correction): denoise the raw hiss bed first, then the per-voice
            # corrective chain. Any non-empty entry also forces a real encode (no
            # stream-copy). All of these stream frame-by-frame, so they are safe even on
            # the long-audiobook path where loudnorm (linear=true) is skipped.
            pre_filters = []
            if final_denoise:
                pre_filters.append(final_denoise_filter)
            if post_render_filter:
                pre_filters.append(post_render_filter)
            ffprobe_cmd = [
                shutil.which('ffprobe'), '-v', 'error', '-threads', '0', '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name,sample_rate,sample_fmt',
                '-of', 'default=nokey=1:noprint_wrappers=1', ffmpeg_combined_audio
            ]
            probe = subprocess.run(ffprobe_cmd, capture_output=True, text=True)
            codec_info = probe.stdout.strip().splitlines()
            input_codec = codec_info[0] if len(codec_info) > 0 else None
            input_rate = codec_info[1] if len(codec_info) > 1 else None
            cmd = [shutil.which('ffmpeg'), '-hide_banner', '-nostats', '-hwaccel', 'auto', '-thread_queue_size', '1024', '-i', ffmpeg_combined_audio]
            target_codec, target_rate = None, None
            if session['output_format'] == 'wav':
                target_codec = 'pcm_s16le'
                target_rate = '44100'
                cmd += ['-map', '0:a', '-ar', target_rate, '-sample_fmt', 's16']
            elif session['output_format'] == 'aac':
                target_codec = 'aac'
                target_rate = '44100'
                cmd += ['-c:a', 'aac', '-b:a', '192k', '-ar', target_rate, '-movflags', '+faststart']
            elif session['output_format'] == 'flac':
                target_codec = 'flac'
                target_rate = '44100'
                cmd += ['-c:a', 'flac', '-compression_level', '5', '-ar', target_rate]
            else:
                cmd += ['-f', 'ffmetadata', '-i', ffmpeg_metadata_file, '-map', '0:a']
                if session['output_format'] in ['m4a', 'm4b', 'mp4', 'mov']:
                    target_codec = 'aac'
                    target_rate = '44100'
                    cmd += ['-c:a', 'aac', '-b:a', '192k', '-ar', target_rate, '-movflags', '+faststart+use_metadata_tags']
                elif session['output_format'] == 'mp3':
                    target_codec = 'mp3'
                    target_rate = '44100'
                    cmd += ['-c:a', 'libmp3lame', '-b:a', '192k', '-ar', target_rate]
                elif session['output_format'] == 'webm':
                    target_codec = 'opus'
                    target_rate = '48000'
                    cmd += ['-c:a', 'libopus', '-b:a', '192k', '-ar', target_rate]
                elif session['output_format'] == 'ogg':
                    target_codec = 'opus'
                    target_rate = '48000'
                    cmd += ['-c:a', 'libopus', '-compression_level', '0', '-b:a', '192k', '-ar', target_rate]
                cmd += ['-map_metadata', '1'] 
            if session['output_channel'] == 'stereo':
                cmd += ['-ac', '2']
            else:
                cmd += ['-ac', '1']
            # Any pre-loudnorm filter (FINAL_DENOISE or a per-voice post_render_filter)
            # needs a real filter pass, so it must skip the stream-copy shortcut.
            if input_codec == target_codec and input_rate == target_rate and not pre_filters:
                cmd = [
                    shutil.which('ffmpeg'), '-hide_banner', '-nostats', '-hwaccel', 'auto', '-thread_queue_size', '1024', '-i', ffmpeg_combined_audio,
                    '-threads', '0', '-f', 'ffmetadata', '-i', ffmpeg_metadata_file,
                    '-map', '0:a', '-map_metadata', '1', '-c', 'copy',
                    '-progress', 'pipe:2',
                    '-y', ffmpeg_final_file
                ]
            else:
                # Check duration - skip heavy filters for long audiobooks (>2 hours)
                # loudnorm with linear=true requires analyzing entire file in memory
                audio_duration = get_audio_duration(ffmpeg_combined_audio)
                if audio_duration and audio_duration > 7200:  # 2 hours in seconds
                    print(f'Skipping loudnorm filter for long audiobook ({audio_duration/3600:.1f} hours) to avoid memory issues')
                    if pre_filters:
                        # These stream (unlike loudnorm linear=true) so they are safe on long audiobooks
                        cmd += ['-af', ','.join(pre_filters)]
                    cmd += [
                        '-threads', '0',
                        '-progress', 'pipe:2',
                        '-y', ffmpeg_final_file
                    ]
                else:
                    audio_filters = 'loudnorm=I=-16:LRA=11:TP=-1.5:linear=true,afftdn=nf=-70'
                    if pre_filters:
                        # Denoise/correct the raw signal BEFORE loudnorm measures and normalizes it
                        audio_filters = f"{','.join(pre_filters)},{audio_filters}"
                    cmd += [
                        '-filter_threads', '0',
                        '-filter_complex_threads', '0',
                        '-af', audio_filters,
                        '-threads', '0',
                        '-progress', 'pipe:2',
                        '-y', ffmpeg_final_file
                    ]
            source_duration = get_audio_duration(ffmpeg_combined_audio)
            proc_pipe = SubprocessPipe(cmd, is_gui_process=session['is_gui_process'], total_duration=source_duration, msg='Export')
            if proc_pipe:
                return finalize_export(ffmpeg_final_file, source_duration, 'export_audio')
        except Exception as e:
            error = f'Export failed: {e}'
            print(error)
            return False

    def finalize_export(ffmpeg_final_file:str, source_duration:float, caller:str)->bool:
        # ffmpeg can stop mid-encode and still FINALIZE a valid, playable,
        # truncated file (moov written, exit clean) — e.g. when it loses its
        # progress-pipe reader. exit-0 + file-exists proves nothing about
        # completeness, so hold the export to the same standard as the
        # chapter concat: the output must carry the whole input's duration.
        # (Nuremberg 2026-08-11: a 20.12h source exported as a valid 14.72h
        # m4b, silently — this guard is that incident's fix.)
        final_duration = get_audio_duration(ffmpeg_final_file)
        if source_duration and abs(final_duration - source_duration) > 2.0:
            error = (
                f'{caller}() Output duration mismatch → {ffmpeg_final_file}: '
                f'source is {source_duration:.2f}s, exported file is {final_duration:.2f}s. '
                f'ffmpeg finalized a truncated export; refusing to ship it.'
            )
            print(error)
            return False
        if os.path.exists(ffmpeg_final_file) and os.path.getsize(ffmpeg_final_file) > 0:
            if session['output_format'] in ['mp3', 'm4a', 'm4b', 'mp4']:
                if session['cover'] is not None:
                    cover_path = session['cover']
                    # open() accepts an integer file descriptor, and bool is an int
                    # subclass — so a stray True here opens fd 1 (stdout) instead of
                    # raising, and the read() below deadlocks the process against its
                    # own pipe. Refuse anything that is not a path, loudly.
                    if not isinstance(cover_path, str):
                        raise TypeError(
                            f"session['cover'] must be a path string or None, got "
                            f"{type(cover_path).__name__}: {cover_path!r}"
                        )
                    msg = f'Adding cover {cover_path} into the final audiobook file…'
                    print(msg)
                    if session['output_format'] == 'mp3':
                        from mutagen.mp3 import MP3
                        from mutagen.id3 import ID3, APIC, error
                        audio = MP3(ffmpeg_final_file, ID3=ID3)
                        try:
                            audio.add_tags()
                        except error:
                            pass
                        with open(cover_path, 'rb') as img:
                            audio.tags.add(APIC(encoding=3, mime='image/jpeg', type=3, desc='Cover', data=img.read()))
                    elif session['output_format'] in ['mp4', 'm4a', 'm4b']:
                        from mutagen.mp4 import MP4, MP4Cover
                        audio = MP4(ffmpeg_final_file)
                        with open(cover_path, 'rb') as f:
                            cover_data = f.read()
                        audio['covr'] = [MP4Cover(cover_data, imageformat=MP4Cover.FORMAT_JPEG)]
                    if audio:
                        audio.save()
            final_vtt = f"{Path(ffmpeg_final_file).stem}.vtt"
            proc_vtt_path = os.path.join(session['process_dir'], final_vtt)
            final_vtt_path = os.path.join(session['audiobooks_dir'], final_vtt)
            shutil.move(proc_vtt_path, final_vtt_path)
            return True
        error = f"{Path(ffmpeg_final_file).name} is corrupted or does not exist"
        print(error)
        return False

    def parallel_export_supported(source_duration:float)->bool:
        # The policy itself lives at module scope (see the docstring there) so that
        # assemble_audiobook() can ask the identical question before it decides to
        # skip a chapter's sentence concat. One copy, two callers.
        return parallel_export_unsupported_reason(session, source_duration) is None

    def export_audio_parallel(chapter_paths:list[str], already_encoded:set[int], ffmpeg_metadata_file:str, ffmpeg_final_file:str, source_duration:float)->bool:
        """
        Encode each chapter to AAC concurrently, then concatenate with -c copy.

        ffmpeg's native AAC encoder is single-threaded, so the serial path pinned one
        core and left the rest of the machine idle: 2216 s (37 min) for a 20.07 h book
        on a 20-core host, 82% of total assembly time. Encoding chapters concurrently
        and stream-copying them together did the same book in 257 s + 95 s — measured
        at 280x realtime against 32.8x, with the joined timeline landing 0.6 ms from
        the sum of the chapter durations.

        The chapter split is free: assembly has ALREADY produced one FLAC per chapter,
        and they are exactly the units the serial path was about to concatenate.

        `already_encoded` holds the INDEXES into chapter_paths whose entry is not a
        FLAC to encode but an .m4a BookForge encoded itself during generation, with
        the same settings this function uses. Those go straight into the concat list:
        the whole point is that their encode happened while the GPU was still busy
        rendering the book, so it costs assembly nothing at all. They are the only
        entries whose path is outside chunk_dir, which is why the cleanup below drops
        chunk_dir rather than the individual chunks — deleting by path would delete
        BookForge's originals.
        """
        cpu_count = os.cpu_count()
        if cpu_count is None:
            # Sizing the pool is not a guess we get to make silently — a wrong worker
            # count either starves the machine or oversubscribes it.
            raise RuntimeError('export_audio_parallel(): os.cpu_count() returned None; cannot size the encoder pool')
        workers = max(1, min(cpu_count, 16))
        chunk_dir = os.path.join(session['process_dir'], 'parallel_encode')
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        os.makedirs(chunk_dir, exist_ok=True)

        target_rate = '44100'
        channels = '2' if session['output_channel'] == 'stereo' else '1'
        ffmpeg_bin = shutil.which('ffmpeg')

        def encode_chunk(item:tuple[int, str])->tuple[int, str, str|None]:
            idx, src_path = item
            out_path = os.path.join(chunk_dir, f'{idx:05d}.m4a')
            cmd = [
                ffmpeg_bin, '-hide_banner', '-nostats', '-v', 'error',
                '-i', src_path,
                '-c:a', 'aac', '-b:a', '192k', '-ar', target_rate, '-ac', channels,
                '-y', out_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                return idx, out_path, (proc.stderr or '').strip()[:400]
            if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
                return idx, out_path, 'encoder exited 0 but produced no output'
            return idx, out_path, None

        # A pre-encoded chapter enters the concat list as itself; only the rest are
        # handed to the encoder pool.
        chunk_paths = [None] * len(chapter_paths)
        for idx in sorted(already_encoded):
            chunk_paths[idx] = chapter_paths[idx]
        items = [(i, p) for i, p in enumerate(chapter_paths) if i not in already_encoded]
        print(
            f'[assembly] Parallel encode: {len(items)} chapters across {workers} workers'
            f'{f"; {len(already_encoded)} already encoded by BookForge" if already_encoded else ""}'
        )
        completed = 0
        failures = []
        # BookForge's progress parser reads the "Export - N%" lines below. With every
        # chapter pre-encoded there is no encode to report on at all, so say so once
        # rather than leaving the bar parked at whatever it last showed.
        if not items:
            print('Export - 100.0%', flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(encode_chunk, it) for it in items]
            for fut in concurrent.futures.as_completed(futures):
                idx, out_path, err = fut.result()
                if err is not None:
                    failures.append(f'chapter index {idx} ({os.path.basename(chapter_paths[idx])}): {err}')
                else:
                    chunk_paths[idx] = out_path
                completed += 1
                print(f'Export - {completed / len(items) * 100:.1f}%', flush=True)
                if session['cancellation_requested']:
                    for f in futures:
                        f.cancel()
                    print('Cancel requested')
                    return False

        if failures:
            print(f'export_audio_parallel() {len(failures)} chapter encode(s) failed:')
            for f in failures[:10]:
                print(f'  {f}')
            return False
        missing = [i for i, p in enumerate(chunk_paths) if p is None]
        if missing:
            print(f'export_audio_parallel() no encoded chunk for chapter index(es) {missing}')
            return False

        concat_list = os.path.join(session['process_dir'], 'concat_list_encoded.txt')
        with open(concat_list, 'w', encoding='utf-8') as f:
            for p in chunk_paths:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")

        cmd = [
            ffmpeg_bin, '-hide_banner', '-nostats', '-v', 'error',
            '-f', 'concat', '-safe', '0', '-i', concat_list,
            '-f', 'ffmetadata', '-i', ffmpeg_metadata_file,
            '-map', '0:a', '-map_metadata', '1', '-c:a', 'copy',
            '-movflags', '+faststart+use_metadata_tags',
            '-threads', '0',
            '-y', ffmpeg_final_file,
        ]
        print('[assembly] Concatenating encoded chapters (stream copy)')
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f'export_audio_parallel() concat failed: {(proc.stderr or "").strip()[:600]}')
            return False

        ok = finalize_export(ffmpeg_final_file, source_duration, 'export_audio_parallel')
        # Only drop the chunks once the result has passed the duration guard — if it
        # failed, they are the evidence for why.
        if ok:
            shutil.rmtree(chunk_dir, ignore_errors=True)
        return ok

    try:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            chapter_files = [f for f in os.listdir(session['chapters_dir']) if f.endswith(f'.{default_audio_proc_format}')]
            chapter_files = sorted(chapter_files, key=lambda x: int(re.search(r'\d+', x).group()))

            # Filter by selected chapters if partial assembly
            selected_chapters = session.get('selected_chapters')
            if selected_chapters:
                chapter_files = [f for f in chapter_files if int(re.search(r'\d+', f).group()) in selected_chapters]
                print(f'[ASSEMBLE] Filtered to {len(chapter_files)} selected chapters')

            chapters_data = session.get('chapters', [])
            # The chapter set this assembly believes it is building:
            #   - session['selected_chapters'] when set (assemble_audiobook always sets
            #     it — 1-indexed; includes the designed partial-assembly path, where the
            #     selection itself is the already-narrowed chapter list);
            #   - otherwise one 0-indexed file per chapter in session['chapters']
            #     (legacy finalize_audiobook/convert_chapters2audio flow).
            if selected_chapters:
                expected_chapter_nums = list(selected_chapters)
            else:
                expected_chapter_nums = list(range(len(chapters_data)))

            # Chapters BookForge already encoded to AAC while the GPU was still
            # rendering (--encoded_chapters_dir). assemble_audiobook() skipped their
            # sentence→chapter concat, so there is no FLAC for them in chapters_dir;
            # splice them into the ordered chapter list under their own .m4a name so
            # every step below still sees exactly one entry per chapter, in chapter
            # order. Where a stale FLAC from an earlier run happens to survive, the
            # pre-encoded chapter wins — it is the one the caller asked us to ship.
            encoded_chapters = load_encoded_chapters(session, expected_chapter_nums)
            if encoded_chapters:
                chapter_files = [
                    f for f in chapter_files
                    if int(re.search(r'\d+', f).group()) not in encoded_chapters
                ]
                chapter_files = sorted(
                    chapter_files + [f'{n}.m4a' for n in encoded_chapters],
                    key=lambda x: int(re.search(r'\d+', x).group())
                )
                print(
                    f'[ASSEMBLE] {len(encoded_chapters)} of {len(chapter_files)} chapters arrive '
                    f'pre-encoded from BookForge; they will be copied into the audiobook as-is'
                )

            # Chapter marker titles are bound to chapters by DOCUMENT IDENTITY.
            # NEVER by position: the TOC and the chapter list describe different sets
            # (a part-title page has a TOC entry but yields no audio; leading front
            # matter yields audio but has no TOC entry) and the counts can coincidentally
            # match while the sets differ, so a length check cannot detect the skew.
            # Assembly can run in a different process than prepare (and under a
            # front-end that persists its own session-state.json), so recover the
            # provenance from process_dir when this session dict does not carry it.
            if len(session.get('chapter_docs', [])) != len(chapters_data):
                load_chapter_provenance(session_id)
            chapter_docs = session.get('chapter_docs', [])
            titles_by_doc = session.get('chapter_titles_by_doc', {})
            # A chapter's own first sentence is correct BY CONSTRUCTION — e2a voices the
            # heading as that chapter's first sentence, so it always describes that audio.
            own_titles = [c[0] if c else f'Chapter {i+1}' for i, c in enumerate(chapters_data)]
            if len(chapter_docs) != len(chapters_data):
                print(
                    f'[ASSEMBLE] Chapter provenance unusable: {len(chapter_docs)} chapter_docs entries for '
                    f'{len(chapters_data)} chapters. This session was prepared before document-identity chapter '
                    f'titles existed (or its provenance was not persisted). Using each chapter\'s OWN first '
                    f'sentence as its marker title. TOC titles are deliberately NOT paired by position — that '
                    f'pairing is what mislabelled chapters.'
                )
                chapter_titles = own_titles
            else:
                chapter_titles = []
                from_toc = 0
                for i, doc_name in enumerate(chapter_docs):
                    toc_title = titles_by_doc.get(doc_name)
                    if toc_title:
                        chapter_titles.append(toc_title)
                        from_toc += 1
                    else:
                        chapter_titles.append(own_titles[i])
                print(
                    f'[ASSEMBLE] {from_toc}/{len(chapters_data)} chapter titles resolved from the TOC by document '
                    f'identity; {len(chapters_data) - from_toc} from the chapter\'s own first sentence'
                )

            # Filter chapter_titles to match selected chapters
            if selected_chapters:
                # chapter_files are already filtered; selected_chapters are 1-indexed
                # chapter numbers. Every one MUST resolve, else the surviving titles
                # would slide onto the wrong files.
                out_of_range = [ch for ch in selected_chapters if not (1 <= ch <= len(chapter_titles))]
                if out_of_range:
                    error = (
                        f'combine_audio_chapters(): selected chapter(s) {out_of_range} have no title entry '
                        f'({len(chapter_titles)} chapters known). Refusing to assemble with misaligned markers.'
                    )
                    print(error)
                    return None
                chapter_titles = [chapter_titles[ch-1] for ch in selected_chapters]
            # INVARIANT: the files about to be concatenated must cover EXACTLY the
            # chapter set this function believes it is assembling (expected_chapter_nums,
            # computed above). The glob above collects whatever files happen to exist,
            # so a missing (or 0-byte) chapter file would otherwise produce a silently
            # SHORTER audiobook with misaligned chapter titles.
            # Every intended chapter file must exist and be non-empty, else abort.
            if expected_chapter_nums:
                files_by_num = {int(re.search(r'\d+', f).group()): f for f in chapter_files}
                missing_chapter_nums = [n for n in expected_chapter_nums if n not in files_by_num]
                empty_chapter_nums = [
                    n for n in expected_chapter_nums
                    if n in files_by_num and os.path.getsize(chapter_source_path(files_by_num[n])) == 0
                ]
                if missing_chapter_nums or empty_chapter_nums:
                    error = 'combine_audio_chapters(): refusing to assemble an incomplete audiobook.'
                    if missing_chapter_nums:
                        error += f' Missing chapter audio files for chapter(s): {missing_chapter_nums}.'
                    if empty_chapter_nums:
                        error += f' Empty (0-byte) chapter audio files for chapter(s): {empty_chapter_nums}.'
                    print(error)
                    return None
            is_gui_process = session['is_gui_process']
            if len(chapter_files) == 0:
                print('No block files exists!')
                return None
            chunks_size = 892
            total_duration = 0.0
            durations = []  # Track per-file durations for output_split mode
            # total_duration is what finalize_export() holds the finished m4b against
            # (the 2026-08-11 truncated-export guard), so it must be the sum of the
            # ACTUAL sources, not of the FLACs we happen to have built this run. A
            # pre-encoded chapter contributes its .m4a's duration — chapter_source_path()
            # is what makes that automatic. Measuring the .m4a is also strictly more
            # honest than measuring a FLAC would have been: the .m4a is the very stream
            # that gets copied into the audiobook.
            for i in range(0, len(chapter_files), chunks_size):
                filepaths = [chapter_source_path(f) for f in chapter_files[i:i + chunks_size]]
                chunk_durations = get_audiolist_duration(filepaths)
                total_duration += sum(chunk_durations.values())
                # Append durations in order. Indexed, not .get(…, 0.0): a chapter whose
                # duration went missing must raise here, because a 0.0 reads downstream
                # as a chapter of silence and would both mis-size the split parts and
                # slacken the export guard by exactly that chapter's length.
                for fp in filepaths:
                    durations.append(chunk_durations[os.path.realpath(fp)])
            if encoded_chapters:
                # The pre-encoded chapters have no FLAC to fall back to — assembly
                # deliberately never built one. So if the parallel export is not on the
                # table after all, there is no way to ship them and no honest way to
                # continue; the two decisions (here and in assemble_audiobook, which
                # asks the same question through the same function) have disagreed and
                # that is a bug to surface, not to paper over by re-concatenating
                # sentences we were told not to.
                reason = parallel_export_unsupported_reason(session, total_duration)
                if reason:
                    error = (
                        f'combine_audio_chapters(): {len(encoded_chapters)} pre-encoded chapter(s) were '
                        f'accepted but the parallel export path is unavailable — {reason}. Those chapters '
                        f'have no chapter FLAC to rebuild from. Refusing to assemble.'
                    )
                    print(error)
                    return None
            exported_files = []
            concat_dir = session['process_dir']
            if session['output_split']:
                part_files = []
                part_chapter_indices = []
                cur_part = []
                cur_indices = []
                cur_duration = 0
                max_part_duration = int(session['output_split_hours']) * 3600
                needs_split = total_duration > (int(session['output_split_hours']) * 2) * 3600
                for idx, (file, dur) in enumerate(zip(chapter_files, durations)):
                    if session['cancellation_requested']:
                        msg = 'Cancel requested'
                        print(msg)
                        return None
                    if cur_part and (cur_duration + dur > max_part_duration):
                        part_files.append(cur_part)
                        part_chapter_indices.append(cur_indices)
                        cur_part = []
                        cur_indices = []
                        cur_duration = 0
                    cur_part.append(file)
                    cur_indices.append(idx)
                    cur_duration += dur
                if cur_part:
                    part_files.append(cur_part)
                    part_chapter_indices.append(cur_indices)
                for part_idx, (part_file_list, indices) in enumerate(zip(part_files, part_chapter_indices)):
                    concat_list = os.path.join(concat_dir, f'concat_list_chapters_{part_idx+1:02d}.txt')
                    with open(concat_list, 'w') as f:
                        for file in part_file_list:
                            if session['cancellation_requested']:
                                msg = 'Cancel requested'
                                print(msg)
                                return None
                            path = Path(chapter_source_path(file))
                            f.write(f"file '{path.as_posix()}'\n")
                    merged_audio = Path(session['process_dir']) / f"{get_sanitized(session['metadata']['title'])}_part{part_idx+1}.{default_audio_proc_format}"
                    result = assemble_audio_chunks(str(concat_list), str(merged_audio), is_gui_process)
                    if not result:
                        error = f'assemble_audio_chunks() Final merge failed for part {part_idx+1}.'
                        print(error)
                        return None
                    metadata_file = Path(session['process_dir']) / f'metadata_part{part_idx+1}.txt'
                    part_chapters = [(chapter_files[i], chapter_titles[i]) for i in indices]
                    generate_ffmpeg_metadata(part_chapters, str(metadata_file), default_audio_proc_format)
                    final_file = Path(session['audiobooks_dir']) / (f"{session['final_name'].rsplit('.', 1)[0]}_part{part_idx+1}.{session['output_format']}" if needs_split else session['final_name'])
                    if export_audio(str(merged_audio), str(metadata_file), str(final_file)):
                        exported_files.append(str(final_file))
            else:
                metadata_file = os.path.join(session['process_dir'], 'metadata.txt')
                chapters_zip = list(zip(chapter_files, chapter_titles))
                generate_ffmpeg_metadata(chapters_zip, metadata_file, default_audio_proc_format)
                final_file = os.path.join(session['audiobooks_dir'], session['final_name'])
                chapter_paths = [chapter_source_path(f) for f in chapter_files]
                # Indexes into chapter_paths that are already AAC and only need placing
                # in the concat list.
                already_encoded = {
                    i for i, f in enumerate(chapter_files)
                    if int(re.search(r'\d+', f).group()) in encoded_chapters
                }
                if parallel_export_supported(total_duration):
                    # Encode chapters concurrently and stream-copy them together.
                    # This also skips building the whole-book intermediate FLAC —
                    # nothing else reads it, and on a 20 h book that alone was a 29 s
                    # write of 1.5 GB purely to hand one file to a single encoder.
                    if export_audio_parallel(chapter_paths, already_encoded, metadata_file, final_file, total_duration):
                        exported_files.append(final_file)
                else:
                    concat_list = os.path.join(concat_dir, f'concat_list_chapters_1.txt')
                    merged_audio = Path(session['process_dir']) / f"{get_sanitized(session['metadata']['title'])}.{default_audio_proc_format}"
                    with open(concat_list, 'w') as f:
                        for file in chapter_files:
                            if session['cancellation_requested']:
                                msg = 'Cancel requested'
                                print(msg)
                                return None
                            path = chapter_source_path(file).replace("\\", "/")
                            f.write(f"file '{path}'\n")
                    if is_gui_process:
                        progress_bar = gr.Progress(track_tqdm=False)
                    ok = assemble_audio_chunks(concat_list, merged_audio, is_gui_process)
                    if not ok:
                        print(f'assemble_audio_chunks() Final merge failed for {merged_audio}.')
                        return None
                    if export_audio(merged_audio, metadata_file, final_file):
                        exported_files.append(final_file)
            return exported_files if exported_files else None
        return None
    except Exception as e:
        DependencyError(e)
        return None

def assemble_audio_chunks(txt_file:str, out_file:str, is_gui_process:bool)->bool:

    def on_progress(p:float)->None:
        progress_bar(p / 100.0, desc='Assemble')

    try:
        total_duration = 0.0
        filepaths = []
        try:
            with open(txt_file, 'r') as f:
                for line in f:
                    if line.strip().startswith('file'):
                        file_path = (
                            line.strip()
                            .split('file ')[1]
                            .strip()
                            .strip("'")
                            .strip('"')
                        )
                        if os.path.exists(file_path):
                            filepaths.append(file_path)
            durations = get_audiolist_duration(filepaths)
            total_duration = sum(durations.values())
        except Exception as e:
            print(f'assemble_audio_chunks() open file {txt_file} Error: {e}')
            return False

        progress_bar = gr.Progress(track_tqdm=False)

        cmd = [
            shutil.which('ffmpeg'),
            '-hide_banner',
            '-nostats',
            '-hwaccel', 'auto',
            '-y', '-safe', '0',
            '-f', 'concat',
            '-i', txt_file,
            '-c:a', default_audio_proc_format,
            '-map_metadata', '-1',
            '-threads', '0',
            '-progress', 'pipe:2',
            '-nostats',
            out_file,
        ]
        proc_pipe = SubprocessPipe(
            cmd=cmd,
            is_gui_process=is_gui_process,
            total_duration=total_duration,
            msg='Assemble',
            on_progress=on_progress
        )
        if proc_pipe:
            # ffmpeg's concat demuxer can drop inputs (e.g. FLAC blocksize
            # mismatches) and still exit 0, so the exit code alone proves
            # nothing — verify the output actually contains all the input.
            actual_duration = get_audio_duration(out_file)
            tolerance = 0.5 + 0.01 * len(filepaths)
            delta = actual_duration - total_duration
            if abs(delta) > tolerance:
                error = (
                    f'assemble_audio_chunks() Output duration mismatch → {out_file}: '
                    f'expected {total_duration:.2f}s, got {actual_duration:.2f}s '
                    f'(delta {delta:+.2f}s, tolerance ±{tolerance:.2f}s). '
                    f'ffmpeg exited 0 but the output is missing input audio.'
                )
                print(error)
                return False
            msg = f'Completed → {out_file}'
            print(msg)
            return True
        else:
            error = f'Failed (proc_pipe) → {out_file}'
            print(error)
            return False
    except subprocess.CalledProcessError as e:
        DependencyError(e)
        return False
    except Exception as e:
        error = f'assemble_audio_chunks() Error: Failed to process {txt_file} → {out_file}: {e}'
        print(error)
        return False

def ellipsize_utf8_bytes(s:str, max_bytes:int, ellipsis:str='…')->str:
    s = '' if s is None else str(s)
    if max_bytes <= 0:
        return ''
    raw = s.encode('utf-8')
    e = ellipsis.encode('utf-8')
    if len(raw) <= max_bytes:
        return s
    if len(e) >= max_bytes:
        # return as many bytes of the ellipsis as fit
        return e[:max_bytes].decode('utf-8', errors='ignore')
    budget = max_bytes - len(e)
    out = bytearray()
    for ch in s:
        b = ch.encode('utf-8')
        if len(out) + len(b) > budget:
            break
        out.extend(b)
    return out.decode('utf-8') + ellipsis

def sanitize_meta_chapter_title(title:str, max_bytes:int=140)->str:
    # avoid None and embedded NULs which some muxers accidentally keep
    title = (title or '').replace('\x00', '')
    # A chapter's title is very often the chapter's own FIRST SENTENCE (see
    # own_titles in combine_audio_chapters — e2a voices the heading as that
    # sentence, so it always describes the audio), which means it arrives with
    # whatever SML that row carries. Since 2026-08-27 that includes the
    # [heading] marker, so this strips the whole unspoken set rather than the
    # one [pause] literal it used to: a marker printed into an m4b chapter name
    # is as wrong as one read aloud.
    title = SML_UNSPOKEN_PATTERN.sub('', title).strip()
    return ellipsize_utf8_bytes(title, max_bytes=max_bytes, ellipsis='…')

def clear_folder(folder_path:str)->None:
    for name in os.listdir(folder_path):
        path = os.path.join(folder_path, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.unlink(path)
        else:
            shutil.rmtree(path)

def delete_unused_tmp_dirs(web_dir:str, days:int, session_id:str)->None:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        dir_array = [
            tmp_dir,
            web_dir,
            os.path.join(models_dir, '__sessions'),
            os.path.join(voices_dir, '__sessions')
        ]
        current_user_dirs = {
            f"proc-{session['id']}",
            f"web-{session['id']}",
            f"voice-{session['id']}",
            f"model-{session['id']}"
        }
        current_time = time.time()
        threshold_time = current_time - (days * 24 * 60 * 60)  # Convert days to seconds
        for dir_path in dir_array:
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                for dir in os.listdir(dir_path):
                    if dir in current_user_dirs:        
                        full_dir_path = os.path.join(dir_path, dir)
                        if os.path.isdir(full_dir_path):
                            try:
                                dir_mtime = os.path.getmtime(full_dir_path)
                                dir_ctime = os.path.getctime(full_dir_path)
                                if dir_mtime < threshold_time and dir_ctime < threshold_time:
                                    shutil.rmtree(full_dir_path, ignore_errors=True)
                                    msg = f'Deleted expired session: {full_dir_path}'
                                    print(msg)
                            except Exception as e:
                                error = f'Error deleting {full_dir_path}: {e}'
                                print(error)

def get_compatible_tts_engines(language:str)->list[str]:
    return [
        engine
        for engine, cfg in default_engine_settings.items()
        if language in cfg.get('languages', {})
    ]

def convert_ebook_batch(args:dict)->tuple:
    if isinstance(args['ebook_list'], list):
        ebook_list = args['ebook_list'][:]
        for file in ebook_list: # Use a shallow copy
            if any(file.endswith(ext) for ext in ebook_formats):
                args['ebook'] = file
                print(f'Processing eBook file: {os.path.basename(file)}')
                progress_status, passed = convert_ebook(args)
                if passed is False:
                    msg = f'Conversion failed: {progress_status}'
                    print(msg)
                    if not args['is_gui_process']:
                        sys.exit(1)
                args['ebook_list'].remove(file) 
        reset_session(args['id'])
        return progress_status, passed
    else:
        error = f'the ebooks source is not a list!'
        print(error)
        if not args['is_gui_process']:
            sys.exit(1)       

def convert_ebook(args:dict)->tuple:
    try:
        if args.get('event') == 'blocks_confirmed':
            if args.get('id', False):
                progress_status, passed = finalize_audiobook(args['id'])
                return progress_status, passed
            else:
                error = f"convert_ebook() error: args['id'] is False"
                print(error)
                return error, False
        else:
            global context        
            error = None
            session_id = None
            info_session = None
            if args['language'] is not None:
                if not os.path.splitext(args['ebook'])[1]:
                    error = f"{args['ebook']} needs a format extension."
                    print(error)
                    return error, False
                if not os.path.exists(args['ebook']):
                    error = 'File does not exist or Directory empty.'
                    print(error)
                    return error, False
                try:
                    if len(args['language']) in (2, 3):
                        lang_dict = Lang(args['language'])
                        if lang_dict:
                            args['language'] = lang_dict.pt3
                            args['language_iso1'] = lang_dict.pt1
                    else:
                        args['language_iso1'] = None
                except Exception as e:
                    pass
                if args['language'] not in language_mapping.keys():
                    error = 'The language you provided is not (yet) supported'
                    print(error)
                    return error, False
                if args['id'] is not None:
                    session_id = str(args['id'])
                    session = context.get_session(session_id)
                    if not session:
                        session = context.set_session(session_id)
                else:
                    session_id = str(uuid.uuid4())
                    session = context.set_session(session_id)
                    if not context_tracker.start_session(session_id):
                        error = 'convert_ebook() error: Session initialization failed!'
                        print(error)
                        return error, False     
                session['custom_model_dir'] = os.path.join(models_dir, '__sessions',f"model-{session_id}")
                session['script_mode'] = str(args['script_mode']) if args.get('script_mode') is not None else NATIVE
                session['is_gui_process'] = bool(args['is_gui_process'])
                session['ebook'] = str(args['ebook']) if args.get('ebook') else None
                session['ebook_list'] = list(args['ebook_list']) if args.get('ebook_list') else None
                session['chapters_preview'] = bool(args['chapters_preview']) if args.get('chapters_preview') else False
                session['device'] = str(args['device'])
                session['language'] = str(args['language'])
                session['language_iso1'] = str(args['language_iso1'])
                session['tts_engine'] = str(args['tts_engine']) if args['tts_engine'] is not None else str(get_compatible_tts_engines(args['language'])[0])
                session['custom_model'] =  os.path.join(session['custom_model_dir'], args['custom_model']) if args['custom_model'] is not None else None
                session['fine_tuned'] = str(args['fine_tuned'])
                session['voice'] = str(args['voice']) if args['voice'] is not None else None
                session['xtts_temperature'] =  float(args['xtts_temperature'])
                session['xtts_length_penalty'] = float(args['xtts_length_penalty'])
                session['xtts_num_beams'] = int(args['xtts_num_beams'])
                session['xtts_repetition_penalty'] = float(args['xtts_repetition_penalty'])
                session['xtts_top_k'] =  int(args['xtts_top_k'])
                session['xtts_top_p'] = float(args['xtts_top_p'])
                session['xtts_speed'] = float(args['xtts_speed'])
                session['xtts_enable_text_splitting'] = bool(args['xtts_enable_text_splitting'])
                session['sentence_per_paragraph'] = bool(args.get('sentence_per_paragraph', False))
                session['skip_headings'] = bool(args.get('skip_headings', False))
                session['bark_text_temp'] =  float(args['bark_text_temp'])
                session['bark_waveform_temp'] =  float(args['bark_waveform_temp'])
                session['audiobooks_dir'] = str(args['audiobooks_dir']) if args['audiobooks_dir'] else None
                session['output_format'] = str(args['output_format'])
                session['output_channel'] = str(args['output_channel'])
                session['output_split'] = bool(args['output_split'])
                session['output_split_hours'] = args['output_split_hours']if args['output_split_hours'] is not None else default_output_split_hours
                session['model_cache'] = f"{session['tts_engine']}-{session['fine_tuned']}"
                cleanup_models_cache()
                if not session['is_gui_process']:
                    session['system'] = sys.platform
                    session['session_dir'] = os.path.join(tmp_dir, f"proc-{session['id']}")
                    session['voice_dir'] = os.path.join(voices_dir, '__sessions', f"voice-{session['id']}", session['language'])
                    os.makedirs(session['voice_dir'], exist_ok=True)
                    if session['custom_model'] is not None:
                        if not os.path.exists(session['custom_model_dir']):
                            os.makedirs(session['custom_model_dir'], exist_ok=True)
                        custom_src_path = Path(session['custom_model'])
                        custom_src_name = custom_src_path.stem
                        if not os.path.exists(os.path.join(session['custom_model_dir'], custom_src_name)):
                            try:
                                if analyze_uploaded_file(session['custom_model'], default_engine_settings[session['tts_engine']]['files']):
                                    model = extract_custom_model(session['custom_model'], session_id, default_engine_settings[session['tts_engine']]['files'])
                                    if model is not None:
                                        session['custom_model'] = model
                                    else:
                                        error = f"{model} could not be extracted or mandatory files are missing"
                                else:
                                    error = f'{os.path.basename(session["custom_model"])} is not a valid model or some required files are missing'
                            except ModuleNotFoundError as e:
                                error = f"No presets module for TTS engine '{session['tts_engine']}': {e}"
                                print(error)
                    if session['voice'] is not None:
                        voice_name = os.path.splitext(os.path.basename(session['voice']))[0].replace('&', 'And')
                        voice_name = get_sanitized(voice_name)
                        final_voice_file = os.path.join(session['voice_dir'], f'{voice_name}.wav')
                        if not os.path.exists(final_voice_file):
                            extractor = VoiceExtractor(session, session['voice'], voice_name)
                            status, msg = extractor.extract_voice()
                            if status:
                                session['voice'] = final_voice_file
                            else:
                                error = f'VoiceExtractor.extract_voice() failed! {msg}'
                                print(error)
                if error is None:
                    if session['script_mode'] == NATIVE:
                        is_installed = check_programs('Calibre', 'ebook-convert', '--version')
                        if not is_installed:
                            error = f'check_programs() Calibre failed: {e}'
                        is_installed = check_programs('FFmpeg', 'ffmpeg', '-version')
                        if not is_installed:
                            error = f'check_programs() FFMPEG failed: {e}'
                    if error is None:
                        old_session_dir = os.path.join(tmp_dir, f"ebook-{session['id']}")
                        if os.path.isdir(old_session_dir):
                            os.rename(old_session_dir, session['session_dir'])
                        session['final_name'] = get_sanitized(Path(session['ebook']).stem + '.' + session['output_format'])
                        session['process_dir'] = os.path.join(session['session_dir'], f"{hashlib.md5(os.path.join(session['audiobooks_dir'], session['final_name']).encode()).hexdigest()}")
                        session['chapters_dir'] = os.path.join(session['process_dir'], "chapters")
                        session['sentences_dir'] = os.path.join(session['chapters_dir'], 'sentences')
                        if prepare_dirs(args['ebook'], session_id):
                            session['filename_noext'] = os.path.splitext(os.path.basename(session['ebook']))[0]
                            msg = ''
                            msg_extra = ''
                            vram_dict = VRAMDetector().detect_vram(session['device'], session['script_mode'])
                            print(f'vram_dict: {vram_dict}')
                            total_vram_gb = vram_dict.get('total_vram_gb', 0)
                            session['free_vram_gb'] = vram_dict.get('free_vram_gb', 0)
                            if session['free_vram_gb'] == 0:
                                session['free_vram_gb'] = 1.0
                                msg_extra += '<br/>Memory capacity not detected! restrict to 1GB max' if session['free_vram_gb'] == 0 else f"<br/>Memory detected with {session['free_vram_gb']}GB"
                            else:
                                msg_extra += f"<br/>Free Memory available: {session['free_vram_gb']}GB"
                                if session['free_vram_gb'] > 4.0:
                                    if session['tts_engine'] == TTS_ENGINES['BARK']:
                                        os.environ['SUNO_USE_SMALL_MODELS'] = 'False'                        
                            if session['device'] == devices['CUDA']['proc'] or session['device'] == devices['JETSON']['proc']:
                                session['device'] = session['device'] if devices['CUDA']['found'] else devices['CPU']['proc']
                                if session['device'] == devices['CPU']['proc']:
                                    msg += f'CUDA not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                            elif session['device'] == devices['MPS']['proc']:
                                if not devices['MPS']['found']:
                                    session['device'] = devices['CPU']['proc']
                                    msg += f'MPS not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                            elif session['device'] == devices['ROCM']['proc']:
                                session['device'] = session['device'] if devices['ROCM']['found'] else devices['CPU']['proc']
                                if session['device'] == devices['CPU']['proc']:
                                    msg += f'ROCM not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU'
                            elif session['device'] == devices['XPU']['proc']:
                                session['device'] = session['device'] if devices['XPU']['found'] else devices['CPU']['proc']
                                if session['device'] == devices['CPU']['proc']:
                                    msg += f"XPU not supported by the Torch installed!<br/>Read {default_gpu_wiki}<br/>Switching to CPU"
                            if session['tts_engine'] == TTS_ENGINES['BARK']:
                                if session['free_vram_gb'] < 12.0:
                                    os.environ['SUNO_OFFLOAD_CPU'] = "True"
                                    os.environ['SUNO_USE_SMALL_MODELS'] = "True"
                                    msg_extra += f"<br/>Switching BARK to SMALL models"  
                                else:
                                    os.environ['SUNO_OFFLOAD_CPU'] = "False"
                                    os.environ['SUNO_USE_SMALL_MODELS'] = "False"
                            if msg == '':
                                msg = f"Using {session['device'].upper()}"
                            msg += msg_extra;
                            device_vram_required = default_engine_settings[session['tts_engine']]['rating']['RAM'] if session['device'] == devices['CPU']['proc'] else default_engine_settings[session['tts_engine']]['rating']['VRAM']
                            if float(total_vram_gb) >= float(device_vram_required):
                                if session['is_gui_process']:
                                    show_alert({"type": "warning", "msg": msg})
                                print(msg.replace('<br/>','\n'))
                                session['epub_path'] = os.path.join(session['process_dir'], f"__{session['filename_noext']}.epub")
                                checksum_path = os.path.join(session['process_dir'], 'checksum')
                                checksum, error = compare_checksums(session['ebook'], checksum_path)
                                if error is None:
                                    saved_json_chapters = os.path.join(session['process_dir'], f"__{session['filename_noext']}.json")
                                    if not checksum:
                                        session['chapters'] = []
                                        if not convert2epub(session_id):
                                            error = 'convert2epub() failed!'
                                    else:
                                        try:
                                            session['chapters'] = load_json_chapters(saved_json_chapters)
                                            load_chapter_provenance(session_id)
                                        except Exception as e:
                                            error = str(e)
                                    if error is None:
                                        epubBook = epub.read_epub(session['epub_path'], {'ignore_ncx': True})
                                        if epubBook:
                                            metadata = dict(session['metadata'])
                                            for key, value in metadata.items():
                                                data = epubBook.get_metadata('DC', key)
                                                if data:
                                                    for value, attributes in data:
                                                        metadata[key] = value
                                            metadata['language'] = session['language']
                                            metadata['title'] = metadata['title'] = metadata['title'] or Path(session['ebook']).stem.replace('_',' ')
                                            metadata['creator'] =  False if not metadata['creator'] or metadata['creator'] == 'Unknown' else metadata['creator']
                                            session['metadata'] = metadata                  
                                            try:
                                                if len(session['metadata']['language']) == 2:
                                                    lang_dict = Lang(session['language'])
                                                    if lang_dict:
                                                        session['metadata']['language'] = lang_dict.pt3
                                            except Exception as e:
                                                pass                         
                                            if session['metadata']['language'] != session['language']:
                                                error = f"WARNING!!! language selected {session['language']} differs from the EPUB file language {session['metadata']['language']}"
                                                print(error)
                                                if session['is_gui_process']:
                                                    show_alert({"type": "warning", "msg": error})
                                            is_lang_in_tts_engine = (
                                                session.get('tts_engine') in default_engine_settings and
                                                session.get('language') in default_engine_settings[session['tts_engine']].get('languages', {})
                                            )
                                            if is_lang_in_tts_engine:
                                                session['cover'] = get_cover(epubBook, session_id)
                                                if session['cover']:
                                                    if not session['chapters']:
                                                        session['chapters'] = get_chapters(session_id, epubBook)
                                                    if session['chapters']:
                                                        #if session['chapters_preview']:
                                                        #   return 'confirm_blocks', True
                                                        #else:
                                                        progress_status, passed = finalize_audiobook(session_id)
                                                        return progress_status, passed
                                                    else:
                                                        error = f"get_chapters() failed! {session['chapters']}"
                                                else:
                                                    error = 'get_cover() failed!'
                                            else:
                                                 error = f"language {session['language']} not supported by {session['tts_engine']}!"
                                        else:
                                            error = 'epubBook.read_epub failed!'
                            else:
                                error = f"Your device has not enough memory ({total_vram_gb}GB) to run {session['tts_engine']} engine ({device_vram_required}GB)"
                        else:
                            error = f"Temporary directory {session['process_dir']} not removed due to failure."
            else:
                error = f"Language {args['language']} is not supported."
        if session['cancellation_requested']:
            error = 'Cancelled' if error is None else error + '. Cancelled'
        print(error)
        if session['is_gui_process']:
            show_alert({"type": "warning", "msg": error})
        return error, False
    except Exception as e:
        print(f'convert_ebook() Exception: {e}')
        return e, False

def finalize_audiobook(session_id:str)->tuple:
    session = context.get_session(session_id)
    if session and session.get('id', False):
        if session['chapters']:
            saved_json_chapters = os.path.join(session['process_dir'], f"__{session['filename_noext']}.json")
            save_json_chapters(session_id, saved_json_chapters)
            if convert_chapters2audio(session_id):
                msg = 'Conversion successful. Combining sentences and chapters…'
                show_alert({"type": "info", "msg": msg})
                exported_files = combine_audio_chapters(session['id'])               
                if exported_files is not None:
                    progress_status = f'Audiobook {", ".join(os.path.basename(f) for f in exported_files)} created!'
                    session['audiobook'] = exported_files[-1]
                    if not session['is_gui_process']:
                        process_dir = os.path.join(session['session_dir'], f"{hashlib.md5(os.path.join(session['audiobooks_dir'], session['audiobook']).encode()).hexdigest()}")
                        shutil.rmtree(process_dir, ignore_errors=True)
                    info_session = f"\n*********** Session: {session_id} **************\nIn headless mode, store it in case of interruption, crash, or reuse of a custom model or custom voice.\nYou can resume the conversion with the --session option."
                    print(info_session)
                    return progress_status, True
                else:
                    error = 'combine_audio_chapters() error: exported_files not created!'
            else:
                error = 'convert_chapters2audio() failed!'
        else:
            error = 'finalize_audiobook() failed!'
    return error, False

def restore_session_from_data(data:dict, session:dict)->None:
    try:
        for key, value in data.items():
            if key in session:
                if isinstance(value, dict) and isinstance(session[key], dict):
                    restore_session_from_data(value, session[key])
                else:
                    if value is None and session[key] is not None:
                        continue
                    session[key] = value
    except Exception as e:
        DependencyError(e)

def cleanup_session(req:gr.Request)->None:
    socket_hash = req.session_hash
    if any(socket_hash in session for session in context.sessions.values()):
        session_id = context.find_id_by_hash(socket_hash)
        context_tracker.end_session(session_id, socket_hash)

def reset_session(session_id:str)->None:
    session = context.get_session(session_id)
    data = {
        "process_id": None,
        "event": None,
        "ticker": 0,
        "process_dir": None,
        "ebook": None,
        "ebook_list": None,
        "epub_path": None,
        "filename_noext": None,
        "final_name": None,
        "metadata": {
            "title": None, 
            "creator": None,
            "contributor": None,
            "language": None,
            "identifier": None,
            "publisher": None,
            "date": None,
            "description": None,
            "subject": None,
            "rights": None,
            "format": None,
            "type": None,
            "coverage": None,
            "relation": None,
            "Source": None,
            "Modified": None,
        },
        "chapters": [],
        "cover": None,
        "duration": 0,
        "playback_time": 0,
        "playback_volume": 0
    }
    restore_session_from_data(data, session)

def cleanup_models_cache()->None:
    try:
        active_models = {
            cache
            for session in context.sessions.values()
            for cache in (session.get('model_cache'), session.get('model_zs_cache'), session.get('stanza_cache'))
            if cache is not None
        }
        for key in list(loaded_tts.keys()):
            if key not in active_models:
                del loaded_tts[key]
        gc.collect()
    except Exception as e:
        error = f"cleanup_models_cache() error: {e}"
        print(error)

def show_alert(state:dict)->None:
    if isinstance(state, dict):
        if state['type'] is not None:
            if state['type'] == 'error':
                gr.Error(state['msg'])
            elif state['type'] == 'warning':
                gr.Warning(state['msg'])
            elif state['type'] == 'info':
                gr.Info(state['msg'])
            elif state['type'] == 'success':
                gr.Success(state['msg'])

def alert_exception(error:str, session_id:str|None)->None:
    if session_id is not None:
        session = context.get_session(session_id)
        if session and session.get('id', False):
            session['status'] = 'ready'
    print(error)
    gr.Error(error)
    DependencyError(error)

def get_all_ip_addresses()->list:
    ip_addresses = []
    for interface, addresses in psutil.net_if_addrs().items():
        for address in addresses:
            if address.family in [socket.AF_INET, socket.AF_INET6]:
                ip_addresses.append(address.address)
    return ip_addresses


# =============================================================================
# PARALLEL WORKER SUPPORT (BookForge Integration)
# These functions support the three-phase parallel TTS architecture:
#   Phase 1: --prep_only    - Parse EPUB, save sentence data to session-state.json
#   Phase 2: --worker_mode  - Load session state, run TTS for assigned range
#   Phase 3: --assemble_only - Combine sentence audio files into final audiobook
# =============================================================================

def prep_ebook_info(args: dict) -> dict | None:
    """
    Prepare EPUB information for parallel worker coordination.
    Returns a dict with session_id, total_sentences, total_chapters, and session_dir.
    This is used by BookForgeApp to coordinate parallel TTS workers.
    """
    try:
        global context, context_tracker

        # Create session
        session_id = args.get('session') or str(uuid.uuid4())
        session = context.set_session(session_id)
        context_tracker.start_session(session_id)

        # Basic setup
        session['script_mode'] = args.get('script_mode', NATIVE)
        session['is_gui_process'] = args.get('is_gui_process', False)
        session['ebook'] = args['ebook']
        session['device'] = args.get('device', default_device)
        session['language'] = args.get('language', default_language_code)

        # Handle language ISO conversion
        try:
            if len(session['language']) in (2, 3):
                lang_dict = Lang(session['language'])
                if lang_dict:
                    session['language'] = lang_dict.pt3
                    session['language_iso1'] = lang_dict.pt1
            if not session.get('language_iso1'):
                session['language_iso1'] = session['language'][:2] if len(session['language']) >= 2 else 'en'
        except Exception:
            session['language_iso1'] = 'en'

        session['tts_engine'] = args.get('tts_engine') or get_compatible_tts_engines(session['language'])[0]
        session['audiobooks_dir'] = args.get('audiobooks_dir', audiobooks_cli_dir)
        session['fine_tuned'] = args.get('fine_tuned', default_fine_tuned)
        session['voice'] = args.get('voice')
        session['output_format'] = args.get('output_format', default_output_format)

        print(f"[PREP] fine_tuned={session['fine_tuned']}, voice={session['voice']}, tts_engine={session['tts_engine']}")

        # Setup directories using ebook- prefix for parallel sessions
        session['session_dir'] = os.path.join(tmp_dir, f"ebook-{session_id}")
        session['process_dir'] = os.path.join(session['session_dir'], f"{hashlib.md5(session['ebook'].encode()).hexdigest()}")
        session['chapters_dir'] = os.path.join(session['process_dir'], 'chapters')
        session['sentences_dir'] = os.path.join(session['chapters_dir'], 'sentences')
        session['custom_model_dir'] = os.path.join(models_dir, '__sessions', f"model-{session_id}")
        session['voice_dir'] = os.path.join(voices_dir, '__sessions', f"voice-{session_id}", session['language'])

        # Detect VRAM
        vram_dict = VRAMDetector().detect_vram(session['device'], session['script_mode'])
        session['free_vram_gb'] = vram_dict.get('free_vram_gb', 4)

        if prepare_dirs(args['ebook'], session_id):
            session['filename_noext'] = os.path.splitext(os.path.basename(session['ebook']))[0]
            session['epub_path'] = os.path.join(session['process_dir'], f"__{session['filename_noext']}.epub")
            session['final_name'] = get_sanitized(session['filename_noext'] + '.' + session['output_format'])

            if convert2epub(session_id):
                epubBook = epub.read_epub(session['epub_path'], {'ignore_ncx': True})

                # Get metadata
                metadata = dict(session['metadata'])
                for key in metadata.keys():
                    data = epubBook.get_metadata('DC', key)
                    if data:
                        for value, attributes in data:
                            metadata[key] = value
                metadata['title'] = metadata['title'] if metadata['title'] else session['filename_noext'].replace('_', ' ')
                session['metadata'] = metadata

                # Get cover
                try:
                    session['cover'] = get_cover(epubBook, session_id)
                except Exception as cover_err:
                    print(f'Warning: Could not get cover: {cover_err}')
                    session['cover'] = None

                # Get chapters
                session['chapters'] = get_chapters(session_id, epubBook)
                if session['chapters']:
                    total_chapters = len(session['chapters'])
                    total_sentences = sum(len(chapter) for chapter in session['chapters'])

                    # Build chapter info
                    chapter_info = []
                    sentence_offset = 0
                    for i, chapter in enumerate(session['chapters']):
                        chapter_info.append({
                            'chapter_num': i + 1,
                            'sentence_count': len(chapter),
                            'sentence_start': sentence_offset,
                            'sentence_end': sentence_offset + len(chapter) - 1
                        })
                        sentence_offset += len(chapter)

                    result = {
                        'session_id': session_id,
                        'session_dir': session['session_dir'],
                        'process_dir': session['process_dir'],
                        'chapters_dir': session['chapters_dir'],
                        'chapters_dir_sentences': session['sentences_dir'],
                        'total_chapters': total_chapters,
                        'total_sentences': total_sentences,
                        'chapters': chapter_info,
                        'chapter_sentences': list(session['chapters']),
                        'metadata': {
                            'title': metadata.get('title'),
                            'creator': metadata.get('creator'),
                            'language': metadata.get('language')
                        }
                    }

                    # Save session state for workers
                    save_session_state(session_id, args, result)
                    return result

        return None
    except Exception as e:
        print(f'prep_ebook_info() Exception: {e}')
        import traceback
        traceback.print_exc()
        return None


def save_session_state(session_id: str, args: dict, prep_result: dict) -> bool:
    """
    Save session state to session-state.json for resume capability.
    """
    try:
        session = context.get_session(session_id)
        state = {
            'version': 2,
            'session_id': prep_result['session_id'],
            'epub_path': session['ebook'],
            'source_epub_path': args.get('ebook'),
            'epub_content_hash': hashlib.md5(session['ebook'].encode()).hexdigest(),
            'total_sentences': prep_result['total_sentences'],
            'total_chapters': prep_result['total_chapters'],
            'chapters': prep_result['chapters'],
            'chapter_sentences': prep_result.get('chapter_sentences', []),
            'language': session.get('language', default_language_code),
            'language_iso1': session.get('language_iso1', 'en'),
            'voice': session.get('voice'),
            'fine_tuned': session.get('fine_tuned'),
            'tts_engine': session.get('tts_engine'),
            'device': args.get('device', default_device),
            'output_format': args.get('output_format', default_output_format),
            'audiobooks_dir': args.get('audiobooks_dir', audiobooks_cli_dir),
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'status': 'prepared',
            'metadata': prep_result.get('metadata', {}),
            'session_dir': session['session_dir'],
            'process_dir': session['process_dir'],
            'chapters_dir': session['chapters_dir'],
            'chapters_dir_sentences': session['sentences_dir'],
            'epub_path_internal': session.get('epub_path'),
            'filename_noext': session.get('filename_noext'),
            'cover': session.get('cover'),
            'final_name': session.get('final_name'),
            'chapter_titles': session.get('chapter_titles', []),  # TOC titles (flat, legacy consumers)
            # Chapter provenance: chapter_docs[i] names the spine document that produced
            # chapters[i]; chapter_titles_by_doc maps that document name to its TOC title.
            # Assembly pairs titles to chapters through these two, never by position.
            'chapter_docs': list(session.get('chapter_docs', [])),
            'chapter_titles_by_doc': dict(session.get('chapter_titles_by_doc', {})),
        }

        state_path = os.path.join(session['process_dir'], 'session-state.json')
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2)

        print(f'Session state saved to {state_path}')
        return True
    except Exception as e:
        print(f'Warning: Failed to save session state: {e}')
        return False


def load_session_state(session_dir: str) -> dict | None:
    """
    Load session state from session-state.json.
    """
    try:
        process_dirs = [d for d in os.listdir(session_dir)
                       if os.path.isdir(os.path.join(session_dir, d))]

        for process_dir_name in process_dirs:
            state_path = os.path.join(session_dir, process_dir_name, 'session-state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                state['process_dir'] = os.path.join(session_dir, process_dir_name)
                state['session_dir'] = session_dir
                return state
        return None
    except Exception as e:
        print(f'Failed to load session state: {e}')
        return None


def scan_completed_sentences(sentences_dir: str, total_sentences: int, min_file_size: int = 1024) -> dict:
    """
    Scan the sentences directory to find completed audio files.
    """
    completed = []
    missing = []

    for i in range(total_sentences):
        found = False
        for ext in ['flac', 'wav', 'mp3']:
            file_path = os.path.join(sentences_dir, f'{i}.{ext}')
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                if file_size >= min_file_size:
                    completed.append(i)
                    found = True
                    break
        if not found:
            missing.append(i)

    return {
        'completed': completed,
        'missing': missing,
        'completed_count': len(completed),
        'missing_count': len(missing),
        'progress_percent': round(len(completed) / total_sentences * 100, 1) if total_sentences > 0 else 0
    }


def list_resumable_sessions() -> list:
    """
    List all sessions that have session-state.json and incomplete sentences.
    """
    resumable = []

    if not os.path.exists(tmp_dir):
        return resumable

    for session_name in os.listdir(tmp_dir):
        if not session_name.startswith('ebook-'):
            continue

        session_dir = os.path.join(tmp_dir, session_name)
        if not os.path.isdir(session_dir):
            continue

        state = load_session_state(session_dir)
        if not state:
            continue

        sentences_dir = os.path.join(state['process_dir'], 'chapters', 'sentences')
        if not os.path.exists(sentences_dir):
            continue

        scan_result = scan_completed_sentences(sentences_dir, state['total_sentences'])

        if scan_result['missing_count'] > 0:
            resumable.append({
                'session_id': state['session_id'],
                'session_dir': session_dir,
                'title': state.get('metadata', {}).get('title', 'Unknown'),
                'total_sentences': state['total_sentences'],
                'completed_sentences': scan_result['completed_count'],
                'missing_sentences': scan_result['missing_count'],
                'progress_percent': scan_result['progress_percent'],
                'created_at': state.get('created_at'),
                'language': state.get('language'),
                'voice': state.get('voice')
            })

    return resumable


def check_resume_compatibility(state: dict, args: dict) -> dict:
    """
    Check if current settings are compatible with saved session state.
    """
    warnings = []

    saved_voice = state.get('voice')
    current_voice = args.get('voice')
    if saved_voice and current_voice and saved_voice != current_voice:
        warnings.append(f"Voice changed from '{saved_voice}' to '{current_voice}'")

    saved_engine = state.get('tts_engine')
    current_engine = args.get('tts_engine')
    if saved_engine and current_engine and saved_engine != current_engine:
        warnings.append(f"TTS engine changed from '{saved_engine}' to '{current_engine}'")

    return {'compatible': True, 'warnings': warnings}


def resume_session(args: dict) -> dict:
    """
    Resume a partially completed TTS session.
    """
    try:
        global context, context_tracker

        session_path = args.get('resume_session')
        if not session_path:
            return {'success': False, 'error': 'No session path provided'}

        if not os.path.isabs(session_path):
            if session_path.startswith('ebook-'):
                session_dir = os.path.join(tmp_dir, session_path)
            else:
                session_dir = os.path.join(tmp_dir, f'ebook-{session_path}')
        else:
            session_dir = session_path

        if not os.path.exists(session_dir):
            return {'success': False, 'error': f'Session directory not found: {session_dir}'}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': 'No session-state.json found'}

        compat = check_resume_compatibility(state, args)
        if compat['warnings']:
            for warning in compat['warnings']:
                print(f'Warning: {warning}')

        sentences_dir = os.path.join(state['process_dir'], 'chapters', 'sentences')
        if not os.path.exists(sentences_dir):
            return {'success': False, 'error': f'Sentences directory not found: {sentences_dir}'}

        scan_result = scan_completed_sentences(sentences_dir, state['total_sentences'])

        if scan_result['missing_count'] == 0:
            return {
                'success': True,
                'complete': True,
                'message': 'All sentences already complete - ready for assembly',
                'session_id': state['session_id'],
                'session_dir': session_dir,
                'process_dir': state['process_dir']
            }

        missing_ranges = calculate_missing_ranges(scan_result['missing'])

        return {
            'success': True,
            'complete': False,
            'session_id': state['session_id'],
            'session_dir': session_dir,
            'process_dir': state['process_dir'],
            'chapters_dir': os.path.join(state['process_dir'], 'chapters'),
            'chapters_dir_sentences': sentences_dir,
            'total_sentences': state['total_sentences'],
            'total_chapters': state['total_chapters'],
            'completed_sentences': scan_result['completed_count'],
            'missing_sentences': scan_result['missing_count'],
            'missing_indices': scan_result['missing'],
            'missing_ranges': missing_ranges,
            'progress_percent': scan_result['progress_percent'],
            'chapters': state.get('chapters', []),
            'metadata': state.get('metadata', {}),
            'warnings': compat['warnings']
        }

    except Exception as e:
        print(f'resume_session() Exception: {e}')
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def calculate_missing_ranges(missing_indices: list) -> list:
    """
    Convert list of missing sentence indices into contiguous ranges.
    """
    if not missing_indices:
        return []

    ranges = []
    sorted_indices = sorted(missing_indices)
    range_start = sorted_indices[0]
    range_end = sorted_indices[0]

    for i in range(1, len(sorted_indices)):
        if sorted_indices[i] == range_end + 1:
            range_end = sorted_indices[i]
        else:
            ranges.append({'start': range_start, 'end': range_end, 'count': range_end - range_start + 1})
            range_start = sorted_indices[i]
            range_end = sorted_indices[i]

    ranges.append({'start': range_start, 'end': range_end, 'count': range_end - range_start + 1})
    return ranges


def build_vtt_file(session_id: str, all_sentences: list) -> bool:
    """
    Build a VTT (WebVTT) subtitle file from sentence audio files.
    Called during assembly phase to create subtitle track for the audiobook.
    """
    from pathlib import Path
    import re

    try:
        session = context.get_session(session_id)
        if not session:
            print(f"[VTT] No session found for {session_id}")
            return False

        sentences_dir = session.get('sentences_dir')
        process_dir = session.get('process_dir')
        final_name = session.get('final_name', 'audiobook.m4b')

        if not sentences_dir or not process_dir:
            print("[VTT] Missing sentences_dir or process_dir in session")
            return False

        # Flatten all_sentences if it's a list of chapters
        flat_sentences = []
        for item in all_sentences:
            if isinstance(item, list):
                flat_sentences.extend(item)
            else:
                flat_sentences.append(item)

        vtt_path = os.path.join(process_dir, Path(final_name).stem + '.vtt')
        print(f"[VTT] Building VTT file: {vtt_path}")

        # Get sorted audio files
        audio_dir = Path(sentences_dir)
        audio_files = sorted(
            audio_dir.glob(f'*.{default_audio_proc_format}'),
            key=lambda p: int(p.stem)
        )

        if len(audio_files) != len(flat_sentences):
            print(f"[VTT] Warning: {len(audio_files)} audio files vs {len(flat_sentences)} sentences")
            # Use minimum to avoid index errors
            count = min(len(audio_files), len(flat_sentences))
            audio_files = audio_files[:count]
            flat_sentences = flat_sentences[:count]

        if not audio_files:
            print("[VTT] No audio files found")
            return False

        # Durations come from each file's own container header — no subprocess.
        # See audio_duration_seconds(): exact, ~1000x faster than one ffprobe per
        # sentence, and it raises on a damaged file rather than timing it as 0.0.
        print("[VTT] Getting audio durations...")
        durations = {str(p): audio_duration_seconds(str(p)) for p in audio_files}

        # Build VTT content
        print("[VTT] Creating VTT blocks...")
        vtt_blocks = []
        current_time = 0.0

        # SML tag pattern for cleaning. Shared with the engines as of 2026-08-27:
        # this local copy had already drifted — it never stripped [pause], so a
        # blank-line cue could read "[pause]" on screen — and it would have shown
        # the new [heading] marker in every chapter-title cue.
        SML_PATTERN = SML_UNSPOKEN_PATTERN

        def format_timestamp(seconds: float) -> str:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            s = seconds % 60
            return f'{int(h):02}:{int(m):02}:{s:06.3f}'

        for idx, audio_file in enumerate(audio_files):
            start_time = current_time
            duration = durations.get(str(audio_file), 0.0)
            end_time = start_time + duration
            current_time = end_time

            # Cue text: markers stripped, and BOLD when the row is a heading.
            # Built by the shared vtt_cue_text so this copy, the parallel
            # session.py one and TTSUtils._build_vtt_file cannot drift apart
            # again (2026-08-27).
            text = vtt_cue_text(str(flat_sentences[idx]), SML_PATTERN)

            start_str = format_timestamp(start_time)
            end_str = format_timestamp(end_time)
            vtt_blocks.append(f'{start_str} --> {end_str}\n{text}\n')

        # Write VTT file
        print(f"[VTT] Writing {len(vtt_blocks)} blocks to {vtt_path}")
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write('WEBVTT\n\n')
            f.write('\n'.join(vtt_blocks))

        print("[VTT] VTT file created successfully")
        return True

    except Exception as e:
        print(f"[VTT] Error building VTT file: {e}")
        import traceback
        traceback.print_exc()
        return False


def parse_chapters_arg(chapters_arg: str, total_chapters: int) -> list[int] | None:
    """
    Parse the --chapters argument into a list of chapter numbers (1-indexed).

    Supported formats:
    - "1-5" -> [1, 2, 3, 4, 5]
    - "1,3,5" -> [1, 3, 5]
    - "1-3,7,9-11" -> [1, 2, 3, 7, 9, 10, 11]
    - "auto" -> None (signals auto-detection)

    Returns None for "auto" mode, or a sorted list of chapter numbers.
    """
    if not chapters_arg:
        return list(range(1, total_chapters + 1))  # All chapters

    chapters_arg = chapters_arg.strip().lower()

    if chapters_arg == 'auto':
        return None  # Signal auto-detection

    result = set()
    parts = chapters_arg.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range: "1-5"
            try:
                start, end = part.split('-', 1)
                start = int(start.strip())
                end = int(end.strip())
                for ch in range(start, end + 1):
                    if 1 <= ch <= total_chapters:
                        result.add(ch)
            except ValueError:
                print(f"[ASSEMBLE] Warning: Invalid chapter range '{part}'")
        else:
            # Single chapter: "5"
            try:
                ch = int(part)
                if 1 <= ch <= total_chapters:
                    result.add(ch)
            except ValueError:
                print(f"[ASSEMBLE] Warning: Invalid chapter number '{part}'")

    return sorted(result) if result else list(range(1, total_chapters + 1))


def detect_completed_chapters(session: dict, state: dict) -> list[int]:
    """
    Auto-detect which chapters have all their sentence audio files completed.
    Returns a list of completed chapter numbers (1-indexed).
    """
    sentences_dir = session.get('sentences_dir') or os.path.join(session['process_dir'], 'chapters', 'sentences')
    chapter_sentences = state.get('chapter_sentences', [])

    completed = []
    sentence_offset = 0

    for i, chapter in enumerate(chapter_sentences):
        chapter_num = i + 1
        chapter_complete = True

        # Check if all sentence files for this chapter exist
        for j in range(len(chapter)):
            sentence_idx = sentence_offset + j
            sentence_file = os.path.join(sentences_dir, f'{sentence_idx}.{default_audio_proc_format}')
            if not os.path.exists(sentence_file):
                chapter_complete = False
                break

        if chapter_complete:
            completed.append(chapter_num)
        else:
            # Once we hit an incomplete chapter, stop (chapters are sequential)
            break

        sentence_offset += len(chapter)

    return completed


def assemble_audiobook(args: dict) -> dict:
    """
    Assemble sentence audio files into the final audiobook.
    """
    try:
        global context, context_tracker

        session_id = args.get('session')
        if not session_id:
            return {'success': False, 'error': 'No session ID provided'}

        # Setup session from saved state
        # Use explicit --session_dir if provided, otherwise default to tmp_dir
        session_dir = args.get('session_dir') or os.path.join(tmp_dir, f"ebook-{session_id}")
        if not os.path.exists(session_dir):
            return {'success': False, 'error': f"Session directory not found: {session_dir}"}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': 'No session-state.json found'}

        # Get or create session
        session = context.get_session(session_id)
        if not session or not session.get('id'):
            session = context.set_session(session_id)

        # Populate from state
        session['session_dir'] = state['session_dir']
        session['process_dir'] = state['process_dir']
        # Always derive directories from corrected process_dir
        # (session-state.json may contain stale paths from a different machine or WSL)
        session['chapters_dir'] = os.path.join(state['process_dir'], 'chapters')
        session['sentences_dir'] = os.path.join(state['process_dir'], 'chapters', 'sentences')
        session['language'] = state.get('language', default_language_code)
        session['output_format'] = args.get('output_format') or state.get('output_format', default_output_format)
        session['audiobooks_dir'] = args.get('audiobooks_dir') or state.get('audiobooks_dir', audiobooks_cli_dir)
        session['cancellation_requested'] = False
        session['script_mode'] = NATIVE
        session['is_gui_process'] = False
        session['output_channel'] = args.get('output_channel', default_output_channel)
        session['output_split'] = args.get('output_split', default_output_split)
        session['output_split_hours'] = args.get('output_split_hours', default_output_split_hours)

        # Language ISO
        try:
            if len(session['language']) in (2, 3):
                lang_dict = Lang(session['language'])
                if lang_dict:
                    session['language'] = lang_dict.pt3
                    session['language_iso1'] = lang_dict.pt1
            if not session.get('language_iso1'):
                session['language_iso1'] = session['language'][:2] if len(session['language']) >= 2 else 'en'
        except Exception:
            session['language_iso1'] = 'en'

        # Load chapter data
        if state.get('chapter_sentences'):
            print('[ASSEMBLE] Loading chapter data from session-state.json')
            session['chapters'] = state['chapter_sentences']
            session['metadata'] = state.get('metadata', {})
            session['filename_noext'] = state.get('filename_noext')
            # Re-derive cover path from corrected process_dir
            # (session-state.json may contain stale WSL path)
            raw_cover = state.get('cover')
            if isinstance(raw_cover, str) and session.get('filename_noext'):
                session['cover'] = os.path.join(state['process_dir'], session['filename_noext'] + '.jpg')
            else:
                session['cover'] = raw_cover
            # Load TOC chapter titles for proper chapter markers
            session['chapter_titles'] = state.get('chapter_titles', [])
            if session['chapter_titles']:
                print(f"[ASSEMBLE] Loaded {len(session['chapter_titles'])} chapter titles from TOC")
            # Chapter provenance — required to bind titles to chapters by identity.
            session['chapter_docs'] = list(state.get('chapter_docs', []))
            session['chapter_titles_by_doc'] = dict(state.get('chapter_titles_by_doc', {}))
            if len(session['chapter_docs']) == len(session['chapters']):
                print(f"[ASSEMBLE] Loaded chapter provenance for {len(session['chapter_docs'])} chapters")
            else:
                print(
                    f"[ASSEMBLE] session-state.json has no usable chapter provenance "
                    f"({len(session['chapter_docs'])} chapter_docs for {len(session['chapters'])} chapters); "
                    f"chapter markers will use each chapter's own first sentence"
                )
        else:
            return {'success': False, 'error': 'No chapter_sentences in session state'}

        if not session['metadata'].get('title'):
            session['metadata']['title'] = session.get('filename_noext', 'audiobook').replace('_', ' ')

        # Apply any command-line metadata overrides
        if args.get('title'):
            session['metadata']['title'] = args['title']
        if args.get('author'):
            session['metadata']['creator'] = args['author']

        # Check for bookforge_metadata in session state (has author, year, etc.)
        if state.get('bookforge_metadata'):
            bf_meta = state['bookforge_metadata']
            if bf_meta.get('title') and not args.get('title'):
                session['metadata']['title'] = bf_meta['title']
            if bf_meta.get('author') and not args.get('author'):
                session['metadata']['creator'] = bf_meta['author']
            if bf_meta.get('year'):
                session['metadata']['year'] = bf_meta['year']

        # Extract year from 'published' field if not already set (format: "YYYY-MM-DD...")
        if not session['metadata'].get('year') and session['metadata'].get('published'):
            published = session['metadata']['published']
            if isinstance(published, str) and len(published) >= 4:
                session['metadata']['year'] = published[:4]

        # Set final_name: use --output_filename if provided, otherwise format with author/year
        if args.get('output_filename'):
            # Use explicit filename (without extension, we'll add it)
            base_name = args['output_filename']
            if base_name.lower().endswith('.' + session['output_format']):
                base_name = base_name[:-len(session['output_format']) - 1]
            session['final_name'] = get_sanitized(base_name + '.' + session['output_format'])
        else:
            # Build filename: "Title. Author. (Year).ext" or just "Title.ext"
            title = session['metadata'].get('title', 'Untitled')
            author = session['metadata'].get('creator', '')
            year = session['metadata'].get('year', '')

            if author and year:
                filename_base = f"{title}. {author}. ({year})"
            elif author:
                filename_base = f"{title}. {author}"
            else:
                filename_base = title

            session['final_name'] = get_sanitized(filename_base + '.' + session['output_format'])

        # Parse --chapters argument for partial assembly
        total_chapters = len(session['chapters'])
        chapters_arg = args.get('chapters')
        selected_chapters = parse_chapters_arg(chapters_arg, total_chapters)

        # Auto-detect completed chapters if requested
        if selected_chapters is None:
            print("[ASSEMBLE] Auto-detecting completed chapters...")
            selected_chapters = detect_completed_chapters(session, state)
            if not selected_chapters:
                return {'success': False, 'error': 'No completed chapters found. TTS may still be in progress.'}
            print(f"[ASSEMBLE] Found {len(selected_chapters)} completed chapters: {selected_chapters[0]}-{selected_chapters[-1]}")

        # Check if this is a partial assembly
        is_partial = len(selected_chapters) < total_chapters
        if is_partial:
            print(f"[ASSEMBLE] Partial assembly: chapters {selected_chapters[0]}-{selected_chapters[-1]} of {total_chapters}")
            # Update filename to indicate partial
            if not args.get('output_filename'):
                base_name = session['final_name'].rsplit('.', 1)[0]
                ext = session['output_format']
                partial_suffix = f" (Partial Ch {selected_chapters[0]}-{selected_chapters[-1]})"
                session['final_name'] = get_sanitized(base_name + partial_suffix + '.' + ext)
        else:
            print(f"[ASSEMBLE] Assembling all {total_chapters} chapters...")

        # Store selected chapters for combine_audio_chapters
        session['selected_chapters'] = selected_chapters

        # Combine sentences into chapters (only for selected chapters)
        sentence_offset = 0
        for i, chapter in enumerate(session['chapters']):
            chapter_num = i + 1

            # Skip chapters not in selection
            if chapter_num not in selected_chapters:
                sentence_offset += len(chapter)
                continue

            chapter_filename = f'{chapter_num}.{default_audio_proc_format}'
            start_sentence = sentence_offset
            end_sentence = sentence_offset + len(chapter) - 1

            print(f"[ASSEMBLE] Chapter {chapter_num}: sentences {start_sentence}-{end_sentence}")

            if not combine_audio_sentences(session_id, chapter_filename, start_sentence, end_sentence):
                return {'success': False, 'error': f'Failed to combine sentences for chapter {chapter_num}'}

            sentence_offset += len(chapter)

        # Build VTT subtitle file (only for selected chapters)
        print("[ASSEMBLE] Creating VTT subtitle file...")
        selected_chapter_data = [session['chapters'][i-1] for i in selected_chapters]
        if not build_vtt_file(session_id, selected_chapter_data):
            print("[ASSEMBLE] Warning: VTT file creation failed (continuing without subtitles)")

        # Combine chapters into final audiobook
        print("[ASSEMBLE] Combining chapters into final audiobook...")
        exported_files = combine_audio_chapters(session_id)

        if exported_files and len(exported_files) > 0:
            return {
                'success': True,
                'session_id': session_id,
                'output_files': exported_files,
                'output_dir': session['audiobooks_dir']
            }
        else:
            return {'success': False, 'error': 'combine_audio_chapters() failed'}

    except Exception as e:
        print(f'assemble_audiobook() Exception: {e}')
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


def worker_only(args: dict) -> dict:
    """
    Run TTS conversion for an assigned sentence range only.
    This is Phase 2 of the parallel processing architecture.

    Note: For optimal memory usage, use worker.py instead of this function.
    This function still loads all core.py dependencies.
    """
    try:
        global context, context_tracker

        session_id = args.get('session')
        if not session_id:
            return {'success': False, 'error': 'No session ID provided'}

        sentence_start = args.get('sentence_start')
        sentence_end = args.get('sentence_end')
        chapter_start = args.get('chapter_start')
        chapter_end = args.get('chapter_end')

        sentence_mode = sentence_start is not None and sentence_end is not None
        chapter_mode = chapter_start is not None and chapter_end is not None

        if not sentence_mode and not chapter_mode:
            return {'success': False, 'error': 'Must specify sentence or chapter range'}

        # Load session state
        session_dir = os.path.join(tmp_dir, f"ebook-{session_id}")
        if not os.path.exists(session_dir):
            return {'success': False, 'error': f"Session directory not found: {session_dir}"}

        state = load_session_state(session_dir)
        if not state:
            return {'success': False, 'error': 'No session-state.json found'}

        if 'chapter_sentences' not in state or not state['chapter_sentences']:
            return {'success': False, 'error': 'session-state.json missing chapter_sentences'}

        print(f"[WORKER] Loaded: {state['total_chapters']} chapters, {state['total_sentences']} sentences")

        # Get or create session
        session = context.get_session(session_id)
        if not session or not session.get('id'):
            session = context.set_session(session_id)

        # Populate session from state
        session['session_dir'] = state['session_dir']
        session['process_dir'] = state['process_dir']
        # Always derive directories from corrected process_dir
        # (session-state.json may contain stale paths from a different machine or WSL)
        session['chapters_dir'] = os.path.join(state['process_dir'], 'chapters')
        session['sentences_dir'] = os.path.join(state['process_dir'], 'chapters', 'sentences')
        session['epub_path'] = state.get('epub_path_internal')
        session['filename_noext'] = state.get('filename_noext')
        # Re-derive cover path from corrected process_dir
        # (session-state.json may contain stale WSL path)
        raw_cover = state.get('cover')
        if isinstance(raw_cover, str) and session.get('filename_noext'):
            session['cover'] = os.path.join(state['process_dir'], session['filename_noext'] + '.jpg')
        else:
            session['cover'] = raw_cover
        session['language'] = state.get('language', default_language_code)
        session['language_iso1'] = state.get('language_iso1', 'en')
        session['tts_engine'] = args.get('tts_engine') or state.get('tts_engine') or get_compatible_tts_engines(session['language'])[0]
        session['fine_tuned'] = args.get('fine_tuned') or state.get('fine_tuned') or default_fine_tuned
        session['voice'] = args.get('voice') or state.get('voice')
        session['output_format'] = args.get('output_format') or state.get('output_format', default_output_format)
        session['audiobooks_dir'] = args.get('audiobooks_dir') or state.get('audiobooks_dir', audiobooks_cli_dir)
        session['cancellation_requested'] = False
        session['final_name'] = state.get('final_name')
        session['chapters'] = state['chapter_sentences']
        session['script_mode'] = NATIVE
        session['is_gui_process'] = False
        session['model_cache'] = f"{session['tts_engine']}-{session['fine_tuned']}"
        session['custom_model_dir'] = state.get('custom_model_dir') or os.path.join(models_dir, '__sessions', f"model-{session_id}")
        session['custom_model'] = state.get('custom_model')

        # Device setup
        session['device'] = args.get('device') or state.get('device', default_device)
        if session['device'] == 'cuda':
            import torch
            session['device'] = session['device'] if torch.cuda.is_available() else 'cpu'
        elif session['device'] == 'mps':
            import torch
            session['device'] = session['device'] if torch.backends.mps.is_available() else 'cpu'

        # VRAM detection
        vram_dict = VRAMDetector().detect_vram(session['device'], session['script_mode'])
        session['free_vram_gb'] = vram_dict.get('free_vram_gb', 4)

        # Calculate work range for chapter mode
        if chapter_mode:
            sentence_offset = 0
            for i, chapter in enumerate(state['chapter_sentences']):
                chapter_num = i + 1
                if chapter_num == chapter_start:
                    sentence_start = sentence_offset
                if chapter_num == chapter_end:
                    sentence_end = sentence_offset + len(chapter) - 1
                    break
                sentence_offset += len(chapter)
            print(f"[WORKER] Chapters {chapter_start}-{chapter_end} = sentences {sentence_start}-{sentence_end}")

        # Set worker mode flags
        session['worker_mode'] = True
        session['sentence_start'] = sentence_start
        session['sentence_end'] = sentence_end

        print(f"[WORKER] Processing sentences {sentence_start}-{sentence_end} on {session['device'].upper()}")

        # Run TTS conversion
        if convert_chapters2audio(session_id):
            return {
                'success': True,
                'session_id': session_id,
                'sentence_start': sentence_start,
                'sentence_end': sentence_end
            }
        else:
            return {'success': False, 'error': 'convert_chapters2audio() failed'}

    except Exception as e:
        print(f'worker_only() Exception: {e}')
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}


# =============================================================================
# BookForge Extension Hooks
# =============================================================================
# Register hook accessors for the BookForge extension.
# This allows the extension to access core.py internals (context, etc.)
# without modifying the extension when core.py internals change.

try:
    from bookforge_ext import hooks as bf_hooks
    bf_hooks.register('get_context', lambda: context)
    bf_hooks.register('get_context_tracker', lambda: context_tracker)
    bf_hooks.register('get_active_sessions', lambda: active_sessions)
except ImportError:
    pass  # Extension not installed