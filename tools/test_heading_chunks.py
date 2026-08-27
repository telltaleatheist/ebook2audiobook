#!/usr/bin/env python
"""Headers and titles must be read as their OWN TTS chunk (2026-08-27).

Drives the REAL pipeline — filter_chapter() on fixture XHTML, which walks the
markup, marks each h1-h6 with the [heading] token (TTS_SML['heading']) and runs
the whole of get_sentences over it: PASS 1-3 splitting, PASS 4, the Orpheus
PASS 5 packer, the Voxtral packer and the min-chars floor. Nothing is stubbed
except the epub document object, which filter_chapter only ever asks for
get_body_content().

What it proves, for the default (XTTS-class), orpheus and voxtral paths:
  1. every header comes out as a chunk of its own, exactly the header text;
  2. nothing an engine will SPEAK still contains the marker — checked through
     each engine's own strip path, not a re-implementation of it;
  3. skip_headings still drops headers entirely.

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_heading_chunks.py
Exit code 0 = all cases passed.
"""

import os
import sys

# The fixture covers the four shapes headers actually come in:
#   h1  a short, UNPUNCTUATED, all-caps header
#   h2  a normal 'Chapter 8: State of Confusion' header
#   h3  a mid-chapter section header sitting between two paragraphs
#   h3 -> 'No.'  a header directly followed by a one-word paragraph (the pair
#         that the 25-char min-chars floor used to fuse into one chunk)
FIXTURE_XHTML = """<body>
<h1>PROLOGUE</h1>
<p>The city had been quiet for a very long time before that morning came.</p>
<h2>Chapter 8: State of Confusion</h2>
<p>Some consider Franklin D. Roosevelt to have been the greatest of them all.</p>
<h3>A Section Within</h3>
<p>No.</p>
<p>The argument continued for another hour without anyone changing their mind.</p>
</body>"""

# What filter_chapter should emit as heading rows: the header text plus the
# terminal period it forces so TTS stops there.
#
# Per engine, because '8' is not a per-engine detail of the HEADING — it is the
# number pipeline. Orpheus keeps book-exact text and expands nothing until the
# engine boundary, so it sees 'Chapter 8'; every other engine runs math2words
# before splitting and sees 'Chapter eight'. Same row either way.
EXPECTED_HEADERS = {
    'xtts': ['PROLOGUE.', 'Chapter eight: State of Confusion.', 'A Section Within.'],
    'orpheus': ['PROLOGUE.', 'Chapter 8: State of Confusion.', 'A Section Within.'],
    'voxtral': ['PROLOGUE.', 'Chapter eight: State of Confusion.', 'A Section Within.'],
}

ENGINES = ['xtts', 'orpheus', 'voxtral']


class _FixtureDoc:
    """The only thing filter_chapter asks of an epub document."""

    def __init__(self, html):
        self._html = html

    def get_body_content(self):
        return self._html


def _run_chapter(core, engine, skip_headings):
    session_id = f'heading-test-{engine}-{int(skip_headings)}'
    session = core.context.set_session(session_id)
    session['tts_engine'] = engine
    session['language'] = 'eng'
    session['language_iso1'] = 'en'
    session['is_gui_process'] = False
    session['skip_headings'] = skip_headings
    # stanza_nlp=False is the real 'no NER pipeline loaded' value core passes;
    # it only gates date expansion, which no fixture line needs.
    return core.filter_chapter(0, _FixtureDoc(FIXTURE_XHTML), session_id, False, True)


def _spoken_text(core, engine, chunk):
    """The text this engine will actually hand its model, produced by the
    engine's OWN stripping code — SML_UNSPOKEN_PATTERN for orpheus/voxtral/f5,
    and the split-and-convert loop for the XTTS-class engines."""
    from lib.conf_models import SML_TAG_PATTERN, SML_UNSPOKEN_PATTERN
    if engine in ('orpheus', 'voxtral'):
        return SML_UNSPOKEN_PATTERN.sub('', chunk).strip()
    # XTTS-class: every fragment that fullmatches an SML tag goes to
    # _convert_sml and is never spoken; the rest is.
    spoken = []
    for part in _split_on_sml(SML_TAG_PATTERN, chunk):
        if SML_TAG_PATTERN.fullmatch(part):
            continue
        spoken.append(part)
    return ''.join(spoken).strip()


def _split_on_sml(pattern, sentence):
    """Same split the XTTS-class engines do (utils.py _split_sentence_on_sml)."""
    parts, last = [], 0
    for m in pattern.finditer(sentence):
        start, end = m.span()
        if start > last:
            parts.append(sentence[last:start])
        parts.append(m.group(0))
        last = end
    if last < len(sentence):
        parts.append(sentence[last:])
    return parts


def _check_convert_sml_accepts_heading():
    """The XTTS-class engines route every SML fragment through _convert_sml,
    which returns (False, error) for a tag it does not know — that would abort
    the whole sentence. Prove it accepts [heading], and appends no audio."""
    from lib.classes.tts_engines.common.utils import TTSUtils
    from lib.conf_models import TTS_SML
    inst = TTSUtils.__new__(TTSUtils)
    inst.audio_segments = []
    ok, error = inst._convert_sml(TTS_SML['heading']['static'])
    if not ok:
        return False, f'_convert_sml rejected [heading]: {error}'
    if inst.audio_segments:
        return False, '_convert_sml appended audio for [heading] (it must be silent markup)'
    return True, ''


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    os.chdir(repo)
    import lib.core as core
    print(f'TESTING TREE: {os.path.dirname(os.path.abspath(core.__file__))}')
    core.context = core.SessionContext()

    failures = []

    ok, error = _check_convert_sml_accepts_heading()
    print(f'\n--- xtts-class _convert_sml([heading]) --- {"OK" if ok else "FAIL"}')
    if not ok:
        failures.append(error)

    for engine in ENGINES:
        print(f'\n===== {engine} =====')
        chunks = _run_chapter(core, engine, skip_headings=False)
        if chunks is None:
            failures.append(f'{engine}: filter_chapter returned None')
            continue
        for i, c in enumerate(chunks):
            print(f'  [{i}] {c!r}')

        # 1. every header is a chunk of its own — the chunk's spoken text is the
        #    header and nothing else.
        spoken = [_spoken_text(core, engine, c) for c in chunks]
        for header in EXPECTED_HEADERS[engine]:
            if header not in spoken:
                owner = next((s for s in spoken if header in s), None)
                failures.append(
                    f'{engine}: header {header!r} is not its own chunk'
                    + (f' — it was merged into {owner!r}' if owner else ' — it is missing entirely')
                )

        # 2. nothing spoken still carries the marker, in tag form or escaped.
        for s in spoken:
            if '[heading]' in s or '[/heading]' in s:
                failures.append(f'{engine}: marker survived into spoken text: {s!r}')
            if any(ord(ch) >= core.sml_escape_tag for ch in s):
                failures.append(f'{engine}: an escaped SML char survived into spoken text: {s!r}')

        # 3. skip_headings still means skipped.
        skipped = _run_chapter(core, engine, skip_headings=True)
        if skipped is None:
            failures.append(f'{engine}: filter_chapter returned None with skip_headings')
            continue
        print(f'  -- skip_headings=True --')
        for i, c in enumerate(skipped):
            print(f'  [{i}] {c!r}')
        joined = ' '.join(skipped)
        for header in EXPECTED_HEADERS[engine]:
            # The h1/h2/h3 text must be gone; a trailing period is the only part
            # of it the fixture's prose could legitimately share.
            if header.rstrip('.') in joined:
                failures.append(f'{engine}: skip_headings did not drop {header!r}')
        if '[heading]' in joined:
            failures.append(f'{engine}: skip_headings left a heading marker behind')

    print('\n==================== RESULT ====================')
    if failures:
        for f in failures:
            print(f'FAIL: {f}')
        print(f'{len(failures)} failure(s)')
        return 1
    print('all heading-chunk cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
