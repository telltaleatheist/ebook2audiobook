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
import re
import sys

# The fixture covers the four shapes headers actually come in:
#   h1  a short, UNPUNCTUATED, all-caps header
#   h2  a normal 'Chapter 8: State of Confusion' header
#   h3  a mid-chapter section header sitting between two paragraphs
#   h3 -> 'No.'  a header directly followed by a one-word paragraph (the pair
#         that the 25-char min-chars floor used to fuse into one chunk)
#   h2  a DECORATED header — '• Silo 1 •', the shape from Hugh Howey's "Shift".
#         conf_lang maps the bullet to a period, so this header reaches the
#         splitter as '. Silo 1 ..' and PASS 1 breaks it at that first period,
#         leaving a heading row whose whole text is '.' — a chunk with nothing
#         to speak, which is what the model improvised 1.6s of noise for
#         (2026-08-29, [ORPHEUS][SHORT_CHUNK_OVERRUN] … text='.').
FIXTURE_XHTML = """<body>
<h1>PROLOGUE</h1>
<p>The city had been quiet for a very long time before that morning came.</p>
<h2>Chapter 8: State of Confusion</h2>
<p>Some consider Franklin D. Roosevelt to have been the greatest of them all.</p>
<h3>A Section Within</h3>
<p>No.</p>
<p>The argument continued for another hour without anyone changing their mind.</p>
<h2>&#8226; Silo 1 &#8226;</h2>
<p>Troy needed to see a doctor, and had needed to for very much longer than he cared to admit to anybody at all.</p>
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


def _check_floor_wordless_rows(core):
    """The min-chars floor, driven DIRECTLY over row sequences the fixture markup
    cannot reliably produce (2026-08-29).

    The fixture proves the real '• Silo 1 •' path end to end; this proves the
    rule itself at the layer that owns it, including the two shapes that only
    ever arise from a specific neighbour arrangement. Returns a list of failures.

    A row is built the way get_sentences sees one: escape_sml has already turned
    each SML block into ONE char whose index into sml_blocks is its identity, so
    chr(sml_escape_tag + i) IS the token here."""
    heading, brk = chr(core.sml_escape_tag), chr(core.sml_escape_tag + 1)
    is_heading = core._heading_row_test(['[heading]', '[break]'])
    # No [item] in this block table, so the item predicate answers False for
    # every row here — these cases are about headings only (see
    # tools/test_list_item_chunks.py for the item exemption).
    is_item = core._marker_row_test(['[heading]', '[break]'], 'item')

    def clean_len(s):
        return len(core._strip_escaped_sml(s))

    def has_words(s):
        return core._has_word_chars(s)

    def run(rows, min_chars=25, max_chars=350):
        return core._apply_min_chars_floor(rows, clean_len, max_chars, min_chars, is_heading, is_item, has_words)

    failures = []

    def expect(label, rows, want):
        got = run(*rows) if isinstance(rows, tuple) else run(rows)
        print(f'  {label}\n      in : {[core._strip_escaped_sml(r) for r in (rows[0] if isinstance(rows, tuple) else rows)]!r}'
              f'\n      out: {[core._strip_escaped_sml(r) for r in got]!r}')
        if got != want:
            failures.append(f'{label}: got {got!r}, want {want!r}')

    # 1. A bare '.' NON-heading row wedged between two headings. Both floor
    #    merges are refused — a heading is not a landing site in either
    #    direction — so this is the arrangement where the fall-through used to
    #    SHIP the row. The '.' must be DROPPED; the two short headings then
    #    coalesce forward (2026-08-29), which is case 3c's rule at work.
    expect('bare "." between two headings is dropped',
           [f'{heading}Chapter One.', f'{brk}.', f'{heading}Chapter Two.'],
           [f'{heading}Chapter One. Chapter Two.'])

    # 2. A HEADING whose whole text is punctuation — what '• Silo 1 •' leaves
    #    behind. The heading exemption must not rescue it.
    expect('a heading with no word in it is dropped, exemption or not',
           [f'{brk}{heading}.', 'Troy needed to see a doctor and had for a long while.'],
           ['Troy needed to see a doctor and had for a long while.'])

    # 3. THE EXEMPTION IS FOR HEADINGS OF 3+ WORDS (2026-08-29). 'II.' has a
    #    word character so it is not dropped — but at one word Orpheus may not
    #    voice it at all, so it merges FORWARD into its first paragraph, demoted
    #    to plain text (its marker rides the dropped lead).
    expect('a 1-word heading ("II.") merges forward into its paragraph',
           [f'{heading}II.', 'The argument continued for another hour without anyone changing their mind.'],
           ['II. The argument continued for another hour without anyone changing their mind.'])

    # 3b. A 3-word heading KEEPS its isolation — the narrowed exemption's floor.
    rows = [f'{heading}A Section Within.', 'The argument continued for another hour without anyone changing their mind.']
    expect('a 3-word heading still stands alone', rows, list(rows))

    # 3c. STACKED short headings coalesce: each merges forward into the next,
    #    the target's marker surviving each time, until the combined title
    #    reaches 3 words and stands as ONE heading. The body row is untouched.
    expect('stacked 1-word headings coalesce into one heading',
           [f'{heading}16.', f'{brk}{heading}2110.', f'{brk}{heading}Silo one.',
            'Troy needed to see a doctor and had for a very long while.'],
           [f'{brk}{heading}16. 2110. Silo one.',
            'Troy needed to see a doctor and had for a very long while.'])

    # 3d. A short heading with NOTHING after it in the chapter stays isolated
    #    (logged) — the render-side re-roll backstop is its guard.
    rows = ['A sentence long enough to clear the floor entirely on its own.', f'{brk}{heading}II.']
    expect('a chapter-final short heading is kept', rows, list(rows))

    # 3e. SENTENCE_MIN_CHARS=0 turns off the LENGTH floor, not the voicing rule:
    #    a 1-word heading still merges forward.
    expect('the short-heading merge holds with the length floor disabled',
           ([f'{heading}II.', 'Some prose that is comfortably long enough on its own.'], 0),
           ['II. Some prose that is comfortably long enough on its own.'])

    # 4. An SML-ONLY row is a PAUSE, not a wordless chunk: the engines never send
    #    it to the model (orpheus.py writes silence for it), so it must survive.
    rows = ['A sentence long enough to clear the floor entirely on its own.', brk,
            'Another sentence that also clears the floor without merging.']
    expect('a bare [break] row is a pause and survives', rows, list(rows))

    # 5. SENTENCE_MIN_CHARS=0 turns the LENGTH floor off. It does not licence a
    #    chunk with nothing to say.
    expect('the wordless rule holds with the length floor disabled',
           ([f'{brk}.', 'Some prose that is comfortably long enough.'], 0),
           ['Some prose that is comfortably long enough.'])

    return failures


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

    print('\n--- min-chars floor: no chunk without a word in it ---')
    failures.extend(_check_floor_wordless_rows(core))

    for engine in ENGINES:
        print(f'\n===== {engine} =====')
        chunks = _run_chapter(core, engine, skip_headings=False)
        if chunks is None:
            failures.append(f'{engine}: filter_chapter returned None')
            continue
        for i, c in enumerate(chunks):
            print(f'  [{i}] {c!r}')

        # 1. a header of 3+ words is a chunk of its own; a shorter one is merged
        #    FORWARD (2026-08-29) — it must open a chunk that continues with the
        #    text under it, never stand alone and never go missing.
        spoken = [_spoken_text(core, engine, c) for c in chunks]
        for header in EXPECTED_HEADERS[engine]:
            if core._word_count(header) >= 3:
                if header not in spoken:
                    owner = next((s for s in spoken if header in s), None)
                    failures.append(
                        f'{engine}: header {header!r} is not its own chunk'
                        + (f' — it was merged into {owner!r}' if owner else ' — it is missing entirely')
                    )
            else:
                if header in spoken:
                    failures.append(f'{engine}: short header {header!r} was left as its own chunk')
                elif not any(s.startswith(header + ' ') for s in spoken):
                    failures.append(f'{engine}: short header {header!r} does not open the chunk under it')

        # 1b. NOTHING IS HANDED TO THE MODEL WITH NO WORD IN IT (2026-08-29).
        #     The '• Silo 1 •' header is the fixture's carrier: its bullets
        #     become periods, PASS 1 splits at the first one, and the leading
        #     '.' used to ship as a heading chunk of its own. An SML-only row is
        #     NOT this case — it never reaches the model (silence is written for
        #     it) — so only chunks with surviving text are judged.
        for c, s in zip(chunks, spoken):
            if s and not re.search(r'\w', s):
                failures.append(f'{engine}: chunk has nothing to speak: {c!r} (spoken {s!r})')

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
