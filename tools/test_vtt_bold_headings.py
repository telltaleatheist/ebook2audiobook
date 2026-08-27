#!/usr/bin/env python
"""A heading's VTT cue is written BOLD (2026-08-27).

Headers already come out as their own chunk (tools/test_heading_chunks.py); this
harness covers what the TRANSCRIPT then does with them. A heading cue's payload
is WebVTT's own <b>…</b> — the portable spelling, understood by every WebVTT
reader, and converted by ffmpeg into a real tx3g 'styl' record carrying the bold
face-style flag when the transcript is muxed into the m4b.

It drives the REAL pipeline for the rows — filter_chapter() over fixture XHTML,
the same fixture the heading-chunk harness uses, so the sentences tested here
are the ones get_sentences actually emits, markers and all — and then the REAL
shared builder, lib.conf_models.vtt_cue_text, which is the ONE place all three
VTT builders now get a cue's text from.

What it proves:
  1. every header's cue payload is exactly '<b>Header text.</b>';
  2. nothing else is bolded, and no cue carries a stray tag;
  3. no cue carries the [heading] marker or an escaped SML char;
  4. a row with no heading marker produces the EXACT bytes the builders emitted
     before this change — old transcripts are untouched;
  5. all three VTT builders really do route through vtt_cue_text, so the two
     that once drifted apart cannot drift again.

Both strip-pattern families are exercised, because the builders legitimately
differ: the two build_vtt_file copies pass SML_UNSPOKEN_PATTERN, while
TTSUtils._build_vtt_file passes SML_TAG_PATTERN.

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_vtt_bold_headings.py
Exit code 0 = all cases passed.
"""

import os
import re
import sys

FIXTURE_XHTML = """<body>
<h1>PROLOGUE</h1>
<p>The city had been quiet for a very long time before that morning came.</p>
<h2>Chapter 8: State of Confusion</h2>
<p>Some consider Franklin D. Roosevelt to have been the greatest of them all.</p>
<h3>A Section Within</h3>
<p>No.</p>
<p>The argument continued for another hour without anyone changing their mind.</p>
</body>"""

# Same per-engine split as the heading-chunk harness: Orpheus keeps book-exact
# text ('Chapter 8'), every other engine runs math2words first ('Chapter eight').
EXPECTED_BOLD = {
    'xtts': ['<b>PROLOGUE.</b>', '<b>Chapter eight: State of Confusion.</b>', '<b>A Section Within.</b>'],
    'orpheus': ['<b>PROLOGUE.</b>', '<b>Chapter 8: State of Confusion.</b>', '<b>A Section Within.</b>'],
    'voxtral': ['<b>PROLOGUE.</b>', '<b>Chapter eight: State of Confusion.</b>', '<b>A Section Within.</b>'],
}

ENGINES = ['xtts', 'orpheus', 'voxtral']

# The three builders that must all route through vtt_cue_text.
BUILDERS = [
    ('bookforge_ext/parallel/session.py', 'def build_vtt_file'),
    ('lib/core.py', 'def build_vtt_file'),
    ('lib/classes/tts_engines/common/utils.py', 'def _build_vtt_file'),
]


class _FixtureDoc:
    """The only thing filter_chapter asks of an epub document."""

    def __init__(self, html):
        self._html = html

    def get_body_content(self):
        return self._html


def _run_chapter(core, engine):
    session_id = f'vtt-bold-{engine}'
    session = core.context.set_session(session_id)
    session['tts_engine'] = engine
    session['language'] = 'eng'
    session['language_iso1'] = 'en'
    session['is_gui_process'] = False
    session['skip_headings'] = False
    return core.filter_chapter(0, _FixtureDoc(FIXTURE_XHTML), session_id, False, True)


def _legacy_cue_text(sentence, strip_pattern):
    """What the builders emitted BEFORE this change, character for character."""
    return re.sub(r'\s+', ' ', strip_pattern.sub('', str(sentence))).strip()


def _function_body(repo, rel_path, signature):
    """The source lines of one function, without importing its module."""
    with open(os.path.join(repo, rel_path), 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip().startswith(signature)), None)
    if start is None:
        return None
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = [lines[start]]
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break
        body.append(line)
    return '\n'.join(body)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    os.chdir(repo)
    import lib.core as core
    from lib.conf_models import SML_TAG_PATTERN, SML_UNSPOKEN_PATTERN, vtt_cue_text
    print(f'TESTING TREE: {os.path.dirname(os.path.abspath(core.__file__))}')
    core.context = core.SessionContext()

    failures = []

    # 5. the three builders all route through the shared cue builder.
    print('\n===== builders route through vtt_cue_text =====')
    for rel_path, signature in BUILDERS:
        body = _function_body(repo, rel_path, signature)
        if body is None:
            failures.append(f'{rel_path}: could not find {signature!r}')
            continue
        ok = 'vtt_cue_text(' in body
        print(f'  {"OK  " if ok else "FAIL"} {rel_path} {signature}')
        if not ok:
            failures.append(f'{rel_path}: {signature} does not call vtt_cue_text')

    families = [('SML_UNSPOKEN_PATTERN', SML_UNSPOKEN_PATTERN), ('SML_TAG_PATTERN', SML_TAG_PATTERN)]

    for engine in ENGINES:
        rows = _run_chapter(core, engine)
        if rows is None:
            failures.append(f'{engine}: filter_chapter returned None')
            continue

        for family_name, pattern in families:
            label = f'{engine} / {family_name}'
            print(f'\n===== {label} =====')
            cues = [vtt_cue_text(str(r), pattern) for r in rows]
            for i, c in enumerate(cues):
                print(f'  [{i}] {c!r}')

            bolded = [c for c in cues if c.startswith('<b>')]
            expected = EXPECTED_BOLD[engine]

            # 1. every header is a bold cue, exactly as written.
            for want in expected:
                if want not in cues:
                    failures.append(f'{label}: expected bold cue {want!r} not among the cues')

            # 2. nothing else is bolded, and no cue carries a stray tag.
            if len(bolded) != len(expected):
                failures.append(
                    f'{label}: {len(bolded)} bold cue(s), expected {len(expected)}: {bolded!r}'
                )
            for c in cues:
                if c.startswith('<b>'):
                    if not c.endswith('</b>'):
                        failures.append(f'{label}: unbalanced bold cue {c!r}')
                    inner = c[3:-4]
                    if '<' in inner or '>' in inner:
                        failures.append(f'{label}: a tag leaked inside a bold cue: {c!r}')
                elif '<' in c or '>' in c:
                    failures.append(f'{label}: a non-heading cue carries a tag: {c!r}')
                if c == '<b></b>':
                    failures.append(f'{label}: an empty row was bolded into {c!r}')

            # 3. no marker, no escape char, ever reaches a cue.
            for c in cues:
                if '[heading]' in c or '[/heading]' in c:
                    failures.append(f'{label}: the marker survived into a cue: {c!r}')
                if any(ord(ch) >= core.sml_escape_tag for ch in c):
                    failures.append(f'{label}: an escaped SML char survived into a cue: {c!r}')

            # 4. a row with NO heading marker is byte-identical to the old output.
            for row in rows:
                row = str(row)
                if core.SML_HEADING_PATTERN.search(row):
                    continue
                got = vtt_cue_text(row, pattern)
                want = _legacy_cue_text(row, pattern)
                if got != want:
                    failures.append(
                        f'{label}: a non-heading row changed: {got!r} != legacy {want!r}'
                    )

    # Show one complete VTT, so the actual file content is on the record.
    print('\n===== the VTT an orpheus run now writes (timings elided) =====')
    rows = _run_chapter(core, 'orpheus')
    print('WEBVTT\n')
    for i, row in enumerate(rows):
        print(f'00:00:{i:02}.000 --> 00:00:{i + 1:02}.000')
        print(f'{vtt_cue_text(str(row), SML_UNSPOKEN_PATTERN)}\n')

    print('==================== RESULT ====================')
    if failures:
        for f in failures:
            print(f'FAIL: {f}')
        print(f'{len(failures)} failure(s)')
        return 1
    print('all bold-heading cue cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
