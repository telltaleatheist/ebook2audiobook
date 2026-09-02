#!/usr/bin/env python
"""The Orpheus engine reads its text AS PRINTED — no engine-side number
transform (Owen, 2026-09-02, permanently).

Number normalization is BookForge's job (a model pass over the narration copy,
run by the BookForge CLI's cleanup step before its TTS step), and "we don't need
the pass done in two places". This proves, through the real code:

  1. _clean_sentence_for_tts strips the SML tags and changes NOTHING else —
     digits, comma-grouped integers, dates and scripture refs come out exactly
     as they went in;
  2. the environment has no say: the ORPHEUS_TEXT_TRANSFORM variable that
     briefly existed on 2026-09-02 is not read (there is no switch — a knob
     would be a second place);
  3. the packer measures rows as printed: a row of digits packs by its printed
     length (proven through filter_chapter, the real prep path);
  4. to_tts_form itself still exists and still works — the ASR gate and the
     corpus builders' port depend on it — it is just never applied here.

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_engine_reads_as_printed.py
Exit code 0 = all cases passed.
"""

import os
import sys


def _clean(engine_cls, text):
    # _clean_sentence_for_tts reads nothing from self, so a bare instance is
    # enough — the same trick test_heading_chunks uses for _convert_sml.
    inst = engine_cls.__new__(engine_cls)
    return engine_cls._clean_sentence_for_tts(inst, text)


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    os.chdir(repo)

    failures = []

    def expect(label, got, want):
        ok = got == want
        print(f'  {"ok  " if ok else "FAIL"} {label}\n       got : {got!r}\n       want: {want!r}')
        if not ok:
            failures.append(label)

    from lib.classes.tts_engines.common import orpheus_text
    from lib.classes.tts_engines.orpheus import Orpheus

    sample = '[break][item]Member number 670,992 joined on October 15, 1944; see John 3:16.[pause]'
    printed = 'Member number 670,992 joined on October 15, 1944; see John 3:16.'

    # 1. As printed, SML stripped.
    for k in ('ORPHEUS_TEXT_TRANSFORM',):
        os.environ.pop(k, None)
    expect('engine: SML stripped, every digit and ref as printed', _clean(Orpheus, sample), printed)

    # 2. The environment has no say.
    for v in ('0', '1', 'anything'):
        os.environ['ORPHEUS_TEXT_TRANSFORM'] = v
        expect(f'engine: ORPHEUS_TEXT_TRANSFORM={v!r} changes nothing', _clean(Orpheus, sample), printed)
    os.environ.pop('ORPHEUS_TEXT_TRANSFORM', None)
    expect('module: no switch exists', hasattr(orpheus_text, 'text_transform_enabled'), False)

    # 4. The transform is still there for its remaining readers.
    expect('to_tts_form still expands (kept for the ASR gate / corpus port)',
           orpheus_text.to_tts_form('670,992 in 1944, John 3:16'),
           'six hundred seventy thousand nine hundred ninety two in nineteen forty-four, John three sixteen')

    # 3. The packer measures as printed. Eight four-digit years are 39 chars
    #    printed and ~190 as words: at a 120-char cap two such rows fit ONE
    #    chunk only if the packer measures printed length.
    import lib.core as core
    print(f'  TESTING TREE: {os.path.dirname(os.path.abspath(core.__file__))}')
    core.context = core.SessionContext()

    class _Doc:
        def __init__(self, html):
            self._html = html

        def get_body_content(self):
            return self._html

    years = ' '.join(['1933', '1934', '1935', '1936', '1937', '1938', '1939', '1940'])
    html = f'<body><p>Years {years} ended.</p><p>Years {years} again.</p></body>'
    os.environ['ORPHEUS_MAX_CHARS'] = '120'
    os.environ['SENTENCE_MIN_CHARS'] = '0'
    session_id = 'reads-as-printed'
    session = core.context.set_session(session_id)
    session['tts_engine'] = 'orpheus'
    session['language'] = 'eng'
    session['language_iso1'] = 'en'
    session['is_gui_process'] = False
    session['skip_headings'] = False
    rows = core.filter_chapter(0, _Doc(html), session_id, False, True)
    from lib.conf_models import SML_TAG_PATTERN
    spoken = [SML_TAG_PATTERN.sub('', r).strip() for r in rows]
    spoken = [r for r in spoken if r]
    for i, r in enumerate(spoken):
        print(f'    [{i}] {r!r}')
    # Both rows survive whole (no space-split into a starved 'ended.' tail),
    # which only happens when their length is measured as printed.
    expect('packer: no row was space-split — printed length is under the cap',
           all(r.endswith(('ended.', 'again.')) for r in spoken), True)
    expect('packer: every chunk within the 120-char cap as printed',
           [r for r in spoken if len(r) > 120], [])
    for k in ('ORPHEUS_MAX_CHARS', 'SENTENCE_MIN_CHARS'):
        os.environ.pop(k, None)

    print('\n==================== RESULT ====================')
    if failures:
        print(f'{len(failures)} failure(s):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('all reads-as-printed cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
