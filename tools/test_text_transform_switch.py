#!/usr/bin/env python
"""ORPHEUS_TEXT_TRANSFORM — the switch that says whether e2a applies its own
book-exact -> model-input transform (2026-09-02).

Number normalization moved upstream: BookForge runs a model pass over the
narration copy and hands e2a a file with no digits left to expand, and sets
ORPHEUS_TEXT_TRANSFORM=0 so the pass is not done in two places (Owen's ruling).
This proves the switch at the ONE engine site that applies the transform
(_clean_sentence_for_tts), through the real method rather than a re-statement
of it, plus the helper's own contract:

  1. unset / '1' -> the transform runs (digits become words, scripture refs
     become their spoken form) — the default, so every raw-text producer keeps
     the behaviour the fine-tunes trained on;
  2. '0'         -> the text is read AS PRINTED; SML tags are still stripped;
  3. any other value raises (NO FALLBACK);
  4. the packer's length measure in get_sentences follows the same switch
     (proven through filter_chapter: a digit-dense row packs differently when
     measured as printed vs as words).

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_text_transform_switch.py
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

    sample = '[break]Member number 670,992 joined on October 15, 1944; see John 3:16.'

    # 1. Default (unset) and explicit '1': the transform runs.
    os.environ.pop('ORPHEUS_TEXT_TRANSFORM', None)
    expect('helper: unset means ON', orpheus_text.text_transform_enabled(), True)
    on_unset = _clean(Orpheus, sample)
    expect('engine: unset expands digits and scripture', on_unset,
           orpheus_text.to_tts_form(sample.replace('[break]', '')).strip())
    if '670,992' in on_unset or '1944' in on_unset:
        failures.append('engine: digits survived with the transform ON')
    os.environ['ORPHEUS_TEXT_TRANSFORM'] = '1'
    expect("helper: '1' means ON", orpheus_text.text_transform_enabled(), True)
    expect("engine: '1' matches unset", _clean(Orpheus, sample), on_unset)

    # 2. '0': read as printed, SML still stripped.
    os.environ['ORPHEUS_TEXT_TRANSFORM'] = '0'
    expect("helper: '0' means OFF", orpheus_text.text_transform_enabled(), False)
    expect("engine: '0' reads the text as printed, SML stripped", _clean(Orpheus, sample),
           'Member number 670,992 joined on October 15, 1944; see John 3:16.')

    # 3. Anything else raises.
    for bad in ('2', 'off', 'true', ' yes '):
        os.environ['ORPHEUS_TEXT_TRANSFORM'] = bad
        try:
            orpheus_text.text_transform_enabled()
            expect(f'helper: {bad!r} raises', 'no error', 'ValueError')
        except ValueError as err:
            expect(f'helper: {bad!r} raises', 'ORPHEUS_TEXT_TRANSFORM' in str(err), True)

    # 4. The packer's measure follows the switch. A row of eight four-digit
    #    years is 8*4 chars printed but ~8*24 as words; at ORPHEUS_MAX_CHARS=120
    #    two such rows pack into ONE chunk when measured as printed and stay
    #    TWO when measured as words.
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

    def run(mode):
        os.environ['ORPHEUS_TEXT_TRANSFORM'] = mode
        os.environ['ORPHEUS_MAX_CHARS'] = '120'
        os.environ['SENTENCE_MIN_CHARS'] = '0'
        session_id = f'transform-switch-{mode}'
        session = core.context.set_session(session_id)
        session['tts_engine'] = 'orpheus'
        session['language'] = 'eng'
        session['language_iso1'] = 'en'
        session['is_gui_process'] = False
        session['skip_headings'] = False
        rows = core.filter_chapter(0, _Doc(html), session_id, False, True)
        return [r for r in rows if core._has_word_chars(r)]

    packed_as_printed = run('0')
    packed_as_words = run('1')
    print(f'  as printed: {len(packed_as_printed)} chunk(s); as words: {len(packed_as_words)} chunk(s)')
    for i, c in enumerate(packed_as_printed):
        print(f'    printed[{i}] {c!r}')
    for i, c in enumerate(packed_as_words):
        print(f'    words[{i}]   {c!r}')
    # Not an exact count — the twin-anchor and near-duplicate passes also cut
    # a run of repeated years, in both modes alike. What the switch changes is
    # the LENGTH the cap bounds: measured as printed the rows are short and pack
    # tighter, measured as words the same rows are ~6x longer and pack looser.
    expect('packer: measured as printed packs into FEWER chunks than measured as words',
           len(packed_as_printed) < len(packed_as_words), True)
    # filter_chapter returns rows with their SML restored ('[break]…'), so the
    # tags are stripped by pattern, not by the escaped-char test.
    from lib.conf_models import SML_TAG_PATTERN
    over = [c for c in packed_as_printed if len(SML_TAG_PATTERN.sub('', c).strip()) > 120]
    expect('packer: measured as printed, every chunk is within the 120-char cap as printed', over, [])

    for k in ('ORPHEUS_TEXT_TRANSFORM', 'ORPHEUS_MAX_CHARS', 'SENTENCE_MIN_CHARS'):
        os.environ.pop(k, None)

    print('\n==================== RESULT ====================')
    if failures:
        print(f'{len(failures)} failure(s):')
        for f in failures:
            print(f'  - {f}')
        return 1
    print('all text-transform-switch cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
