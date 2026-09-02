#!/usr/bin/env python
"""Every <li> is its OWN TTS chunk (2026-09-01).

THE DEFECT. Orpheus' end-of-speech failure concentrates on enumerated LIST text:
when several list items are packed into one ~350-500 char chunk, the model
re-speaks the final list phrase instead of stopping. Measured across 25 runs of
the 272-draw loop battery — the worst text in the whole battery is a numbered
list. e2a's prep was throwing the list structure away before the packer could
act on it: <li> is in none of heading_tags/break_tags/pause_tags, so it fell into
_tuple_row's transparent-inline branch and consecutive items welded into ONE
prose row at the ' '.join, long before get_sentences saw them.

THE FIX under test: filter_chapter's walker emits ('item_start'|'item_end')
around each <li>, the consumer marks the item's first text with TTS_SML['item'],
and both packers treat that marker as a RUN BOUNDARY — all of ONE item's
sentences pack together up to max_chars, an item never packs with a neighbouring
item or with the prose around the list, and the min-chars floor may not merge an
item away or merge anything into it.

Drives the REAL pipeline — filter_chapter() on fixture XHTML, which walks the
markup and runs the whole of get_sentences over it: PASS 1-3 splitting, PASS 4,
the Orpheus PASS 5 packer, the Voxtral packer, the min-chars floor and the
near-duplicate split. Nothing is stubbed except the epub document object, which
filter_chapter only ever asks for get_body_content().

What it proves, for the default (XTTS-class), orpheus and voxtral paths:
  1. each item's text comes out as its own group of chunks, exactly the item
     text plus the terminal period the walker forces, in document order;
  2. a two-sentence item is ONE chunk on the engines that PACK — the ordinal
     '13.' stays with the sentence it numbers;
  3. the intro paragraph and the paragraph after the list are chunks of their
     own, never welded to item one or item eight;
  4. nested lists give the inner <li> its own marker too;
  5. nothing an engine will SPEAK still contains the marker — checked through
     each engine's own strip path, not a re-implementation of it;
  6. the min-chars floor exempts item rows in BOTH directions, regardless of
     length, and still drops a wordless one;
  7. _convert_sml accepts [item] silently, and normalize_sml_tags accepts the
     flattened text (and still rejects a closing half, which is why the marker
     is non-paired).

XTTS-CLASS (and F5) REACH THE SAME RESULT BY A DIFFERENT ROUTE: that path has
NO packer at all — every row it gets from PASS 1-4 would be its own chunk — so
a two-sentence item arrives at the min-chars floor as '[item]13.' + 'A printer
is represented by a press.'. The floor's item exemption therefore lets a SHORT
item row gather the token-free row immediately after it (its own next
sentence, never the next item, which always opens on a [break]), so the
two-sentence item is one chunk on every engine (EXPECTED_ITEM_CHUNKS is [1]*8
for all three).

Run it with an interpreter that has e2a's dependencies, e.g. the bundled env:
    python_env/python.exe tools/test_list_item_chunks.py
Exit code 0 = all cases passed.
"""

import os
import re
import sys

# The fixture covers the shapes list items actually come in:
#   an intro <p> ending in a BARE LETTER — no terminal punctuation, so it can
#     only be kept out of item one by the period item_start's _close_block adds
#   a plain short item, and a longer one
#   an item with INLINE markup (<em>) — it must come out as one phrase, not as
#     'The third point' split at the emphasis
#   a TWO-SENTENCE item opening with an ordinal ('13.'). PASS 1 ends a row at
#     that ordinal's period, so this is exactly the case where isolating the
#     marked row alone would strand the item's own text in the next generation
#   a ONE-WORD item ('fourteen') with no punctuation at all — below the 25-char
#     min-chars floor, which used to merge such a row into its neighbour
#   a trailing <p> of prose, which must not join the last item
FIXTURE_XHTML = """<body>
<p>The committee recorded the following points before it adjourned for the evening</p>
<ol>
<li>The first point about printing</li>
<li>A second point that runs a little longer than the first one did</li>
<li>The <em>third</em> point</li>
<li>13. A printer is represented by a press. It stands for the whole trade.</li>
<li>fourteen</li>
<li>A point about the guild and its charter</li>
<li>The seventh point, briefly stated</li>
<li>A final point to close the list</li>
</ol>
<p>The argument continued for another hour without anyone changing their mind about it.</p>
</body>"""

# NESTED lists, kept as a second fixture so the first one stays readable. ul/ol
# are left TRANSPARENT by the walker (they hold no text of their own), so the
# only thing that can mark the inner item is the inner <li> itself.
FIXTURE_NESTED = """<body>
<p>The nested case is recorded here for the walker to prove out properly</p>
<ul><li>outer<ul><li>inner</li></ul></li></ul>
<p>And the prose after the nested list continues for a while without stopping.</p>
</body>"""

INTRO = 'The committee recorded the following points before it adjourned for the evening.'
TRAILING = 'The argument continued for another hour without anyone changing their mind about it.'

# What each <li> must read as, in order. Per engine, because '13' is not a
# per-engine detail of the ITEM — it is the number pipeline. Orpheus keeps
# book-exact text and expands nothing until the engine boundary, so it sees
# '13.'; every other engine runs math2words before splitting and sees
# 'thirteen.'. Same item either way.
_ITEMS = [
    'The first point about printing.',
    'A second point that runs a little longer than the first one did.',
    'The third point.',
    '{thirteen} A printer is represented by a press. It stands for the whole trade.',
    'fourteen.',
    'A point about the guild and its charter.',
    'The seventh point, briefly stated.',
    'A final point to close the list.',
]
EXPECTED_ITEMS = {
    'xtts': [s.format(thirteen='thirteen.') for s in _ITEMS],
    'orpheus': [s.format(thirteen='13.') for s in _ITEMS],
    'voxtral': [s.format(thirteen='thirteen.') for s in _ITEMS],
}

# How many CHUNKS each item is allowed to be: ONE, on every engine.
#
# Orpheus and Voxtral get there through their packers. The XTTS-class path has
# no packer — get_sentences returns the PASS 1-4 rows straight to the min-chars
# floor — so the two-sentence item arrives there as the two rows PASS 1 made of
# it ('thirteen.' and the two sentences PASS 4 merged). The floor's item
# exemption lets that short ordinal row gather the token-free row right after
# it (its own sentence; the next item always opens on a [break] so it can never
# be gathered), which is what keeps '13.' attached to its text on the engines
# that have no packer to do it. What the floor may NOT do, on any engine, is
# merge an item row into the item BEHIND it — the merge that produced the
# welded lists in the first place.
EXPECTED_ITEM_CHUNKS = {
    'xtts': [1] * 8,
    'orpheus': [1] * 8,
    'voxtral': [1] * 8,
}

ENGINES = ['xtts', 'orpheus', 'voxtral']


class _FixtureDoc:
    """The only thing filter_chapter asks of an epub document."""

    def __init__(self, html):
        self._html = html

    def get_body_content(self):
        return self._html


def _run_chapter(core, engine, html, tag):
    session_id = f'item-test-{engine}-{tag}'
    session = core.context.set_session(session_id)
    session['tts_engine'] = engine
    session['language'] = 'eng'
    session['language_iso1'] = 'en'
    session['is_gui_process'] = False
    session['skip_headings'] = False
    # stanza_nlp=False is the real 'no NER pipeline loaded' value core passes;
    # it only gates date expansion, which no fixture line needs.
    return core.filter_chapter(0, _FixtureDoc(html), session_id, False, True)


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


def _spoken_text(engine, chunk):
    """The text this engine will actually hand its model, produced by the
    engine's OWN stripping code — SML_UNSPOKEN_PATTERN for orpheus/voxtral/f5,
    and the split-and-convert loop for the XTTS-class engines."""
    from lib.conf_models import SML_TAG_PATTERN, SML_UNSPOKEN_PATTERN
    if engine in ('orpheus', 'voxtral'):
        return SML_UNSPOKEN_PATTERN.sub('', chunk).strip()
    spoken = []
    for part in _split_on_sml(SML_TAG_PATTERN, chunk):
        if SML_TAG_PATTERN.fullmatch(part):
            continue
        spoken.append(part)
    return re.sub(r'\s+', ' ', ''.join(spoken)).strip()


def _group_chunks(chunks):
    """Group a chapter's chunks into ('item'|'prose', [chunk indices]).

    A chunk carrying [item] OPENS an item group. A chunk carrying no SML tag at
    all that directly follows one CONTINUES it — that shape can only be an
    item's own later sentence, because PASS 1 ends a row immediately before any
    token and starts rows on tokens, so the next item and the paragraph after
    the list both open with at least a [break]. Everything else is prose.

    This is the test's model of what the pipeline promises, so a chunk that
    welded an item to its neighbour would land in the wrong group and be caught
    by the text comparison rather than passing on a lenient 'contains' check."""
    from lib.conf_models import SML_TAG_PATTERN
    groups = []
    for i, c in enumerate(chunks):
        if '[item]' in c:
            groups.append(['item', [i]])
        elif groups and groups[-1][0] == 'item' and SML_TAG_PATTERN.search(c) is None:
            groups[-1][1].append(i)
        else:
            groups.append(['prose', [i]])
    return groups


def _check_flat(core, engine):
    """The main fixture, end to end. Returns a list of failures."""
    failures = []
    chunks = _run_chapter(core, engine, FIXTURE_XHTML, 'flat')
    if chunks is None:
        return [f'{engine}: filter_chapter returned None']
    for i, c in enumerate(chunks):
        print(f'  [{i}] {c!r}')

    spoken = [_spoken_text(engine, c) for c in chunks]
    groups = _group_chunks(chunks)
    items = [g for g in groups if g[0] == 'item']
    prose = [g for g in groups if g[0] == 'prose']

    expected = EXPECTED_ITEMS[engine]
    expected_counts = EXPECTED_ITEM_CHUNKS[engine]

    # 1. EVERY <li> IS ITS OWN GROUP, IN ORDER, READING EXACTLY THE ITEM TEXT.
    if len(items) != len(expected):
        failures.append(f'{engine}: {len(items)} item group(s), want {len(expected)}')
    for n, (group, want, want_chunks) in enumerate(zip(items, expected, expected_counts)):
        got = ' '.join(spoken[i] for i in group[1]).strip()
        if got != want:
            failures.append(f'{engine}: item {n} reads {got!r}, want {want!r}')
        # 2. …and it is ONE chunk wherever a packer exists. The two-sentence
        #    item is the case this proves: '13.' must not be a generation of its
        #    own with its sentence stranded in the next.
        if len(group[1]) != want_chunks:
            failures.append(
                f'{engine}: item {n} ({want!r}) is {len(group[1])} chunk(s), want {want_chunks}: '
                f'{[chunks[i] for i in group[1]]!r}'
            )

    # 3. THE PROSE EITHER SIDE OF THE LIST IS NOT PART OF IT. Checked as whole
    #    chunk text, not 'in': a welded intro would still contain the intro.
    prose_spoken = [' '.join(spoken[i] for i in g[1]).strip() for g in prose]
    prose_spoken = [p for p in prose_spoken if p]
    if INTRO not in prose_spoken:
        failures.append(f'{engine}: the intro paragraph is not a chunk of its own: {prose_spoken!r}')
    if TRAILING not in prose_spoken:
        failures.append(f'{engine}: the trailing paragraph is not a chunk of its own: {prose_spoken!r}')
    # …and belt-and-braces at the chunk level: no single chunk may hold prose
    # AND an item, nor two items.
    for c, s in zip(chunks, spoken):
        if not s:
            continue
        hits = [w for w in expected if w in s]
        if len(hits) > 1:
            failures.append(f'{engine}: one chunk holds {len(hits)} items: {c!r}')
        if hits and (INTRO in s or TRAILING in s):
            failures.append(f'{engine}: a chunk welds prose to a list item: {c!r}')

    # 4. NOTHING SPOKEN STILL CARRIES THE MARKER, in tag form or escaped.
    for s in spoken:
        if '[item]' in s or '[/item]' in s:
            failures.append(f'{engine}: marker survived into spoken text: {s!r}')
        if any(ord(ch) >= core.sml_escape_tag for ch in s):
            failures.append(f'{engine}: an escaped SML char survived into spoken text: {s!r}')
        if s and not re.search(r'\w', s):
            failures.append(f'{engine}: chunk has nothing to speak: {s!r}')

    # 5. The flattened rows must still be legal SML (this is what
    #    filter_chapter itself asserts before splitting).
    ok, out = core.normalize_sml_tags(' '.join(chunks))
    if not ok:
        failures.append(f'{engine}: normalize_sml_tags rejected the chapter text: {out}')
    return failures


def _check_nested(core, engine):
    """Nested lists: the inner <li> is an item in its own right."""
    failures = []
    chunks = _run_chapter(core, engine, FIXTURE_NESTED, 'nested')
    if chunks is None:
        return [f'{engine}: filter_chapter returned None (nested)']
    for i, c in enumerate(chunks):
        print(f'  [{i}] {c!r}')
    spoken = [_spoken_text(engine, c) for c in chunks]
    items = [g for g in _group_chunks(chunks) if g[0] == 'item']
    got = [' '.join(spoken[i] for i in g[1]).strip() for g in items]
    if got != ['outer.', 'inner.']:
        failures.append(f'{engine}: nested items read {got!r}, want ["outer.", "inner."]')
    return failures


def _check_convert_sml_accepts_item():
    """The XTTS-class engines route every SML fragment through _convert_sml,
    which returns (False, error) for a tag it does not know — that would abort
    the whole sentence. Prove it accepts [item], and appends no audio."""
    from lib.classes.tts_engines.common.utils import TTSUtils
    from lib.conf_models import TTS_SML
    inst = TTSUtils.__new__(TTSUtils)
    inst.audio_segments = []
    ok, error = inst._convert_sml(TTS_SML['item']['static'])
    if not ok:
        return False, f'_convert_sml rejected [item]: {error}'
    if inst.audio_segments:
        return False, '_convert_sml appended audio for [item] (it must be silent markup)'
    return True, ''


def _check_normalize_sml(core):
    """[item] is NON-PAIRED, and normalize_sml_tags is where that is enforced:
    it rejects a closing half for a non-paired tag. That rejection is the reason
    the marker is a single leading token — see TTS_SML['item']."""
    failures = []
    ok, out = core.normalize_sml_tags('[break][item]One. [break][item]Two.')
    if not ok:
        failures.append(f'normalize_sml_tags rejected a marked list: {out}')
    elif out != '[break][item]One. [break][item]Two.':
        failures.append(f'normalize_sml_tags rewrote a marked list: {out!r}')
    ok, out = core.normalize_sml_tags('[item]One.[/item]')
    if ok:
        failures.append('normalize_sml_tags accepted [/item] — the tag must stay non-paired')
    return failures


def _check_floor_item_exemption(core):
    """The min-chars floor, driven DIRECTLY over row sequences (2026-09-01).

    The fixtures prove the real path end to end; this proves the exemption at
    the layer that owns it, including the two neighbour arrangements the fixture
    markup cannot force. Returns a list of failures.

    A row is built the way get_sentences sees one: escape_sml has already turned
    each SML block into ONE char whose index into sml_blocks is its identity, so
    chr(sml_escape_tag + i) IS the token here."""
    blocks = ['[item]', '[break]']
    item, brk = chr(core.sml_escape_tag), chr(core.sml_escape_tag + 1)
    # No [heading] in this block table, so the heading predicate answers False
    # for every row here — these cases are about the ITEM exemption only.
    is_heading = core._marker_row_test(blocks, 'heading')
    is_item = core._marker_row_test(blocks, 'item')

    def clean_len(s):
        return len(core._strip_escaped_sml(s))

    def has_words(s):
        return core._has_word_chars(s)

    def run(rows, min_chars=25, max_chars=350):
        return core._apply_min_chars_floor(rows, clean_len, max_chars, min_chars,
                                           is_heading, is_item, has_words)

    failures = []

    def expect(label, rows, want):
        args = rows if isinstance(rows, tuple) else (rows,)
        got = run(*args)
        print(f'  {label}\n      in : {[core._strip_escaped_sml(r) for r in args[0]]!r}'
              f'\n      out: {[core._strip_escaped_sml(r) for r in got]!r}')
        if got != want:
            failures.append(f'{label}: got {got!r}, want {want!r}')

    # 1. A SHORT ITEM IS NOT MERGED AWAY. 'fourteen.' is 9 chars, well under the
    #    25-char floor, and the row after it is a comfortable landing site — the
    #    exemption is the only thing keeping the two apart.
    rows = [f'{brk}{item}fourteen.', f'{brk}{item}A point about the guild and its charter.']
    expect('a 9-char list item still stands alone', rows, list(rows))

    # 2. NOTHING IS MERGED INTO AN ITEM, FORWARD. The row in front of a list is
    #    the prose the list belongs to; gluing it onto item one is the weld the
    #    marker exists to forbid.
    rows = ['No.', f'{brk}{item}A list item long enough to clear the floor.']
    expect('a short row is not merged forward into an item', rows, list(rows))

    # 3. …nor BACKWARD. Symmetric refusal, same reason.
    rows = [f'{brk}{item}A list item long enough to clear the floor.', 'No.']
    expect('a short row is not merged backward into an item', rows, list(rows))

    # 4. THE EXEMPTION IS NOT A LICENCE TO SHIP NOTHING. _drop_wordless_rows runs
    #    FIRST, so an item whose whole text is decoration ('<li>•</li>', the
    #    bullet conf_lang maps to a period) is dropped, exemption or not — the
    #    same order that fixed the 30 chunks reading '.' in Hugh Howey's "Shift".
    expect('a list item with no word in it is dropped, exemption or not',
           [f'{brk}{item}.', 'Some prose that is comfortably long enough.'],
           ['Some prose that is comfortably long enough.'])

    # 5. SENTENCE_MIN_CHARS=0 turns the LENGTH floor off; the item exemption has
    #    nothing to do either way, and the wordless rule still holds.
    expect('the wordless rule holds for items with the length floor disabled',
           ([f'{brk}{item}.', 'Some prose that is comfortably long enough.'], 0),
           ['Some prose that is comfortably long enough.'])

    # 6. A SHORT ITEM GATHERS THE TOKEN-FREE ROW RIGHT AFTER IT — its own next
    #    sentence, on the engines with no packer (XTTS-class, F5) where the
    #    PASS 1-4 rows reach the floor unpacked. The marker rides the kept lead,
    #    so the merged row is still the item. (Orpheus/Voxtral never present
    #    this shape: their packers gathered the sentence already.)
    expect('a short item gathers its own next (token-free) sentence',
           [f'{brk}{item}13.', 'A printer is represented by a press.'],
           [f'{brk}{item}13. A printer is represented by a press.'])

    # 6b. …and re-examines the result: a three-sentence item gathers twice when
    #     the first gather is still under the floor.
    expect('a short item keeps gathering its own sentences until it clears the floor',
           [f'{brk}{item}13.', 'A press.', 'It stands for the whole trade.'],
           [f'{brk}{item}13. A press. It stands for the whole trade.'])

    # 6c. BUT NEVER A ROW THAT CARRIES A TOKEN. The next item and the paragraph
    #     after the list always open on a [break], so a short item that is the
    #     whole of its <li> stays alone — this is the merge that welded lists.
    rows = [f'{brk}{item}fourteen.', f'{brk}The paragraph after the list is long enough on its own.']
    expect('a short item does not gather the next paragraph', rows, list(rows))
    rows = [f'{brk}{item}fourteen.', f'{brk}{item}A following item that is long enough on its own.']
    expect('a short item does not gather the next item', rows, list(rows))

    # 6d. A SHORT ITEM IS NOT MERGED FORWARD BY THE SHORT-HEADING PASS EITHER.
    #     _merge_short_headings_forward is deliberately heading-only: a demoted
    #     heading joins the text it already belongs to, a demoted item would join
    #     a DIFFERENT item. Proven by the gather refusing a token-carrying row
    #     above: the heading pass would have merged 'fourteen.' into it.

    return failures


def main():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, repo)
    os.chdir(repo)
    import lib.core as core
    print(f'TESTING TREE: {os.path.dirname(os.path.abspath(core.__file__))}')
    core.context = core.SessionContext()

    failures = []

    ok, error = _check_convert_sml_accepts_item()
    print(f'\n--- xtts-class _convert_sml([item]) --- {"OK" if ok else "FAIL"}')
    if not ok:
        failures.append(error)

    print('\n--- normalize_sml_tags: [item] is a non-paired leading marker ---')
    nf = _check_normalize_sml(core)
    print(f'  {"OK" if not nf else "FAIL"}')
    failures.extend(nf)

    print('\n--- min-chars floor: a list item is exempt in both directions ---')
    failures.extend(_check_floor_item_exemption(core))

    for engine in ENGINES:
        print(f'\n===== {engine} =====')
        failures.extend(_check_flat(core, engine))
        print(f'  -- nested list --')
        failures.extend(_check_nested(core, engine))

    print('\n==================== RESULT ====================')
    if failures:
        for f in failures:
            print(f'FAIL: {f}')
        print(f'{len(failures)} failure(s)')
        return 1
    print('all list-item-chunk cases passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
