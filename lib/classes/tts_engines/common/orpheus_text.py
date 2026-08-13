"""Orpheus book-exact text transforms — the ONLY lexical changes between the
book text and what an Orpheus fine-tune reads.

Faithful ports of the training-corpus extractor (orpheus-finetune repo:
cut_audiobook.py normalize_scripture, cut_excerpts.py expand_digits,
align_excerpts.py num_to_words). The fine-tunes are trained on epub-exact
transcripts with exactly these two transforms applied, in this order:
normalize_scripture, then expand_digits. Inference must feed the same
distribution, so any change here must land in the training pipeline too (and
vice versa).

Style notes that matter:
- num_to_words is cardinal, no hyphens, no 'and': 1923 ->
  'one thousand nine hundred twenty three' (NOT num2words style).
- YEARS are the one place this file no longer matches the corpora it was ported
  from: a bare 4-digit year goes through lib.core.year2words ('nineteen
  forty-three') instead of the cardinal. See year_words for the measurement and
  the argument. The cost is the ordinary year/quantity ambiguity — '1200 people'
  reads as 'twelve hundred people' — which is the same trade lib/core.py:1429
  already makes for every acoustic engine, with the same heuristic.
- expand_digits only touches a whitespace-delimited token that is
  punctuation + a bare 1-4 digit integer + punctuation. Everything else stays
  as printed: '5,000', '1930s', '7th', '$5.50', '160299', 'Henry VIII', 'Mr.'.
- Both transforms are idempotent on their own output (no digits remain), so
  re-applying to sentences from an old already-expanded session is a no-op.

Stdlib re only — importable from both lib/core.py and the engine without
dragging heavy deps into workers.
"""
import re

_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety"]


def num_to_words(n: int) -> list[str] | None:
    # Port of orpheus-finetune align_excerpts.num_to_words (training-text style)
    if n < 0 or n > 9999:
        return None
    if n < 20:
        return [_ONES[n]]
    if n < 100:
        w = [_TENS[n // 10]]
        if n % 10:
            w.append(_ONES[n % 10])
        return w
    if n < 1000:
        w = [_ONES[n // 100], "hundred"]
        if n % 100:
            w += num_to_words(n % 100)
        return w
    w = [_ONES[n // 1000], "thousand"]
    if n % 1000:
        w += num_to_words(n % 1000)
    return w


_YEAR_RE = re.compile(r'1\d{3}|20\d{2}')


def year_words(token: str) -> str | None:
    """A four-digit YEAR as a year is read, through e2a's OWN year heuristic.

    ── Why a year needs its own reading, and why this is not a new rule ───────

    Every training corpus on disk expands a bare year through num_to_words, so
    the text the fine-tunes were trained on says 'one thousand nine hundred
    thirty three' where the narrator's audio says 'nineteen thirty three'
    (measured across all of E:/training, 2026-08-13: tr_ae8h_v2 has 426
    word-years to 14 digit-years, and every corpus holding a year matches). The
    pairing was meant to teach the model the year reading; it did not take —
    Owen hears years read out as quantities — and matching that distribution
    more exactly cannot fix a mapping the training never learned. So the model
    is handed the words it should SPEAK, because literal reading is an LLM TTS's
    most reliable behaviour.

    The reading itself is `lib.core.year2words`, the SAME heuristic every
    acoustic engine has used all along (Owen, 2026-08-13: "we shouldnt have to
    build out date naming logic - it should already be present in e2a"). The
    import is deferred because lib.core imports this module.

    Range and shape match the acoustic branch's own call (core.py ~1429):
    `1\\d{3}` or `20\\d{2}`, bare. A year RANGE ('1914-1918'), a comma'd number
    and anything else never reach here — the caller's token shape excludes them.
    """
    if not _YEAR_RE.fullmatch(token):
        return None
    from lib.core import year2words   # deferred: lib.core imports this module
    return year2words(token, 'eng', 'en', True)


def expand_digits(text: str) -> str:
    # Port of orpheus-finetune cut_excerpts.expand_digits, made whitespace-safe:
    # the original split()/join collapsed ALL whitespace (fine on single
    # training cues); here paragraph breaks must survive. Token shape is
    # identical: punctuation + bare 1-4 digit integer + punctuation, delimited
    # by whitespace. Any digit-bearing token that does not match is left
    # exactly as printed — the fine-tunes were TRAINED on those as digits.
    def repl(m: re.Match) -> str:
        # A four-digit year is read as a year; everything else is the quantity it
        # looks like. '50 people' -> 'fifty people' is right and stays;
        # '1943' -> 'one thousand nine hundred forty three' never was.
        year = year_words(m.group(2))
        if year is not None:
            return m.group(1) + year + m.group(3)
        words = num_to_words(int(m.group(2)))
        if not words:
            return m.group(0)
        return m.group(1) + " ".join(words) + m.group(3)
    return re.sub(r'(?<!\S)([^\w\s]*)(\d{1,4})([^\w\s]*)(?!\S)', repl, text)


def normalize_scripture(text: str) -> str:
    # Port of orpheus-finetune cut_audiobook.normalize_scripture: Bible refs to
    # the spoken form ('1 John 1:9' -> 'First John one nine',
    # 'Matthew 5:16-18' -> 'Matthew five sixteen through eighteen'). Runs
    # BEFORE expand_digits so the ordinal book prefix is not turned into a
    # cardinal ('one John').
    # Deliberate deviation from the training version: its ordinal-prefix rule
    # rewrote EVERY '[123] Capitalized' pair, safe in a scripture-dense book
    # but wrong on general prose ('Chapter 3 The Journey' -> 'Third The
    # Journey'); here the ordinal applies only when a chapter:verse ref
    # follows the name.
    text = re.sub(r"\b([123]) ([A-Z][a-z]+ \d+:\d+)",
                  lambda m: {'1': 'First', '2': 'Second', '3': 'Third'}[m.group(1)] + " " + m.group(2),
                  text)

    def _num(n: str) -> str:
        x = num_to_words(int(n))
        return " ".join(x) if x else str(n)

    def repl(m: re.Match) -> str:
        s = f"{_num(m.group('c'))} {_num(m.group('v'))}"
        if m.group('v2'):
            s += f" through {_num(m.group('v2'))}"
        return s
    return re.sub(r"(?P<c>\d+):(?P<v>\d+)(?:[–-](?P<v2>\d+))?(?:ff\.)?", repl, text)


def to_tts_form(text: str) -> str:
    """The full book-exact -> model-input transform, in training order."""
    return expand_digits(normalize_scripture(text))
