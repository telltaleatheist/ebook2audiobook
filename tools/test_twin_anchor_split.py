#!/usr/bin/env python
"""A repeated phrase inside one Orpheus generation is a SKIP primer (2026-08-29).

THE DEFECT. Whisper-vs-VTT census on Gods People (thirdreich tr_pc1_ep732,
2026-08-29 render): a chunk containing "which made him unacceptable to the
church moderates, or an incompetent who couldn't control his own movement,
which made him unacceptable to Hitler" was spoken as "...which made him
unacceptable to Hitler" — the model's attention resolved to the SECOND copy of
the repeated phrase and silently deleted the 85 chars between. The forward
mirror of the near-duplicate LOOP primer PASS 6 already guards against.

THE FIX. PASS 6 (_split_near_dup_chunk) now also splits at TWIN ANCHORS: a
qualifying 4-gram (>= 14 normalized chars) occurring twice in one chunk. Twins
across sentences split at the sentence boundary; a twin inside ONE sentence
splits at a comma/semicolon/dash between the copies (_split_intra_twin), each
piece keeping its own copy. Measured split rate on the full Gods People chunk
list: 2.0% (21/1037) — surgical, not a packing rewrite.

Run (WSL e2a env): python tools/test_twin_anchor_split.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.core import (_split_intra_twin, _split_near_dup_chunk,  # noqa: E402
                      _twin_anchor_grams)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  ok   {name}")
    else:
        failures.append(name)
        print(f"  FAIL {name}  {detail}")


KRAUSE = (
    "Krause's speech was catastrophic because it stripped away the respectable "
    "language the movement had been using and exposed its core ideology to the "
    "rest of the German church for the first time. Müller stripped Krause of "
    "his offices within days. But the damage was structural, not personal. He "
    "was now either a fraud who'd hidden the ideology, which made him "
    "unacceptable to the church moderates, or an incompetent who couldn't "
    "control his own movement, which made him unacceptable to Hitler."
)

print("twin-anchor split (PASS 6 extension):")

# 1. The measured real-world defect chunk splits, and the split separates the
#    two copies of the anchor phrase.
r = _split_near_dup_chunk(KRAUSE)
check("krause chunk splits", len(r) == 2, f"got {len(r)} pieces")
if len(r) == 2:
    check("each piece keeps one anchor copy",
          "unacceptable to" in r[0] and "unacceptable to" in r[1])
    check("no text lost", "".join(KRAUSE.split()) == "".join(("".join(r)).split()))

# 2. The intra-sentence splitter alone handles the final sentence.
last = KRAUSE.split("But the damage was structural, not personal. ")[1]
r = _split_intra_twin(last)
check("intra-sentence twin splits at a comma", len(r) == 2,
      f"got {len(r)} pieces")

# 3. Plain prose is returned byte-identical — the pass must stay a no-op for
#    non-repetitive text (the near-dup pass's documented contract).
plain = ("Szálasi, head of the Arrow Cross, seized power in Budapest on "
         "October fifteen, nineteen forty-four. The movement had spent a "
         "generation writing its program. Now it had a country.")
check("non-twin prose unsplit and byte-identical",
      _split_near_dup_chunk(plain) == [plain])

# 4. Stock collocations stay below the 14-char bar: a repeated short glue
#    phrase must NOT split ("at the end of" = 10 normalized chars).
glue = ("At the end of the street stood a bakery that had served the town for "
        "decades. At the end of the war it was the only building left standing "
        "on the block, and people remembered that.")
check("short collocation does not split",
      _split_near_dup_chunk(glue) == [glue])

# 5. Cross-sentence twins split at the sentence boundary.
cross = ("The instruction to party members was explicit and it was printed in "
         "the morning edition for everyone to read. The instruction to party "
         "members was repeated on the radio that evening in the same words.")
r = _split_near_dup_chunk(cross)
check("cross-sentence twin splits", len(r) == 2, f"got {len(r)} pieces")

# 6. A twin whose pieces would be tiny is left alone (no starved prompts).
tiny = "Yes he said, yes he said."
check("tiny twin left unsplit", _split_near_dup_chunk(tiny) == [tiny])

# 7. Grams: the qualifying bar admits the real anchor, rejects short glue.
check("anchor phrase qualifies",
      ("which", "made", "him", "unacceptable") in _twin_anchor_grams(
          "which made him unacceptable to everyone"))
check("short glue does not qualify",
      ("at", "the", "end", "of") not in _twin_anchor_grams(
          "at the end of the day"))

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all twin-anchor tests pass")
