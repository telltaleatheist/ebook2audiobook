"""ASR verify gate — did the model actually SAY the words it was given?

The Gods People census (orpheus-finetune TESTING_BIBLE, 2026-08-29) measured
mid-chunk derailments the duration guards cannot see: the audio is the right
LENGTH but the wrong WORDS ('October fifteen, nineteen forty-four. The movement
had spent a generation writing' spoken as 'October 9th, Cineology, rating').
Number-word runs and digit clusters are where it happens (asr_gate_risk in
orpheus_text.py picks those chunks); this module is the check itself.

wav2vec2 CTC (torchaudio's bundled WAV2VEC2_ASR_BASE_960H) on CPU: no LM, no
beam — it tracks acoustics tightly, so a >= 4-word hole in its greedy transcript
means words genuinely were not spoken, while its ordinary letter-level noise
('KUHN' for 'Kun') stays below the word-run thresholds. The census used the
same thresholds against whisper output and every flagged case was real.

Fail-open by design: if torch/torchaudio are missing (Mac MLX env, stripped
installs) the gate reports itself unavailable once and every check passes.
Disable explicitly with ORPHEUS_ASR_GATE=0.
"""
import difflib
import os
import re

_BUNDLE = None          # (model, labels, blank_idx) once loaded
_UNAVAILABLE = False    # import or load failed — gate passes everything


def gate_enabled() -> bool:
    return os.environ.get('ORPHEUS_ASR_GATE', '1') != '0'


def _load():
    global _BUNDLE, _UNAVAILABLE
    if _BUNDLE is not None or _UNAVAILABLE:
        return _BUNDLE
    try:
        import torch
        import torchaudio
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().eval()
        # CPU on purpose: vLLM owns the GPU's memory reservation; a second CUDA
        # context fighting it for the remainder is how OOM ladders start.
        model.to('cpu')
        _BUNDLE = (model, bundle.get_labels(), bundle.sample_rate)
    except Exception as e:  # noqa: BLE001 — any failure means "no gate", never "no render"
        print(f'[ASR-GATE] unavailable, checks pass open: {e}')
        _UNAVAILABLE = True
        _BUNDLE = None
    return _BUNDLE


def _greedy_decode(emission, labels) -> str:
    # Index 0 is the CTC blank in torchaudio's wav2vec2 bundles (its label
    # prints as '-', which is NOT a hyphen in the output alphabet).
    prev = -1
    out = []
    for idx in emission.argmax(-1).tolist():
        if idx != prev and idx != 0 and labels[idx] not in ('<s>', '<pad>', '<unk>'):
            out.append(labels[idx])
        prev = idx
    return ''.join(out).replace('|', ' ').strip()


def _norm_words(text: str) -> list:
    text = re.sub(r"[^a-z\s']", ' ', text.lower())
    text = re.sub(r"'", '', text)
    return text.split()


def check(audio_np, sample_rate: int, expected_text: str) -> dict:
    """Compare generated audio against the text the model was told to speak.

    Returns {'ok': bool, 'ratio': float, 'drop_run': int, 'heard': str}.
    ok=False means a >= 4-word run of the expected text has no counterpart in
    the transcript (the census DROP threshold) or a compressing >= 5-word
    replace (the GARBLE threshold). ASR being unavailable returns ok=True.
    """
    bundle = _load()
    if bundle is None:
        return {'ok': True, 'ratio': 1.0, 'drop_run': 0, 'heard': ''}
    import torch
    import torchaudio.functional as F
    model, labels, asr_sr = bundle
    wave = torch.as_tensor(audio_np, dtype=torch.float32).reshape(1, -1)
    if sample_rate != asr_sr:
        wave = F.resample(wave, sample_rate, asr_sr)
    with torch.inference_mode():
        emission, _ = model(wave)
    heard = _greedy_decode(emission[0], labels)
    want, got = _norm_words(expected_text), _norm_words(heard)
    if not want:
        return {'ok': True, 'ratio': 1.0, 'drop_run': 0, 'heard': heard}
    sm = difflib.SequenceMatcher(a=want, b=got, autojunk=False)
    worst = 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'delete':
            worst = max(worst, i2 - i1)
        elif op == 'replace' and i2 - i1 >= 5 and (i2 - i1) - (j2 - j1) >= 3:
            worst = max(worst, i2 - i1)
    ratio = round(sm.ratio(), 3)
    # ratio floor: a transcript that broadly disagrees with the text is a fail
    # even without one long hole (a garble is substitutions, not deletions).
    ok = worst < 4 and ratio >= 0.5
    return {'ok': ok, 'ratio': ratio, 'drop_run': worst, 'heard': heard}
