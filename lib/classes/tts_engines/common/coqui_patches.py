"""
Runtime patches for coqui-tts (the `TTS` package) bugs that bite on CUDA.

These are deterministic correctness fixes for upstream bugs — NOT fallbacks.
Inference still runs entirely on the user's selected device (CUDA when chosen).

## XTTS get_conditioning_latents + librosa.trim on a CUDA tensor

coqui-tts `Xtts.get_conditioning_latents` moves the reference audio to the model
device, then calls `librosa.effects.trim(audio, ...)` directly on that tensor:

    audio = audio[:, : load_sr * max_ref_length].to(self.device)   # -> cuda:0
    if librosa_trim_db is not None:
        audio = librosa.effects.trim(audio, top_db=librosa_trim_db)[0]

`librosa.effects.trim` is a numpy/CPU silence detector; handed a CUDA tensor it
hits `Tensor.__array__`, which raises
    "can't convert cuda:0 device type tensor to numpy"
On CPU the bug is masked (the tensor is already on host); on the GPU it surfaces,
and inside the parallel worker it shows up as a native 0xC0000005 crash
(exit code 3221225477) instead of a clean traceback.

The fix runs the silence DETECTION on a CPU copy (librosa's natural domain) but
keeps the audio as an on-device tensor by slicing it with the trim indices that
librosa returns. The tensor never leaves the GPU; only the boundary search uses
numpy, which is what librosa requires anyway.
"""

from __future__ import annotations

_APPLIED = False


def apply_coqui_patches() -> None:
    """Idempotently install the XTTS CUDA-trim fix. Safe to call repeatedly."""
    global _APPLIED
    if _APPLIED:
        return

    import librosa
    import torch
    from TTS.tts.models.xtts import Xtts

    @torch.inference_mode()
    def get_conditioning_latents(
        self,
        audio_path,
        max_ref_length: int = 30,
        gpt_cond_len: int = 6,
        gpt_cond_chunk_len: int = 6,
        librosa_trim_db=None,
        sound_norm_refs: bool = False,
        load_sr: int = 22050,
    ):
        # Faithful re-implementation of coqui-tts Xtts.get_conditioning_latents,
        # with ONE change: the librosa trim runs on a CPU copy and we slice the
        # on-device tensor by the returned indices, instead of feeding a CUDA
        # tensor straight into librosa (which can't convert it to numpy).
        from TTS.tts.models.xtts import load_audio

        if not isinstance(audio_path, list):
            audio_paths = [audio_path]
        else:
            audio_paths = audio_path

        speaker_embeddings = []
        audios = []
        speaker_embedding = None
        for file_path in audio_paths:
            audio = load_audio(file_path, load_sr)
            audio = audio[:, : load_sr * max_ref_length].to(self.device)
            if sound_norm_refs:
                audio = (audio / torch.abs(audio).max()) * 0.75
            if librosa_trim_db is not None:
                # Detect the non-silent span on CPU (librosa needs numpy), then
                # slice the device tensor by those sample indices so `audio`
                # stays on the model's device for the latent computations below.
                _, index = librosa.effects.trim(
                    audio.detach().cpu().numpy(), top_db=librosa_trim_db
                )
                audio = audio[:, int(index[0]) : int(index[1])]

            # compute latents for the decoder
            speaker_embedding = self.get_speaker_embedding(audio, load_sr)
            speaker_embeddings.append(speaker_embedding)

            audios.append(audio)

        # merge all the audios and compute the latents for the gpt
        full_audio = torch.cat(audios, dim=-1)
        gpt_cond_latents = self.get_gpt_cond_latents(
            full_audio, load_sr, length=gpt_cond_len, chunk_length=gpt_cond_chunk_len
        )

        if speaker_embeddings:
            speaker_embedding = torch.stack(speaker_embeddings)
            speaker_embedding = speaker_embedding.mean(dim=0)

        return gpt_cond_latents, speaker_embedding

    Xtts.get_conditioning_latents = get_conditioning_latents
    _APPLIED = True
    print("[COQUI-PATCH] Applied XTTS CUDA librosa-trim fix.")
