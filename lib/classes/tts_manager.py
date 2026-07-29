import lib.classes.tts_engines

from typing import Any
from lib.classes.tts_registry import TTSRegistry

class TTSManager:

    def __init__(self, session:Any)->None:
        self.session = session
        engine_name = session.get("tts_engine")
        if engine_name is None:
            raise ValueError("session['tts_engine'] is missing")
        try:
            engine_cls = TTSRegistry.ENGINES[engine_name]
        except KeyError:
            raise ValueError(
                f"Invalid tts_engine '{engine_name}'. "
                f"Expected one of: {', '.join(TTSRegistry.ENGINES)}"
            )
        self.engine = engine_cls(session)

    def convert_sentence2audio(self, sentence_number: int, sentence: str) -> bool:
        return self.engine.convert(sentence_number, sentence)

    @property
    def supports_batch(self) -> bool:
        """True when the engine can convert a batch of sentences in one call
        (e.g. Orpheus via vLLM). The worker uses this to drive batched inference."""
        return bool(getattr(self.engine, 'SUPPORTS_BATCH', False)
                    and hasattr(self.engine, 'convert_batch'))

    @property
    def batch_size(self) -> int:
        return int(getattr(self.engine, 'BATCH_SIZE', 1) or 1)

    @property
    def batch_pool_size(self) -> int:
        """How many sentences the caller should accumulate before calling
        convert_sentences_batch(). Defaults to batch_size; an engine may ask for a
        DEEPER pool and re-slice it internally while still capping the batches it
        actually runs. Never smaller than batch_size. (Orpheus/MLX used to ask for
        4x so length-bucketing could rebuild full-width batches; bucketing and the
        pool both went away when mlx-lm 0.31.3 fixed batch prefill padding.)"""
        pool = getattr(self.engine, 'batch_pool_size', None)
        try:
            pool = int(pool or 0)
        except (TypeError, ValueError):
            pool = 0
        return max(self.batch_size, pool)

    def convert_sentences_batch(self, items: list) -> list:
        """items: list of (sentence_index, sentence). Returns list[bool] aligned to items."""
        return self.engine.convert_batch(items)

    def create_sentences2vtt(self, all_sentences:list)->bool:
        return self.engine.create_vtt(all_sentences)