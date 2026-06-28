"""
Orpheus TTS Presets

Orpheus has 8 built-in voices with natural conversational quality.
No external files needed - voices are embedded in the model.

Voice quality ranking (per Orpheus docs, most conversational first):
tara, leah, jess, leo, dan, mia, zac, zoe
"""

from lib.conf_models import TTS_ENGINES, default_engine_settings

# Orpheus voices are built into the model, no external files needed
models = {
    "tara": {
        "lang": "eng",
        "voice": "tara",
        "description": "Tara - Female, most conversational/realistic",
        "gender": "female",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "leah": {
        "lang": "eng",
        "voice": "leah",
        "description": "Leah - Female, highly conversational",
        "gender": "female",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "jess": {
        "lang": "eng",
        "voice": "jess",
        "description": "Jess - Female, conversational",
        "gender": "female",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "mia": {
        "lang": "eng",
        "voice": "mia",
        "description": "Mia - Female",
        "gender": "female",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "zoe": {
        "lang": "eng",
        "voice": "zoe",
        "description": "Zoe - Female",
        "gender": "female",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "leo": {
        "lang": "eng",
        "voice": "leo",
        "description": "Leo - Male, conversational",
        "gender": "male",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "dan": {
        "lang": "eng",
        "voice": "dan",
        "description": "Dan - Male, conversational",
        "gender": "male",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    },
    "zac": {
        "lang": "eng",
        "voice": "zac",
        "description": "Zac - Male",
        "gender": "male",
        "samplerate": default_engine_settings[TTS_ENGINES['ORPHEUS']]['samplerate']
    }
}
