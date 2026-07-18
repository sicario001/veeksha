from veeksha.types.base_int_enum import BaseIntEnum


# ----- Traffic -----
class TrafficType(BaseIntEnum):
    RATE = 1
    CONCURRENT = 2
    SEQUENTIAL_LAUNCH = 3


# ----- Interval / length generators -----
class IntervalGeneratorType(BaseIntEnum):
    POISSON = 1
    GAMMA = 2
    FIXED = 3


class LengthGeneratorType(BaseIntEnum):
    ZIPF = 1
    UNIFORM = 2
    FIXED = 3
    FIXED_STAIR = 4
    INVERSE_GAUSSIAN = 5


# ----- Content -----
class SessionGeneratorType(BaseIntEnum):
    SYNTHETIC = 1
    LMEVAL = 2
    TRACE = 3


class TraceFlavorType(BaseIntEnum):
    TIMED_SYNTHETIC_SESSION = 1
    SHARED_PREFIX = 2
    RAG = 3
    REQUEST_LOG = 4
    UNTIMED_CONTENT_MULTI_TURN = 5
    SHAREGPT = 6  # reserved (branch flavor; generator not yet ported)
    SEED_TTS_TEXT = 7  # TTS input-text trace
    AUDIO = 8  # ASR audio-manifest trace


class ChannelModality(BaseIntEnum):
    TEXT = 1
    IMAGE = 2
    AUDIO = 3
    VIDEO = 4


class AudioTask(BaseIntEnum):
    """The workload an AUDIO channel represents (tags ChannelResponse.metrics)."""

    TTS = 1  # text -> audio
    STT = 2  # audio -> text (ASR)
    LLM_AUDIO = 3


class SessionGraphType(BaseIntEnum):
    LINEAR = 1
    SINGLE_REQUEST = 2
    BRANCHING = 3


# ----- Evaluation -----
class EvaluationType(BaseIntEnum):
    PERFORMANCE = 1
    ACCURACY_LMEVAL = 2
    AUDIO_QUALITY = 3


# ----- Client -----
class ClientType(BaseIntEnum):
    OPENAI_CHAT_COMPLETIONS = 1
    OPENAI_COMPLETIONS = 2
    OPENAI_ROUTER = 3
    TTS = 4
    REALTIME_TTS = 5
    STT = 6


# ----- Server -----
class ServerType(BaseIntEnum):
    VLLM = 1
    SGLANG = 2
    VAJRA = 3


# ----- Verification -----
class VerificationType(BaseIntEnum):
    AUDIO = 1


# ----- SLOs -----
class SloType(BaseIntEnum):
    CONSTANT = 1


# ----- LMEval -----
class LMEvalOutputType(BaseIntEnum):
    LOGLIKELIHOOD = 1
    LOGLIKELIHOOD_ROLLING = 2
    GENERATE_UNTIL = 3
    MULTIPLE_CHOICE = 4
