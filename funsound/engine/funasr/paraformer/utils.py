# -*- encoding: utf-8 -*-

from funsound.lib import *
from onnxruntime import (
        GraphOptimizationLevel,
        InferenceSession,
        SessionOptions,
        get_available_providers,
        get_device,
        ExecutionMode,
    )
import onnxruntime


root_dir = Path(__file__).resolve().parent

logger_initialized = {}


def pad_list(xs, pad_value, max_len=None):
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    # pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    # numpy format
    pad = (np.zeros((n_batch, max_len)) + pad_value).astype(np.int32)
    for i in range(n_batch):
        pad[i, : xs[i].shape[0]] = xs[i]

    return pad


class TokenIDConverter:
    def __init__(
        self,
        token_list: Union[List, str],
    ):

        self.token_list = token_list
        self.unk_symbol = token_list[-1]
        self.token2id = {v: i for i, v in enumerate(self.token_list)}
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise TokenIDConverterError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:

        return [self.token2id.get(i, self.unk_id) for i in tokens]


class CharTokenizer:
    def __init__(
        self,
        symbol_value: Union[Path, str, Iterable[str]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
    ):

        self.space_symbol = space_symbol
        self.non_linguistic_symbols = self.load_symbols(symbol_value)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    @staticmethod
    def load_symbols(value: Union[Path, str, Iterable[str]] = None) -> Set:
        if value is None:
            return set()

        if isinstance(value, Iterable[str]):
            return set(value)

        file_path = Path(value)
        if not file_path.exists():
            logging.warning("%s doesn't exist.", file_path)
            return set()

        with file_path.open("r", encoding="utf-8") as f:
            return set(line.rstrip() for line in f)

    def text2tokens(self, line: Union[str, list]) -> List[str]:
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols:
                if line.startswith(w):
                    if not self.remove_non_linguistic_symbols:
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = "<space>"
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f")"
        )


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: np.ndarray
    score: Union[float, np.ndarray] = 0
    scores: Dict[str, Union[float, np.ndarray]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
            scores={k: float(v) for k, v in self.scores.items()},
        )._asdict()


class TokenIDConverterError(Exception):
    pass



def time_stamp_lfr6_onnx(us_cif_peak, char_list, begin_time=0.0, total_offset=-1.5):

    if not len(char_list):
        return []
    START_END_THRESHOLD = 5
    MAX_TOKEN_DURATION = 30
    TIME_RATE = 10.0 * 6 / 1000 / 3  #  3 times upsampled
    cif_peak = us_cif_peak.reshape(-1)
    num_frames = cif_peak.shape[-1]
    if char_list[-1] == '</s>':
        char_list = char_list[:-1]
    # char_list = [i for i in text]
    timestamp_list = []
    new_char_list = []
    # for bicif model trained with large data, cif2 actually fires when a character starts
    # so treat the frames between two peaks as the duration of the former token
    fire_place = np.where(cif_peak>1.0-1e-4)[0] + total_offset  # np format
    num_peak = len(fire_place)
    assert num_peak == len(char_list) + 1 # number of peaks is supposed to be number of tokens + 1
    # begin silence
    if fire_place[0] > START_END_THRESHOLD:
        # char_list.insert(0, '<sil>')
        timestamp_list.append([0.0, fire_place[0]*TIME_RATE])
        new_char_list.append('<sil>')
    # tokens timestamp
    for i in range(len(fire_place)-1):
        new_char_list.append(char_list[i])
        if i == len(fire_place)-2 or MAX_TOKEN_DURATION < 0 or fire_place[i+1] - fire_place[i] < MAX_TOKEN_DURATION:
            timestamp_list.append([fire_place[i]*TIME_RATE, fire_place[i+1]*TIME_RATE])
        else:
            # cut the duration to token and sil of the 0-weight frames last long
            _split = fire_place[i] + MAX_TOKEN_DURATION
            timestamp_list.append([fire_place[i]*TIME_RATE, _split*TIME_RATE])
            timestamp_list.append([_split*TIME_RATE, fire_place[i+1]*TIME_RATE])
            new_char_list.append('<sil>')
    # tail token and end silence
    if num_frames - fire_place[-1] > START_END_THRESHOLD:
        _end = (num_frames + fire_place[-1]) / 2
        timestamp_list[-1][1] = _end*TIME_RATE
        timestamp_list.append([_end*TIME_RATE, num_frames*TIME_RATE])
        new_char_list.append("<sil>")
    else:
        timestamp_list[-1][1] = num_frames*TIME_RATE
    if begin_time:  # add offset time in model with vad
        for i in range(len(timestamp_list)):
            timestamp_list[i][0] = timestamp_list[i][0] + begin_time / 1000.0
            timestamp_list[i][1] = timestamp_list[i][1] + begin_time / 1000.0
    assert len(new_char_list) == len(timestamp_list)
    res_str = ""
    for char, timestamp in zip(new_char_list, timestamp_list):
        res_str += "{} {} {};".format(char, timestamp[0], timestamp[1])
    res = []
    for char, timestamp in zip(new_char_list, timestamp_list):
        if char != '<sil>':
            res.append([int(timestamp[0] * 1000), int(timestamp[1] * 1000)])
    return res_str, res


def process_tokens(tokens):
        def is_chinese_word(word):
            for char in word:
                if not ('\u4e00' <= char <= '\u9fff'):
                    return False
            return True
        result = []
        current_word = ''
        previous_token_type = None  # 'C' for Chinese, 'E' for English

        for token in tokens:
            if is_chinese_word(token):
                # Token is a Chinese character
                if previous_token_type == 'E':
                    result.append(' ')
                result.append(token)
                previous_token_type = 'C'
            else:
                # Token is part of an English word
                token_clean = token.replace('@@', '')
                current_word += token_clean

                if not token.endswith('@@'):
                    # End of the English word
                    if previous_token_type in ('C', 'E') and previous_token_type is not None:
                        result.append(' ')
                    result.append(current_word)
                    current_word = ''
                    previous_token_type = 'E'
        ans = ''.join(result)
        # if ans[-1]==" ":ans = ans[:-1]
        return ans


def pad_list(xs, pad_value, max_len=None):
    n_batch = len(xs)
    if max_len is None:
        max_len = max(x.size(0) for x in xs)
    # pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)
    # numpy format
    pad = (np.zeros((n_batch, max_len)) + pad_value).astype(np.int32)
    for i in range(n_batch):
        pad[i, : xs[i].shape[0]] = xs[i]

    return pad

