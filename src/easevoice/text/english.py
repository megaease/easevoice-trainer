from nltk import pos_tag
import nltk
import pickle
import os
import re
import wordsegment
from g2p_en import G2p

from .symbols import PUNCTUATION, SYMBOLS


import unicodedata
from builtins import str as unicode
from g2p_en.expand import normalize_numbers
from nltk.tokenize import TweetTokenizer
from src.logger import logger


class DictLoader:
    def __init__(self):
        current_file_path = os.path.dirname(__file__)
        self._cmu_dict_path = os.path.join(current_file_path, "data", "english", "cmudict.rep")
        self._cmu_dict_fast_path = os.path.join(current_file_path, "data", "english", "cmudict-fast.rep")
        self._cmu_dict_hot_path = os.path.join(current_file_path, "data", "english", "engdict-hot.rep")
        self._cache_path = os.path.join(current_file_path, "data", "english", "engdict_cache.pickle")
        self._namecache_path = os.path.join(current_file_path, "data", "english", "namedict_cache.pickle")

        self.dict = self.get_dict()
        self.namedict = self.get_namedict()

    def _read_dict(self):
        g2p_dict = {}
        with open(self._cmu_dict_path) as f:
            line = f.readline()
            line_index = 1
            while line:
                if line_index >= 57:
                    line = line.strip()
                    word_split = line.split("  ")
                    word = word_split[0].lower()
                    g2p_dict[word] = [word_split[1].split(" ")]

                line_index = line_index + 1
                line = f.readline()

        with open(self._cmu_dict_fast_path) as f:
            line = f.readline()
            line_index = 1
            while line:
                if line_index >= 0:
                    line = line.strip()
                    word_split = line.split(" ")
                    word = word_split[0].lower()
                    if word not in g2p_dict:
                        g2p_dict[word] = [word_split[1:]]

                line_index = line_index + 1
                line = f.readline()

        return g2p_dict

    def _hot_reload_hot(self, g2p_dict):
        with open(self._cmu_dict_hot_path) as f:
            line = f.readline()
            line_index = 1
            while line:
                if line_index >= 0:
                    line = line.strip()
                    word_split = line.split(" ")
                    word = word_split[0].lower()
                    # Custom pronunciation words directly overwrite the dictionary
                    g2p_dict[word] = [word_split[1:]]

                line_index = line_index + 1
                line = f.readline()

        return g2p_dict

    def _cache_dict(self, g2p_dict):
        with open(self._cache_path, "wb") as pickle_file:
            pickle.dump(g2p_dict, pickle_file)

    def get_dict(self):
        if os.path.exists(self._cache_path):
            with open(self._cache_path, "rb") as pickle_file:
                g2p_dict = pickle.load(pickle_file)
        else:
            g2p_dict = self._read_dict()
            self._cache_dict(g2p_dict)

        g2p_dict = self._hot_reload_hot(g2p_dict)

        return g2p_dict

    def get_namedict(self):
        if os.path.exists(self._namecache_path):
            with open(self._namecache_path, "rb") as pickle_file:
                name_dict = pickle.load(pickle_file)
        else:
            name_dict = {}

        return name_dict


def replace_phs(phs):
    rep_map = {"'": "-"}
    phs_new = []
    for ph in phs:
        if ph in SYMBOLS:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            print("ph not in symbols: ", ph)
    return phs_new


def replace_consecutive_punctuation(text):
    punctuations = ''.join(re.escape(p) for p in PUNCTUATION)
    pattern = f'([{punctuations}])([{punctuations}])+'
    result = re.sub(pattern, r'\1', text)
    return result


def text_normalize(text):
    # todo: eng text normalize
    # Adapt to Chinese and g2p_en punctuation
    rep_map = {
        "[;:：，；]": ",",
        '["’]': "'",
        "。": ".",
        "！": "!",
        "？": "?",
    }
    for p, r in rep_map.items():
        text = re.sub(p, r, text)


    # g2p_en text formatting
    # Add uppercase compatibility
    text = unicode(text)
    text = normalize_numbers(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
    text = re.sub(r"(?i)i\.e\.", "that is", text)
    text = re.sub(r"(?i)e\.g\.", "for example", text)

    # Avoid reference leakage caused by repeated punctuation
    text = replace_consecutive_punctuation(text)

    return text


class EnglishG2p(G2p):
    def __init__(self):
        super().__init__()
        logger.info("Loading English G2P model")

        try:
            nltk.data.find("taggers/averaged_perceptron_tagger_eng")
        except LookupError:
            logger.info("Downloading nltk averaged_perceptron_tagger_eng")
            nltk.download('averaged_perceptron_tagger_eng')

        self.word_tokenize = TweetTokenizer().tokenize

        # init word segment
        wordsegment.load()

        # Expand outdated dictionary, add name dictionary
        manager = DictLoader()
        self.cmu = manager.dict
        self.namedict = manager.namedict

        # Eliminate several abbreviations with incorrect pronunciation
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        # Fix polyphonic characters
        self.homograph2features["read"] = (['R', 'IY1', 'D'], ['R', 'EH1', 'D'], 'VBP')
        self.homograph2features["complex"] = (['K', 'AH0', 'M', 'P', 'L', 'EH1', 'K', 'S'], ['K', 'AA1', 'M', 'P', 'L', 'EH0', 'K', 'S'], 'JJ')
        logger.info("Finishing loading English G2P model")

    def __call__(self, text):
        # tokenization
        words = self.word_tokenize(text)
        tokens = pos_tag(words)  # tuples of (word, tag)

        # steps
        prons = []
        for o_word, pos in tokens:
            # Restore g2p_en lowercase operation logic
            word = o_word.lower()

            if re.search("[a-z]", word) is None:
                pron = [word]
            # Push out single letters first
            elif len(word) == 1:
                # Single reading A pronunciation correction, here you need the original format o_word to determine capitalization
                if o_word == "A":
                    pron = ['EY1']
                else:
                    pron = self.cmu[word][0]
            # g2p_en original polyphonic word processing
            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                # pos1 is longer than pos and only appears in read
                elif len(pos) < len(pos1) and pos == pos1[:len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                # Recursively search for predictions
                pron = self.qryword(o_word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def qryword(self, o_word):
        word = o_word.lower()

        # Look up the dictionary, except single letters
        if len(word) > 1 and word in self.cmu:  # lookup CMU dict
            return self.cmu[word][0]

        # Search the name dictionary when only the first letter of the word is capitalized
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        # oov length is less than or equal to 3, read letters directly
        if len(word) <= 3:
            phones = []
            for w in word:
                # Pronunciation correction for single reading A, there is no capitalization here
                if w == "a":
                    phones.extend(['EY1'])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        # Try to separate possessives
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            # P T K F TH HH Silent consonant ending 's is pronounced ['S']
            if phones[-1] in ['P', 'T', 'K', 'F', 'TH', 'HH']:
                phones.extend(['S'])
            # S Z SH ZH CH JH The fricative ending 's is pronounced as ['IH1', 'Z'] or ['AH0', 'Z']
            elif phones[-1] in ['S', 'Z', 'SH', 'ZH', 'CH', 'JH']:
                phones.extend(['AH0', 'Z'])
            # B D G DH V M N NG L R W Y Voiced consonant ending 's is pronounced ['Z']
            # AH0 AH1 AH2 EY0 EY1 EY2 AE0 AE1 AE2 EH0 EH1 EH2 OW0 OW1 OW2 UH0 UH1 UH2 IY0 IY1 IY2 AA0 AA1 AA2 AO0 AO1 AO2
            # ER ER0 ER1 ER2 UW0 UW1 UW2 AY0 AY1 AY2 AW0 AW1 AW2 OY0 OY1 OY2 IH IH0 IH1 IH2 vowels ending in 's are pronounced ['Z']
            else:
                phones.extend(['Z'])
            return phones

        # Try to segment words and deal with compound words
        comps = wordsegment.segment(word.lower())

        # Send words that cannot be segmented back for prediction
        if len(comps) == 1:
            return self.predict(word)

        # Recursive processing that can be divided into words
        return [phone for comp in comps for phone in self.qryword(comp)]


_EnglishG2p = EnglishG2p()


def g2p(text):
    # g2p_en The entire reasoning, excluding non-existent arpa returns
    phone_list = _EnglishG2p(text)
    phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]

    return replace_phs(phones)


if __name__ == "__main__":
    print(g2p("hello"))
    print(g2p(text_normalize("e.g. I used openai's AI tool to draw a picture.")))
    print(g2p(text_normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.")))
