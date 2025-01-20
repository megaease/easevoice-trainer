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
from ...logger import logger


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

    def read_dict(self):
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

    def hot_reload_hot(self, g2p_dict):
        with open(self._cmu_dict_hot_path) as f:
            line = f.readline()
            line_index = 1
            while line:
                if line_index >= 0:
                    line = line.strip()
                    word_split = line.split(" ")
                    word = word_split[0].lower()
                    # 自定义发音词直接覆盖字典
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
            g2p_dict = self.read_dict()
            self._cache_dict(g2p_dict)

        g2p_dict = self.hot_reload_hot(g2p_dict)

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
    # 适配中文及 g2p_en 标点
    rep_map = {
        "[;:：，；]": ",",
        '["’]': "'",
        "。": ".",
        "！": "!",
        "？": "?",
    }
    for p, r in rep_map.items():
        text = re.sub(p, r, text)

    # 来自 g2p_en 文本格式化处理
    # 增加大写兼容
    text = unicode(text)
    text = normalize_numbers(text)
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                   if unicodedata.category(char) != 'Mn')  # Strip accents
    text = re.sub("[^ A-Za-z'.,?!\-]", "", text)
    text = re.sub(r"(?i)i\.e\.", "that is", text)
    text = re.sub(r"(?i)e\.g\.", "for example", text)

    # 避免重复标点引起的参考泄露
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

        # 分词初始化
        wordsegment.load()

        # 扩展过时字典, 添加姓名字典
        manager = DictLoader()
        self.cmu = manager.dict
        self.namedict = manager.namedict

        # 剔除读音错误的几个缩写
        for word in ["AE", "AI", "AR", "IOS", "HUD", "OS"]:
            del self.cmu[word.lower()]

        # 修正多音字
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
            # 还原 g2p_en 小写操作逻辑
            word = o_word.lower()

            if re.search("[a-z]", word) is None:
                pron = [word]
            # 先把单字母推出去
            elif len(word) == 1:
                # 单读 A 发音修正, 这里需要原格式 o_word 判断大写
                if o_word == "A":
                    pron = ['EY1']
                else:
                    pron = self.cmu[word][0]
            # g2p_en 原版多音字处理
            elif word in self.homograph2features:  # Check homograph
                pron1, pron2, pos1 = self.homograph2features[word]
                if pos.startswith(pos1):
                    pron = pron1
                # pos1比pos长仅出现在read
                elif len(pos) < len(pos1) and pos == pos1[:len(pos)]:
                    pron = pron1
                else:
                    pron = pron2
            else:
                # 递归查找预测
                pron = self.qryword(o_word)

            prons.extend(pron)
            prons.extend([" "])

        return prons[:-1]

    def qryword(self, o_word):
        word = o_word.lower()

        # 查字典, 单字母除外
        if len(word) > 1 and word in self.cmu:  # lookup CMU dict
            return self.cmu[word][0]

        # 单词仅首字母大写时查找姓名字典
        if o_word.istitle() and word in self.namedict:
            return self.namedict[word][0]

        # oov 长度小于等于 3 直接读字母
        if len(word) <= 3:
            phones = []
            for w in word:
                # 单读 A 发音修正, 此处不存在大写的情况
                if w == "a":
                    phones.extend(['EY1'])
                elif not w.isalpha():
                    phones.extend([w])
                else:
                    phones.extend(self.cmu[w][0])
            return phones

        # 尝试分离所有格
        if re.match(r"^([a-z]+)('s)$", word):
            phones = self.qryword(word[:-2])[:]
            # P T K F TH HH 无声辅音结尾 's 发 ['S']
            if phones[-1] in ['P', 'T', 'K', 'F', 'TH', 'HH']:
                phones.extend(['S'])
            # S Z SH ZH CH JH 擦声结尾 's 发 ['IH1', 'Z'] 或 ['AH0', 'Z']
            elif phones[-1] in ['S', 'Z', 'SH', 'ZH', 'CH', 'JH']:
                phones.extend(['AH0', 'Z'])
            # B D G DH V M N NG L R W Y 有声辅音结尾 's 发 ['Z']
            # AH0 AH1 AH2 EY0 EY1 EY2 AE0 AE1 AE2 EH0 EH1 EH2 OW0 OW1 OW2 UH0 UH1 UH2 IY0 IY1 IY2 AA0 AA1 AA2 AO0 AO1 AO2
            # ER ER0 ER1 ER2 UW0 UW1 UW2 AY0 AY1 AY2 AW0 AW1 AW2 OY0 OY1 OY2 IH IH0 IH1 IH2 元音结尾 's 发 ['Z']
            else:
                phones.extend(['Z'])
            return phones

        # 尝试进行分词，应对复合词
        comps = wordsegment.segment(word.lower())

        # 无法分词的送回去预测
        if len(comps) == 1:
            return self.predict(word)

        # 可以分词的递归处理
        return [phone for comp in comps for phone in self.qryword(comp)]


EN_G2P = EnglishG2p()


def g2p(text):
    # g2p_en 整段推理，剔除不存在的arpa返回
    phone_list = EN_G2P(text)
    phones = [ph if ph != "<unk>" else "UNK" for ph in phone_list if ph not in [" ", "<pad>", "UW", "</s>", "<s>"]]

    return replace_phs(phones)


if __name__ == "__main__":
    print(g2p("hello"))
    print(g2p(text_normalize("e.g. I used openai's AI tool to draw a picture.")))
    print(g2p(text_normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder.")))
