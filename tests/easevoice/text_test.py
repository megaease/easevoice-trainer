from typing import Optional
import unittest

from regex import F

from src.easevoice.text import cantonese, english, japanese, korean, chinese, cleaner, cleaned_text_to_sequence
from src.utils.helper import set_seed


class TestTextProcessing(unittest.TestCase):
    def _log(self, language: str, text: str, text_norm: str, phonemes: list, other: Optional[dict] = None):
        msg = f"Text process for {language}:\n\ttext: {text}\n\tnorm: {text_norm}\n\tphonemes: {phonemes}"
        for k, v in ({} if other is None else other).items():
            msg += f"\n\t{k}: {v}"
        print(msg+"\n")

    def test_cantonese(self):
        text = "佢個鋤頭太短啦。"
        text_norm = cantonese.text_normalize(text)
        phonemes, word2ph = cantonese.g2p(text_norm)

        self.assertEqual(text_norm, "佢个锄头太短啦.")
        self.assertEqual(phonemes, ['Yk', 'Yeoi5', 'Yg', 'Yo3', 'Yc', 'Yo4', 'Yt', 'Yau4', 'Yt', 'Yaai3', 'Yd', 'Yyun2', 'Yl', 'Yaa1', '.'])
        self.assertEqual(word2ph, [2, 2, 2, 2, 2, 2, 2, 1])

        self._log("yue", text, text_norm, phonemes, {"word2ph": word2ph})

    def test_english(self):
        text = "In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder."
        text_norm = english.text_normalize(text)
        phonemes = english.g2p(text_norm)

        self.assertEqual(text_norm, "In this, paper, we propose one DSPGAN, a GAN-based universal vocoder.")
        self.assertEqual(phonemes, [
            'IH0', 'N', 'DH', 'IH1', 'S', ',', 'P', 'EY1', 'P', 'ER0', ',', 'W', 'IY1', 'P', 'R',
            'AH0', 'P', 'OW1', 'Z', 'W', 'AH1', 'N', 'D', 'IY1', 'EH1', 'S', 'P', 'IY1', 'G', 'AE1',
            'N', ',', 'AH0', 'G', 'AE1', 'N', 'B', 'EY1', 'S', 'T', 'Y', 'UW2', 'N', 'AH0', 'V',
            'ER1', 'S', 'AH0', 'L', 'V', 'OW1', 'K', 'OW0', 'D', 'ER0', '.'
        ])

        self._log("en", text, text_norm, phonemes)

    def test_japanese(self):
        text = "Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！"
        text_norm = japanese.text_normalize(text)
        phonemes = japanese.g2p(text_norm)

        self.assertEqual(text_norm, "Hello.こんにちは！今日もNiCe天気ですね！tokyotowerに行きましょう！")
        self.assertEqual(phonemes, [
            'h', 'a', '[', 'r', 'o', 'o', '.', 'k', 'o', '[', 'N', 'n', 'i', 'ch', 'i', 'w', 'a', '!', 'ky', 'o', ']',
            'o', 'm', 'o', '#', 'n', 'a', '[', 'i', 's', 'u', 't', 'e', ']', 'N', 'k', 'i', 'd', 'e', 's', 'u', 'n',
            'e', '!', 't', 'o', ']', 'u', 'ky', 'o', 'o', 'z', 'u', 't', 'a', 'w', 'a', 'a', 'n', 'i', '#', 'i', '[',
            'k', 'i', 'm', 'a', 'sh', 'o', ']', 'o', '!'
        ])

        self._log("ja", text, text_norm, phonemes)

    def test_korean(self):
        text = "안녕하세요. 안녕하세요! 역시 좋은 하루야! 서울 강남구로 가자!"
        phonemes = korean.g2p(text)

        self.assertEqual(phonemes, [
            'ㅇ', 'ㅏ', 'ㄴ', 'ㄴ', 'ㅣ', 'ㅓ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅅ', 'ㅔ', 'ㅇ', 'ㅣ', 'ㅗ', '.', '空', 'ㅇ', 'ㅏ', 'ㄴ', 'ㄴ', 'ㅣ',
            'ㅓ', 'ㅇ', 'ㅎ', 'ㅏ', 'ㅅ', 'ㅔ', 'ㅇ', 'ㅣ', 'ㅗ', '!', '空', 'ㅇ', 'ㅣ', 'ㅓ', 'ㄱ', 'ㅆ', 'ㅣ', '空', 'ㅈ', 'ㅗ', 'ㅇ',
            'ㅡ', 'ㄴ', '空', 'ㅎ', 'ㅏ', 'ㄹ', 'ㅜ', 'ㅇ', 'ㅣ', 'ㅏ', '!', '空', 'ㅅ', 'ㅓ', 'ㅇ', 'ㅜ', 'ㄹ', '空', 'ㄱ', 'ㅏ', 'ㅇ',
            'ㄴ', 'ㅏ', 'ㅁ', 'ㄱ', 'ㅜ', 'ㄹ', 'ㅗ', '空', 'ㄱ', 'ㅏ', 'ㅈ', 'ㅏ', '!'
        ])

        self._log("ko", text, text, phonemes)

    def test_chinese(self):
        set_seed(100)
        text = "成熟是一种明亮而不刺眼的光辉，一种不再需要对别人察言观色的从容。"
        text_norm = chinese.text_normalize(text)
        phonemes, word2ph = chinese.g2p(text_norm)

        self.assertEqual(text_norm, "成熟是一种明亮而不刺眼的光辉,一种不再需要对别人察言观色的从容.")
        self.assertEqual(phonemes, [
            'ch', 'eng2', 'sh', 'ou2', 'sh', 'ir4', 'y', 'i4', 'zh', 'ong3', 'm', 'ing2', 'l', 'iang4',
            'EE', 'er2', 'b', 'u2', 'c', 'i04', 'y', 'En3', 'd', 'e5', 'g', 'uang1', 'h', 'ui1', ',',
            'y', 'i4', 'zh', 'ong3', 'b', 'u2', 'z', 'ai4', 'x', 'v1', 'y', 'ao4', 'd', 'ui4', 'b', 'ie2',
            'r', 'en2', 'ch', 'a2', 'y', 'En2', 'g', 'uan1', 's', 'e4', 'd', 'e5', 'c', 'ong2', 'r', 'ong2', '.'
        ])
        self.assertEqual(word2ph, [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1])

        self._log("zh", text, text_norm, phonemes, {"word2ph": word2ph})

    def test_clean_text(self):
        set_seed(100)

        text = "成熟是一种明亮而不刺眼的光辉，一种不再需要对别人察言观色的从容。"
        phonemes, word2ph, text_norm = cleaner.clean_text(text, "zh")
        phonemes = cleaned_text_to_sequence(phonemes)

        self.assertEqual(phonemes, [
            125, 146, 251, 241, 251, 214, 318, 169, 320, 237, 225, 202, 224, 184, 33, 151, 122,
            256, 124, 164, 318, 46, 127, 134, 156, 275, 158, 280, 1, 318, 169, 320, 237, 122, 256,
            319, 105, 317, 296, 318, 120, 127, 283, 122, 192, 248, 141, 125, 98, 318, 45, 156, 270,
            250, 133, 127, 134, 124, 236, 248, 236, 3
        ])

        self._log("zh", text, text_norm, phonemes, {"word2ph": word2ph})


if __name__ == "__main__":
    unittest.main()
