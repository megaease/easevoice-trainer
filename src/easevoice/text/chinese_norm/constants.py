# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import string

from pypinyin.compat import SUPPORT_UCS4

# Fullwidth to halfwidth conversion
# Mapping table for fullwidth English letters -> halfwidth (num: 52)
F2H_ASCII_LETTERS = {
    ord(char) + 65248: ord(char)
    for char in string.ascii_letters
}

# Mapping table for halfwidth English letters -> fullwidth
H2F_ASCII_LETTERS = {value: key for key, value in F2H_ASCII_LETTERS.items()}

# Mapping table for fullwidth digits -> halfwidth (num: 10)
F2H_DIGITS = {ord(char) + 65248: ord(char) for char in string.digits}
# Mapping table for halfwidth digits -> fullwidth
H2F_DIGITS = {value: key for key, value in F2H_DIGITS.items()}

# Mapping table for fullwidth punctuation -> halfwidth (num: 32)
F2H_PUNCTUATIONS = {ord(char) + 65248: ord(char) for char in string.punctuation}
# Mapping table for halfwidth punctuation -> fullwidth
H2F_PUNCTUATIONS = {value: key for key, value in F2H_PUNCTUATIONS.items()}

# Space (num: 1)
F2H_SPACE = {'\u3000': ' '}
H2F_SPACE = {' ': '\u3000'}

# Regex for strings that do NOT contain "Chinese characters with pinyin", which can be used for NSW extraction
if SUPPORT_UCS4:
    RE_NSW = re.compile(r'(?:[^'
                        r'\u3007'  # 〇
                        r'\u3400-\u4dbf'  # CJK Extension A: [3400-4DBF]
                        r'\u4e00-\u9fff'  # CJK Unified Ideographs: [4E00-9FFF]
                        r'\uf900-\ufaff'  # CJK Compatibility Ideographs: [F900-FAFF]
                        r'\U00020000-\U0002A6DF'  # CJK Extension B: [20000-2A6DF]
                        r'\U0002A703-\U0002B73F'  # CJK Extension C: [2A700-2B73F]
                        r'\U0002B740-\U0002B81D'  # CJK Extension D: [2B740-2B81D]
                        r'\U0002F80A-\U0002FA1F'  # CJK Compatibility Supplement: [2F800-2FA1F]
                        r'])+')
else:
    RE_NSW = re.compile(  # pragma: no cover
        r'(?:[^'
        r'\u3007'  # 〇
        r'\u3400-\u4dbf'  # CJK Extension A: [3400-4DBF]
        r'\u4e00-\u9fff'  # CJK Unified Ideographs: [4E00-9FFF]
        r'\uf900-\ufaff'  # CJK Compatibility Ideographs: [F900-FAFF]
        r'])+')
