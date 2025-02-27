#!/usr/bin/env python
# -*- encoding=utf8 -*-
import os.path


class Labeling(object):
    def __init__(self, source_file_path: str, language: str, text_content: str):
        self.source_file_path = source_file_path
        self.language = language
        self.text_content = text_content


class Refinement(object):
    def __init__(self, source_file_path: str, output_file_path: str):
        self.source_file_path = source_file_path
        self.output_file_path = output_file_path
        self.source_file_content: dict[str, Labeling] = dict()
        self.load_text()

    @staticmethod
    def _load_file(file_path: str) -> dict[str, Labeling]:
        if not os.path.exists(file_path):
            return dict()

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()
            labels = dict()
            for line in content:
                source_file_path, language, text_content = line.split("|")
                labels[source_file_path] = Labeling(source_file_path, language, text_content)
        return labels

    @staticmethod
    def _save_file(file_path: str, labels: dict[str, Labeling]):
        open(file_path, "w").close()
        with open(file_path, "w", encoding="utf-8") as f:
            for key, value in labels.items():
                value.text_content = value.text_content.rstrip("\n").rstrip("\r")
                f.write(f"{value.source_file_path}|{value.language}|{value.text_content}\n")

    def submit_text(self, source_file_path: str, language: str, text_content: str):
        """submit text to refinements.list"""
        self.source_file_content[source_file_path] = Labeling(source_file_path, language, text_content)
        self._save_file(self.output_file_path, self.source_file_content)

    def load_text(self) -> dict[str, Labeling]:
        """load current source content from refinements.list"""
        self.source_file_content = self._load_file(self.output_file_path)
        return self.source_file_content

    def reload_text(self):
        """reload source file content from asr.list"""
        self.source_file_content = self._load_file(self.source_file_path)
        self._save_file(self.output_file_path, self.source_file_content)
        return self.source_file_content

    def delete_text(self, source_file_path: str):
        """delete source file content from refinements.list"""
        print(self.source_file_content.keys())
        self.source_file_content.pop(source_file_path)
        self._save_file(self.output_file_path, self.source_file_content)
