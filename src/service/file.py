import os
import base64
from datetime import datetime
from typing import List, Dict


class FileService:

    def __init__(self):
        pass

    def create_directory(self, directory_path: str):
        pass

    def delete_directory(self, directory_path: str):
        pass

    def list_directory(self, directory_path: str) -> Dict[str, List[Dict[str, str]]]:
        pass

    def upload_file(self, directory_path: str, file_name: str, file_content: str):
        pass

    def delete_files(self, file_paths: List[str]):
        pass
