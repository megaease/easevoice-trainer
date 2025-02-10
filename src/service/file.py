import os
import base64
from datetime import datetime
from typing import List, Dict


class FileService:
    """Service class for managing directories and files."""

    def __init__(self):
        """Initialize the FileService."""
        pass

    def create_directory(self, directory_path: str):
        """
        Create a new directory.

        Args:
            directory_path (str): Path of the directory to create.

        Raises:
            ValueError: If the directory already exists or the path is invalid.
        """
        if os.path.exists(directory_path):
            raise ValueError("Conflict: Directory already exists.")
        try:
            os.makedirs(directory_path)
        except Exception as e:
            raise ValueError(f"Bad Request: Unable to create directory. {str(e)}")

    def delete_directory(self, directory_path: str):
        """
        Delete a directory.

        Args:
            directory_path (str): Path of the directory to delete.

        Raises:
            ValueError: If the directory does not exist or the path is invalid.
        """
        if not os.path.exists(directory_path):
            raise ValueError("Not Found: Directory does not exist.")
        if not os.path.isdir(directory_path):
            raise ValueError("Bad Request: Path is not a directory.")
        try:
            for root, dirs, files in os.walk(directory_path, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(directory_path)
        except Exception as e:
            raise ValueError(f"Bad Request: Unable to delete directory. {str(e)}")

    def list_directory(self, directory_path: str) -> Dict[str, List[Dict[str, str]]]:
        """
        List files and directories in a specified directory.

        Args:
            directory_path (str): Path of the directory to list.

        Returns:
            Dict[str, List[Dict[str, str]]]: Directory information with file and directory details.

        Raises:
            ValueError: If the directory does not exist or the path is not a directory.
        """
        if not os.path.exists(directory_path):
            raise ValueError("Not Found: Directory does not exist.")
        if not os.path.isdir(directory_path):
            raise ValueError("Bad Request: Path is not a directory.")

        files = []
        directories = []

        for entry in os.scandir(directory_path):
            entry_info = {
                "type": "directory" if entry.is_dir() else "file"
            }

            if entry.is_dir():
                entry_info.update({
                    "directoryName": entry.name
                })
                directories.append(entry_info)
            elif entry.is_file():
                stat = entry.stat()
                entry_info.update({
                    "fileName": entry.name,
                    "fileSize": stat.st_size,
                    "modifiedAt": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
                files.append(entry_info)

        return {
            "directoryPath": directory_path,
            "files": files,
            "directories": directories
        }

    def upload_file(self, directory_path: str, file_name: str, file_content: str):
        """
        Upload a file to a directory.

        Args:
            directory_path (str): Path of the directory to upload to.
            file_name (str): Name of the file.
            file_content (str): Base64-encoded content of the file.

        Raises:
            ValueError: If the directory does not exist or the file size is too large.
        """
        if not os.path.exists(directory_path):
            raise ValueError("Bad Request: Directory does not exist.")
        if not os.path.isdir(directory_path):
            raise ValueError("Bad Request: Path is not a directory.")

        try:
            file_path = os.path.join(directory_path, file_name)
            with open(file_path, "wb") as file:
                file.write(base64.b64decode(file_content))
        except Exception as e:
            raise ValueError(f"Bad Request: Unable to upload file. {str(e)}")

    def delete_files(self, file_paths: List[str]):
        """
        Delete specified files.

        Args:
            file_paths (List[str]): List of file paths to delete.

        Raises:
            ValueError: If a file does not exist or the path is invalid.
        """
        errors = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                errors.append(f"Not Found: {file_path}")
                continue
            if not os.path.isfile(file_path):
                errors.append(f"Bad Request: {file_path} is not a file.")
                continue
            try:
                os.remove(file_path)
            except Exception as e:
                errors.append(f"Unable to delete {file_path}. {str(e)}")
        if errors:
            raise ValueError("; ".join(errors))
