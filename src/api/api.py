from pydantic import BaseModel
from typing import List
from datetime import datetime


# Task models
class Task(BaseModel):
    """Task model."""

    taskID: str
    name: str
    createdAt: datetime
    homePath: str


"""Response model for creating a new task."""
CreateTaskResponse = Task


class UpdateTaskRequest(BaseModel):
    """Request model for updating a task."""

    taskID: str
    name: str


class ListTaskResponse(BaseModel):
    """Response model for listing tasks."""

    tasks: List[Task]

# File models
class CreateDirectoryRequest(BaseModel):
    """Request model for creating a directory."""
    directoryPath: str


class DeleteDirectoryRequest(BaseModel):
    """Request model for deleting a directory."""
    directoryPath: str


class ListDirectoryRequest(BaseModel):
    """Request model for listing directory contents."""
    directoryPath: str


class UploadFileRequest(BaseModel):
    """Request model for uploading a file."""
    directoryPath: str
    fileName: str
    fileContent: str  # Base64-encoded content


class DeleteFilesRequest(BaseModel):
    """Request model for deleting files."""
    filePaths: List[str]


class FileMetadata(BaseModel):
    """Metadata for a file."""
    fileName: str
    fileSize: int
    modifiedAt: str  # RFC3339 format timestamp


class ListDirectoryResponse(BaseModel):
    """Response model for listing directory contents."""
    directoryPath: str
    files: List[FileMetadata]

# General models
class HTTPError(BaseModel):
    """Model for HTTP error response."""

    code: int
    message: str
