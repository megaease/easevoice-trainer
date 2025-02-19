from dataclasses import dataclass

from pydantic import BaseModel, Field, field_validator
from typing import List
from datetime import datetime
import re


class AudioServiceSteps:
    UVR5 = "Separating vocals using U-Net V5"
    Slicer = "Removing silent intervals and slicing audio"
    Denoise = "Denoising audio"
    ASR = "Transcribing audio to text"


class TaskStatus:
    """Status of a task."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ServiceNames:
    AUDIO = "audio"
    VOICE_CLONE = "voice_clone"


class Progress(BaseModel):
    """Progress of a task."""
    status: str = TaskStatus.PENDING
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    current_step_progress: int = 0
    message: str = ""


class AudioTaskProgressInitial(Progress):
    """Progress of an audio task."""
    status: str = TaskStatus.IN_PROGRESS
    current_step: str = AudioServiceSteps.UVR5
    total_steps: int = 5
    completed_steps: int = 0
    current_step_progress: int = 0


class VoiceCloneProgress(Progress):
    """Progress of a task."""
    status: str = TaskStatus.PENDING
    current_step: str = ""
    total_steps: int = 1
    completed_steps: int = 0
    current_step_progress: int = 0
    message: str = ""


# Namespace models
class Namespace(BaseModel):
    """Namespace model."""

    name: str
    createdAt: datetime
    homePath: str


"""Response model for creating a new namespace."""
CreateNamespaceResponse = Namespace

class CreateNamespaceRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=64)

    @field_validator("name")
    @classmethod
    def validate_name(cls, name: str):
        # Ensure it doesn't contain invalid characters
        if "/" in name or "\0" in name:
            raise ValueError("Namespace name cannot contain '/' or null characters")

        # Optional: Avoid names with only dots ('.' or '..')
        if name in {".", ".."}:
            raise ValueError("Namespace name cannot be '.' or '..'")

        # Optional: Ensure it consists of valid filename characters
        if not re.match(r"^[\w.-]+$", name):  # Allows letters, numbers, underscores, dashes, and dots
            raise ValueError("Namespace name contains invalid characters")

        return name


class UpdateNamespaceRequest(BaseModel):
    """Request model for updating a namespace."""

    name: str


class ListNamespaceResponse(BaseModel):
    """Response model for listing namespaces."""

    namespaces: List[Namespace]


# File models
class CreateDirectoryRequest(BaseModel):
    """Request model for creating a directory."""
    directoryPath: str


class DeleteDirsFilesRequest(BaseModel):
    """Request model for deleting directores or files."""
    paths: List[str]


class UploadFileRequest(BaseModel):
    """Request model for uploading a file."""
    directoryPath: str
    fileName: str
    fileContent: str  # Base64-encoded content


class FileMetadata(BaseModel):
    """Metadata for a file."""
    type: str
    fileName: str
    fileSize: int
    modifiedAt: str  # RFC3339 format timestamp

class DirMetadata(BaseModel):
    """Metadata for a directory."""
    type: str
    directoryName: str

class ListDirectoryResponse(BaseModel):
    """Response model for listing directory contents."""
    directoryPath: str
    files: List[FileMetadata]
    directories: List[DirMetadata]


# General models
class HTTPError(BaseModel):
    """Model for HTTP error response."""

    code: int
    message: str

# Task models

class CreateTaskRequest(BaseModel):
    """Request model for creating a new namespace."""
    service_name: str
    args: dict


@dataclass
class EaseVoiceRequest:
    """Request model for creating a new namespace."""
    source_dir: str
