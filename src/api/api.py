from pydantic import BaseModel
from typing import List
from datetime import datetime


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

    namespaceID: str
    name: str
    createdAt: datetime
    homePath: str


"""Response model for creating a new namespace."""
CreateNamespaceResponse = Namespace

class CreateNamespaceRequest(BaseModel):
    ...


class UpdateNamespaceRequest(BaseModel):
    """Request model for updating a namespace."""

    namespaceID: str
    name: str


class ListNamespaceResponse(BaseModel):
    """Response model for listing namespaces."""

    namespaces: List[Namespace]


# File models
class CreateDirectoryRequest(BaseModel):
    """Request model for creating a directory."""
    directoryPath: str


class DeleteDirectoryRequest(BaseModel):
    """Request model for deleting a directory."""
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