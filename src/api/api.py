from pydantic import BaseModel
from typing import List
from datetime import datetime


# API Models
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


class HTTPError(BaseModel):
    """Model for HTTP error response."""

    code: int
    message: str
