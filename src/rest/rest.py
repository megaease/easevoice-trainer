from fastapi import FastAPI, APIRouter, HTTPException
from service.task import TaskService
from service.file import FileService
from api.api import (
    Task,
    CreateTaskResponse,
    UpdateTaskRequest,
    ListTaskResponse,
    CreateDirectoryRequest,
    DeleteDirectoryRequest,
    ListDirectoryRequest,
    UploadFileRequest,
    DeleteFilesRequest,
    ListDirectoryResponse,
)


class TaskAPI:
    """Class to encapsulate task-related API endpoints."""

    def __init__(self, task_service: TaskService):
        self.router = APIRouter()
        self.task_service = task_service
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""
        self.router.add_api_route(
            path="/tasks",
            endpoint=self.list_tasks,
            methods=["GET"],
            response_model=ListTaskResponse,
            summary="List all tasks",
        )
        self.router.add_api_route(
            path="/tasks",
            endpoint=self.new_task,
            methods=["POST"],
            response_model=CreateTaskResponse,
            summary="Create a new task",
        )
        self.router.add_api_route(
            path="/tasks/{task_id}",
            endpoint=self.change_task,
            methods=["PUT"],
            response_model=Task,
            summary="Update a task",
        )
        self.router.add_api_route(
            path="/tasks/{task_id}",
            endpoint=self.remove_task,
            methods=["DELETE"],
            status_code=204,  # No body in response
            summary="Delete a task",
        )

    async def list_tasks(self):
        """List all tasks."""
        tasks = self.task_service.get_tasks()
        return {"tasks": tasks}

    async def new_task(self):
        """Create a new task."""
        task = self.task_service.create_task()
        return task

    async def change_task(self, task_id: str, update_request: UpdateTaskRequest):
        """Update a task."""
        try:
            task = self.task_service.update_task(task_id, update_request.name)
            return task
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def remove_task(self, task_id: str):
        """Delete a task."""
        try:
            self.task_service.delete_task(task_id)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

class FileAPI:
    """Encapsulated API logic for file operations."""

    def __init__(self, file_service: FileService):
        """Initialize the FileAPI with a router and file service."""
        self.router = APIRouter()
        self.file_service = file_service
        self._setup_routes()

    def _setup_routes(self):
        """Setup API routes."""
        self.router.post("/directories")(self.create_directory)
        self.router.delete("/directories")(self.delete_directory)
        self.router.get("/directories", response_model=ListDirectoryResponse)(self.list_directory)
        self.router.post("/files")(self.upload_file)
        self.router.delete("/files")(self.delete_files)

    async def create_directory(self, request: CreateDirectoryRequest):
        """
        Create a new directory.

        Returns:
            200: OK
            400: Bad Request
            409: Conflict (Directory already exists)
        """
        try:
            self.file_service.create_directory(request.directoryPath)
            return {"message": "Directory created successfully"}
        except ValueError as e:
            if "Conflict" in str(e):
                raise HTTPException(status_code=409, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

    async def delete_directory(self, request: DeleteDirectoryRequest):
        """
        Delete a directory.

        Returns:
            200: Succeed
            400: Bad Request
            404: Not Found (Directory does not exist)
        """
        try:
            self.file_service.delete_directory(request.directoryPath)
            return {"message": "Directory deleted successfully"}
        except ValueError as e:
            if "Not Found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

    async def list_directory(self, request: ListDirectoryRequest):
        """
        List files in a directory.

        Returns:
            200: OK
            400: Bad Request
            404: Not Found
        """
        try:
            return self.file_service.list_directory(request.directoryPath)
        except ValueError as e:
            if "Not Found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))

    async def upload_file(self, request: UploadFileRequest):
        """
        Upload a file.

        Returns:
            200: OK
            400: Bad Request
            413: File Too Large
        """
        try:
            self.file_service.upload_file(request.directoryPath, request.fileName, request.fileContent)
            return {"message": "File uploaded successfully"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def delete_files(self, request: DeleteFilesRequest):
        """
        Delete files.

        Returns:
            200: Succeed
            400: Bad Request
            404: Not Found
        """
        try:
            self.file_service.delete_files(request.filePaths)
            return {"message": "Files deleted successfully"}
        except ValueError as e:
            if "Not Found" in str(e):
                raise HTTPException(status_code=404, detail=str(e))
            raise HTTPException(status_code=400, detail=str(e))


# Initialize FastAPI and TaskService
app = FastAPI()

task_service = TaskService()
task_api = TaskAPI(task_service)
app.include_router(task_api.router, prefix="/apis/v1")

file_service = FileService()
file_api = FileAPI(file_service)
app.include_router(file_api.router, prefix="/apis/v1")

