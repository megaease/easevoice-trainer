from fastapi import FastAPI, APIRouter, HTTPException
from service.task import TaskService
from api.api import Task, CreateTaskResponse, UpdateTaskRequest, ListTaskResponse


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


# Initialize FastAPI and TaskService
app = FastAPI()
task_service = TaskService()
task_api = TaskAPI(task_service)
app.include_router(task_api.router, prefix="/apis/v1")
