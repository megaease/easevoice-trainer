from fastapi import FastAPI, APIRouter, HTTPException
from http import HTTPStatus

from src.api.api import (
    Namespace,
    CreateNamespaceResponse,
    UpdateNamespaceRequest,
    ListNamespaceResponse,
    CreateDirectoryRequest,
    DeleteDirectoryRequest,
    ListDirectoryRequest,
    UploadFileRequest,
    DeleteFilesRequest,
    ListDirectoryResponse,
)
from src.service.audio import AudioService
from src.service.file import FileService
from src.service.namespace import NamespaceService
from src.service.train import TrainGPTService, TrainSovitsService
from src.service.voice import VoiceCloneService
from src.service.session import SessionManager, session_guard
from src.train.gpt import GPTTrainParams
from src.train.sovits import SovitsTrainParams
from src.utils.response import EaseVoiceResponse
from src.logger import logger


class NamespaceAPI:
    """Class to encapsulate namespace-related API endpoints."""

    def __init__(self, namespace_service: NamespaceService):
        self.router = APIRouter()
        self.namespace_service = namespace_service
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""
        self.router.add_api_route(
            path="/namespaces",
            endpoint=self.list_namespaces,
            methods=["GET"],
            response_model=ListNamespaceResponse,
            summary="List all namespaces",
        )
        self.router.add_api_route(
            path="/namespaces",
            endpoint=self.new_namespace,
            methods=["POST"],
            response_model=CreateNamespaceResponse,
            summary="Create a new namespace",
        )
        self.router.add_api_route(
            path="/namespaces/{namespace_id}",
            endpoint=self.change_namespace,
            methods=["PUT"],
            response_model=Namespace,
            summary="Update a namespace",
        )
        self.router.add_api_route(
            path="/namespaces/{namespace_id}",
            endpoint=self.remove_namespace,
            methods=["DELETE"],
            status_code=204,  # No body in response
            summary="Delete a namespace",
        )

    async def list_namespaces(self):
        """List all namespaces."""
        namespaces = self.namespace_service.get_namespaces()
        return {"namespaces": namespaces}

    async def new_namespace(self):
        """Create a new namespace."""
        namespace = self.namespace_service.create_namespace()
        return namespace

    async def change_namespace(self, namespace_id: str, update_request: UpdateNamespaceRequest):
        """Update a namespace."""
        try:
            namespace = self.namespace_service.update_namespace(namespace_id, update_request.name)
            return namespace
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def remove_namespace(self, namespace_id: str):
        """Delete a namespace."""
        try:
            self.namespace_service.delete_namespace(namespace_id)
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


class SessionAPI:
    """Class to encapsulate session-related API endpoints."""

    def __init__(self, session_manager: SessionManager):
        self.router = APIRouter()
        self.session_manager = session_manager
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""
        self.router.add_api_route(
            path="/session",
            endpoint=self.get_session,
            methods=["GET"],
            summary="Get current session info",
        )

    async def get_session(self):
        """Retrieve current session info."""
        session_info = self.session_manager.get_session_info()
        if session_info is None:
            raise HTTPException(status_code=404, detail="No active session")
        return session_info


class VoiceCloneAPI:
    _name = "VoiceClone"

    def __init__(self, session_manager: SessionManager):
        self.router = APIRouter()
        self.session_manager = session_manager
        self._register_routes()

        self.service = None

    def _register_routes(self):
        self.router.post("/voiceclone/start")(self.start_service)
        self.router.post("/voiceclone/clone")(self.clone)
        self.router.get("/voiceclone/stop")(self.stop_service)
        self.router.get("/voiceclone/status")(self.get_status)

    async def get_status(self):
        if self.service is None:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail={"error": "voice clone service is not started"})
        status = self.service.get_status()
        return {"status": status}

    async def start_service(self):
        if self.service is None:
            try:
                self.session_manager.start_session(self._name)
            except Exception as e:
                msg = f"failed to start session for voice clone service: {str(e)}"
                logger.error(msg)
                raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": msg})
            try:
                self.service = VoiceCloneService()
            except Exception as e:
                msg = f"failed to start voice clone service: {str(e)}"
                logger.error(msg)
                self.service = None
                self.session_manager.fail_session(msg)
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": msg})
        return {"message": "Voice Clone Service is started"}

    async def clone(self, request: dict):
        if self.service is None:
            raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail={"error": "please start the service before clone"})
        try:
            return self.service.clone(request)
        except Exception as e:
            logger.error(f"failed to clone voice for {request}, err: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to clone voice: {e}"})

    async def stop_service(self):
        if self.service is not None:
            try:
                self.service.close()
                self.service = None
                self.session_manager.end_session("stop voice clone service")
            except Exception as e:
                msg = f"failed to stop voice clone service: {str(e)}"
                logger.error(msg)
                self.service = None
                self.session_manager.fail_session(msg)
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": msg})


class TrainAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.post("/train/gpt")(self.train_gpt)
        self.router.post("/train/sovits")(self.train_sovits)

    def train_gpt(self, params: GPTTrainParams):
        result = self._do_train_gpt(params)

        # session_guard wrapper return a dict
        if isinstance(result, EaseVoiceResponse):
            return result
        logger.error(f"failed to train gpt: {result}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result)

    async def train_sovits(self, params: SovitsTrainParams):
        result = self._do_train_sovits(params)
        
        # session_guard wrapper return a dict
        if isinstance(result, EaseVoiceResponse):
            return result
        logger.error(f"failed to train sovits: {result}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result)

    @session_guard("TrainGPT")
    def _do_train_gpt(self, params: GPTTrainParams):
        service = TrainGPTService(params)
        return service.train()

    @session_guard("TrainSovits")
    def _do_train_sovits(self, params: SovitsTrainParams):
        service = TrainSovitsService(params)
        return service.train()


# Initialize FastAPI and NamespaceService
app = FastAPI()

namespace_service = NamespaceService()
namespace_api = NamespaceAPI(namespace_service)
app.include_router(namespace_api.router, prefix="/apis/v1")

file_service = FileService()
file_api = FileAPI(file_service)
app.include_router(file_api.router, prefix="/apis/v1")

session_manager = SessionManager()
session_api = SessionAPI(session_manager)
app.include_router(session_api.router, prefix="/apis/v1")

voice_clone_api = VoiceCloneAPI(session_manager)
app.include_router(voice_clone_api.router, prefix="/apis/v1")

train_api = TrainAPI()
app.include_router(train_api.router, prefix="/apis/v1")
