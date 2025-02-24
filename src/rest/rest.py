import os
import uuid
from dataclasses import asdict
from http import HTTPStatus

from fastapi import FastAPI, APIRouter, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from typing import AsyncGenerator

from src.api.api import (
    Namespace,
    CreateNamespaceRequest,
    CreateNamespaceResponse,
    UpdateNamespaceRequest,
    ListNamespaceResponse,
    CreateDirectoryRequest,
    DeleteDirsFilesRequest,
    UploadFileRequest,
    ListDirectoryResponse, EaseVoiceRequest,
)
from src.logger import logger
from src.rest.types import TaskType, TaskCMD
from src.service.audio import AudioUVR5Params, AudioSlicerParams, AudioASRParams, AudioService, AudioDenoiseParams, AudioRefinementSubmitParams, AudioRefinementDeleteParams
from src.service.file import FileService
from src.service.namespace import NamespaceService
from src.service.normalize import NormalizeParams
from src.service.session import SessionManager, backtask_with_session_guard, start_task_with_subprocess, stop_task_with_subprocess
from src.service.session import session_manager
from src.service.voice import VoiceCloneService
from src.service.tensorboard import TensorBoardService
from src.train.gpt import GPTTrainParams
from src.train.helper import list_train_gpts, list_train_sovits, generate_random_name, get_gpt_train_dir, get_sovits_train_dir
from src.train.sovits import SovitsTrainParams
from src.utils.helper import random_choice
from src.utils.response import EaseVoiceResponse, ResponseStatus


class FrontendAssetsAPI:
    """Class to handle serving static assets (JS, CSS, images) from /assets."""

    def __init__(self, frontend_dir: str):
        self.router = APIRouter()
        self.frontend_dir = frontend_dir
        self._register_routes()

    def _register_routes(self):
        """Register route to serve static assets."""
        # Handle assets manually for `/assets/*`
        self.router.add_api_route("/assets/{file_path:path}", self.serve_asset, methods=["GET"])

    async def serve_asset(self, file_path: str) -> Response:
        """Serve a static file from the `dist/assets` directory."""
        asset_path = os.path.join(self.frontend_dir, "assets", file_path)

        # Check if the file exists
        if not os.path.exists(asset_path):
            raise HTTPException(status_code=404, detail=f"Asset '{file_path}' not found")

        # Return the file
        return FileResponse(asset_path)

class FrontendIndexAPI:
    """Class to handle serving index.html for the root path."""

    def __init__(self, frontend_dir: str):
        self.router = APIRouter()
        self.frontend_dir = frontend_dir
        self._register_routes()

    def _register_routes(self):
        """Register routes to serve index.html."""
        # Serve index.html for any route that is not `/assets`
        self.router.add_api_route("/", self.serve_index, methods=["GET"])
        #self.router.add_api_route("/{path:path}", self.serve_index, methods=["GET"])

    async def serve_index(self, path: str = "") -> Response:
        """Serve index.html for any request (except /assets)."""
        # If the request is for /assets, let the FrontendAssetsAPI handle it
        if path.startswith("assets/"):
            return Response(status_code=404)

        index_path = os.path.join(self.frontend_dir, "index.html")

        if not os.path.exists(index_path):
            return Response("index.html not found.", status_code=404)

        # Read and return the index.html file
        with open(index_path, "r") as f:
            content = f.read()

        return Response(content=content, media_type="text/html")


class TensorBoardAPI:
    """Class to encapsulate TensorBoard-related API endpoints."""

    def __init__(self, log_dir: str):
        self.router = APIRouter()
        self.log_dir = log_dir
        self._register_routes()

    def _register_routes(self):
        """Register API routes."""
        self.router.add_api_route(
            path="/tensorboard",
            endpoint=self.view_tensorboard,
            methods=["GET"],
            response_class=HTMLResponse,
            summary="View TensorBoard UI",
        )

    async def view_tensorboard(self):
        """Serve the TensorBoard UI in an iframe."""
        return f'''
        <html>
            <body>
                <iframe src="http://localhost:6006" width="100%" height="1000px"></iframe>
            </body>
        </html>
        '''


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
            path="/namespaces/{name}",
            endpoint=self.change_namespace,
            methods=["PUT"],
            response_model=Namespace,
            summary="Update a namespace",
        )
        self.router.add_api_route(
            path="/namespaces/{name}",
            endpoint=self.remove_namespace,
            methods=["DELETE"],
            status_code=204,  # No body in response
            summary="Delete a namespace",
        )

    async def list_namespaces(self):
        """List all namespaces."""
        namespaces = self.namespace_service.get_namespaces()
        return {"namespaces": namespaces}

    async def new_namespace(self, create_request: CreateNamespaceRequest):
        """Create a new namespace."""
        try:
            return self.namespace_service.create_namespace(create_request.name)
        except FileExistsError:
            raise HTTPException(status_code=409, detail="Namespace already exists")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def change_namespace(self, name: str, update_request: UpdateNamespaceRequest):
        """Update a namespace."""
        try:
            return self.namespace_service.update_namespace(name, update_request.name)
        except FileExistsError:
            raise HTTPException(status_code=409, detail="Namespace already exists")
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

    async def remove_namespace(self, name: str):
        """Delete a namespace."""
        try:
            self.namespace_service.delete_namespace(name)
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
        self.router.get("/directories", response_model=ListDirectoryResponse)(self.list_directory)
        self.router.post("/files")(self.upload_file)
        self.router.get("/files", response_class=FileResponse)(self.download_file)
        self.router.post("/delete-dirs-files")(self.delete_dirs_files)

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

    async def delete_dirs_files(self, request: DeleteDirsFilesRequest):
        """
        Delete directories or files.

        Returns:
            200: Succeed
            400: Bad Request
        """
        result = self.file_service.delete_dirs_files(request.paths)
        if result["hasFailure"] is True:
            print(result)
        return result

    async def list_directory(self, directoryPath: str):
        """
        List files in a directory.

        Returns:
            200: OK
            400: Bad Request
            404: Not Found
        """
        try:
            return self.file_service.list_directory(directoryPath)
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

    async def download_file(self, filePath: str):
        """
        Download a file.

        Returns:
            200: OK
            400: Bad Request
            404: Not Found
        """
        if not os.path.exists(filePath):
            raise HTTPException(status_code=404, detail="File not found")
        if os.path.isdir(filePath):
            raise HTTPException(status_code=400, detail="Path is a directory, not a file")

        return FileResponse(filePath, filename=os.path.basename(filePath))


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
            endpoint=self.get_sessions,
            methods=["GET"],
            summary="List sessions info",
        )
        self.router.add_api_route(
            path="/session/current",
            endpoint=self.get_current_session,
            methods=["GET"],
            summary="Get current session info",
        )

    async def get_sessions(self):
        """Retrieve current session info."""
        session_info = self.session_manager.get_session_info()
        return session_info

    async def get_current_session(self):
        return self.session_manager.get_current_session_info()


class VoiceCloneAPI:
    _name = "VoiceClone"

    def __init__(self, session_manager: SessionManager):
        self.router = APIRouter()
        self.session_manager = session_manager
        self._register_routes()

        self.service = None

    def _register_routes(self):
        self.router.post("/voiceclone/clone")(self.clone)
        self.router.get("/voiceclone/models")(self.get_available_models)

    async def get_available_models(self):
        try:
            gpts = ["default"] + list(list_train_gpts().keys())
            sovits = ["default"] + list(list_train_sovits().keys())
            return {"gpts": gpts, "sovits": sovits}
        except Exception as e:
            logger.error(f"failed to get available models: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to get available models: {e}"})

    async def clone(self, request: dict):
        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.voice_clone, request, VoiceCloneAPI._do_clone, uid=uid, task=request)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Voice clone started", uuid=uid)

    @staticmethod
    def _do_clone(uid: str, task: dict):
        logger.info(f"Start to clone voice {uuid} for {task}")
        service = None
        try:
            session_manager.update_session_info(uid, {"message": "start to load voice clone model"})
            service = VoiceCloneService(session_manager)
            session_manager.update_session_info(uid, {"message": "voice clone model loaded"})

            result = service.clone(uid, task)
        except Exception as e:
            logger.error(f"Failed to clone voice for {task}: {e}", exc_info=True)
            result = EaseVoiceResponse(ResponseStatus.FAILED, str(e))
        finally:
            session_manager.end_session_with_ease_voice_response(uid, result)
            if service is not None:
                service.close()
        return result


class TrainAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.post("/train/gpt/start")(self.train_gpt)
        self.router.delete("/train/gpt/stop")(self.train_gpt_stop)
        self.router.post("/train/sovits/start")(self.train_sovits)
        self.router.delete("/train/sovits/stop")(self.train_sovits_stop)

    async def train_gpt(self, params: GPTTrainParams):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = str(uuid.uuid4())
        # Note: it could not be empty, because the training processing has multiple processes, each process could generate a model name
        if params.output_model_name == "":
            params.output_model_name = "gpt_" + generate_random_name()
        model_path = get_gpt_train_dir(params.output_model_name)

        backtask_with_session_guard(uid, TaskType.train_gpt, asdict(params), start_task_with_subprocess, uid=uid, request=params, cmd_file=TaskCMD.train_gpt)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "GPT training started", uuid=str(uid), data={"model_path": model_path})

    async def train_sovits(self, params: SovitsTrainParams):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})
        # Note: it could not be empty, because the training processing has multiple processes, each process could generate a model name
        if params.output_model_name == "":
            params.output_model_name = "sovits_" + generate_random_name()
        model_path = get_sovits_train_dir(params.output_model_name)
        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.train_sovits, asdict(params), start_task_with_subprocess, uid=uid, request=params, cmd_file=TaskCMD.tran_sovits)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Sovits training started", uuid=uid, data={"model_path": model_path})

    async def train_gpt_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.train_gpt)
        except Exception as e:
            logger.error(f"failed to stop GPT training: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop GPT training: {e}"})

    async def train_sovits_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.train_sovits)
        except Exception as e:
            logger.error(f"failed to stop Sovits training: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Sovits training: {e}"})


class NormalizeAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.post("/normalize/start")(self.normalize)
        self.router.delete("/normalize/stop")(self.normalize_stop)

    async def normalize(self, request: NormalizeParams):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = str(uuid.uuid4())
        request.predefined_output_path = random_choice()
        backtask_with_session_guard(uid, TaskType.normalize, asdict(request), start_task_with_subprocess, uid=uid, request=request, cmd_file=TaskCMD.normalize)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Normalize started", uuid=str(uid), data={"normalize_path": str(os.path.join(request.output_dir, request.predefined_output_path))})

    async def normalize_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.normalize)
        except Exception as e:
            logger.error(f"failed to stop Normalize: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Normalize: {e}"})


class AudioAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.post("/audio/uvr5/start")(self.audio_uvr5)
        self.router.delete("/audio/uvr5/stop")(self.audio_uvr5_stop)
        self.router.post("/audio/slicer/start")(self.audio_slicer)
        self.router.delete("/audio/slicer/stop")(self.audio_slicer_stop)
        self.router.post("/audio/denoise/start")(self.audio_denoise)
        self.router.delete("/audio/denoise/stop")(self.audio_denoise_stop)
        self.router.post("/audio/asr/start")(self.audio_asr)
        self.router.delete("/audio/asr/stop")(self.audio_asr_stop)
        self.router.get("/audio/refinement")(self.list_audio_refinement)
        self.router.post("/audio/refinement")(self.update_audio_refinement)
        self.router.delete("/audio/refinement")(self.delete_audio_refinement)

    async def audio_uvr5(self, request: AudioUVR5Params):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.audio_uvr5, asdict(request), start_task_with_subprocess, uid=uid, request=request, cmd_file=TaskCMD.audio_uvr5)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio UVR5 started", uuid=str(uid))

    async def audio_uvr5_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.audio_uvr5)
        except Exception as e:
            logger.error(f"failed to stop Audio UVR5: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio UVR5: {e}"})

    async def audio_slicer(self, request: AudioSlicerParams):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.audio_slicer, asdict(request), start_task_with_subprocess, uid=uid, request=request, cmd_file=TaskCMD.audio_slicer)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio Slicer started", uuid=str(uid))

    async def audio_slicer_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.audio_slicer)
        except Exception as e:
            logger.error(f"failed to stop Audio Slicer: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio Slicer: {e}"})

    async def audio_denoise(self, request: AudioDenoiseParams):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.audio_denoise, asdict(request), start_task_with_subprocess, uid=uid, request=request, cmd_file=TaskCMD.audio_denoise)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio Denoise started", uuid=str(uid))

    async def audio_denoise_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.audio_denoise)
        except Exception as e:
            logger.error(f"failed to stop Audio Denoise: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio Denoise: {e}"})

    async def audio_asr(self, request: AudioASRParams):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.audio_asr, asdict(request), start_task_with_subprocess, uid=uid, request=request, cmd_file=TaskCMD.audio_asr)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio ASR started", uuid=str(uid))

    async def audio_asr_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.audio_asr)
        except Exception as e:
            logger.error(f"failed to stop Audio ASR: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio ASR: {e}"})

    def list_audio_refinement(self, input_dir: str, output_dir: str):
        service = AudioService(source_dir=input_dir, output_dir=output_dir)
        result = service.refinement_reload()
        if isinstance(result, EaseVoiceResponse):
            return result
        logger.error(f"failed to list audio refinement: {result}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result)

    def update_audio_refinement(self, request: AudioRefinementSubmitParams):
        service = AudioService(source_dir=request.source_dir, output_dir=request.output_dir)
        result = service.refinement_submit_text(request.source_file_path, request.language, request.text_content)
        if isinstance(result, EaseVoiceResponse):
            return result
        logger.error(f"failed to update audio refinement: {result}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result)

    def delete_audio_refinement(self, params: AudioRefinementDeleteParams):
        service = AudioService(source_dir=params.source_dir, output_dir=params.output_dir)
        result = service.refinement_delete_text(params.source_file_path)
        if isinstance(result, EaseVoiceResponse):
            return result
        logger.error(f"failed to delete audio refinement: {result}")
        raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail=result)


class EaseVoiceAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.post("/easevoice/start")(self.easevoice)
        self.router.delete("/easevoice/stop")(self.easevoice_stop)

    async def easevoice(self, request: EaseVoiceRequest):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        gpt_output = "gpt_" + generate_random_name()
        sovits_output = "sovits_" + generate_random_name()
        request.gpt_output_name = gpt_output
        request.sovits_output_name = sovits_output

        uid = str(uuid.uuid4())
        backtask_with_session_guard(uid, TaskType.ease_voice, asdict(request), start_task_with_subprocess, uid=uid, request=request, cmd_file=TaskCMD.ease_voice)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "EaseVoice started", uuid=str(uid), data={"sovits_output": sovits_output, "gpt_output": gpt_output})

    async def easevoice_stop(self, uid: str):
        try:
            return stop_task_with_subprocess(uid, TaskType.ease_voice)
        except Exception as e:
            logger.error(f"failed to stop EaseVoice: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop EaseVoice: {e}"})


# TensorBoard setup
async def lifespan_context(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for application startup and shutdown."""
    # Start TensorBoard as a background task when the app starts
    tensorboard_service.start()
    yield  # The application is running here
    # Stop TensorBoard when the app shuts down
    tensorboard_service.stop()


# FastAPI app setup
app = FastAPI(lifespan=lifespan_context)  # pyright: ignore

frontend_dir = os.path.join(os.getcwd(), "dist")
if not os.path.exists(frontend_dir):
    raise FileNotFoundError(f"Frontend build directory '{frontend_dir}' not found. Please build the frontend first.")

frontend_assets_api = FrontendAssetsAPI(frontend_dir)
app.include_router(frontend_assets_api.router)

frontend_index_api = FrontendIndexAPI(frontend_dir)
app.include_router(frontend_index_api.router)


tb_log_dir = "tb_logs"
tensorboard_service = TensorBoardService(tb_log_dir)
tensorboard_api = TensorBoardAPI(tb_log_dir)
app.include_router(tensorboard_api.router, prefix="/apis/v1")

namespace_service = NamespaceService()
namespace_api = NamespaceAPI(namespace_service)
app.include_router(namespace_api.router, prefix="/apis/v1")

file_service = FileService()
file_api = FileAPI(file_service)
app.include_router(file_api.router, prefix="/apis/v1")

session_api = SessionAPI(session_manager)
app.include_router(session_api.router, prefix="/apis/v1")

voice_clone_api = VoiceCloneAPI(session_manager)
app.include_router(voice_clone_api.router, prefix="/apis/v1")

train_api = TrainAPI()
app.include_router(train_api.router, prefix="/apis/v1")

normalize_api = NormalizeAPI()
app.include_router(normalize_api.router, prefix="/apis/v1")

audio_api = AudioAPI()
app.include_router(audio_api.router, prefix="/apis/v1")

easevoice_api = EaseVoiceAPI()
app.include_router(easevoice_api.router, prefix="/apis/v1")


# Function to print all routing information
# def print_routes(app: FastAPI):
#     for route in app.routes:
#         print(f"Path: {route.path}")
#         print(f"Methods: {route.methods}")
#         print(f"Endpoint: {route.endpoint.__name__}")
#         print("-" * 40)
#
# # Call the function to print all routes
# print_routes(app)