import os
import uuid
from http import HTTPStatus
from time import sleep

from fastapi import FastAPI, APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

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
from src.service.audio import AudioUVR5Params, AudioSlicerParams, AudioASRParams, AudioService, AudioDenoiseParams, AudioRefinementSubmitParams, AudioRefinementDeleteParams
from src.service.file import FileService
from src.service.namespace import NamespaceService
from src.service.normalize import NormalizeService, NormalizeParams
from src.service.session import SessionManager, async_start_session, start_train_session_with_spawn, async_stop_session
from src.service.session import session_manager
from src.service.train import do_train_gpt, do_train_sovits
from src.service.voice import VoiceCloneService
from src.train.gpt import GPTTrainParams
from src.train.helper import list_train_gpts, list_train_sovits
from src.train.sovits import SovitsTrainParams
from src.utils.response import EaseVoiceResponse, ResponseStatus


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
            endpoint=self.get_session,
            methods=["GET"],
            summary="Get current session info",
        )

    async def get_session(self):
        """Retrieve current session info."""
        session_info = self.session_manager.get_session_info()
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
        self.router.delete("/voiceclone/stop")(self.stop_service)
        self.router.get("/voiceclone/models")(self.get_available_models)
        self.router.get("/voiceclone/status")(self.get_status)

    async def get_available_models(self):
        try:
            gpts = ["default"] + list(list_train_gpts().keys())
            sovits = ["default"] + list(list_train_sovits().keys())
            return {"gpts": gpts, "sovits": sovits}
        except Exception as e:
            logger.error(f"failed to get available models: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to get available models: {e}"})

    async def get_status(self):
        if self.service is None:
            raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail={"error": "voice clone service is not started"})
        status = self.service.get_status()
        return {"status": status}

    async def start_service(self):
        uid = uuid.uuid4()
        if self.service is None:
            try:
                self.session_manager.start_session(str(uid), self._name)
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
                self.session_manager.fail_session(str(uid), msg)
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": msg})
        return {"message": "Voice Clone Service is started", "uid": str(uid)}

    async def clone(self, request: dict):
        if self.service is None:
            raise HTTPException(status_code=HTTPStatus.SERVICE_UNAVAILABLE, detail={"error": "please start the service before clone"})
        try:
            return self.service.clone(request)
        except Exception as e:
            logger.error(f"failed to clone voice for {request}, err: {e}", exc_info=True)
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to clone voice: {e}"})

    async def stop_service(self, uid: str):
        if self.service is not None:
            try:
                self.service.close()
                self.service = None
                self.session_manager.end_session(uid, "stop voice clone service")
            except Exception as e:
                msg = f"failed to stop voice clone service: {str(e)}"
                logger.error(msg)
                self.service = None
                self.session_manager.fail_session(uid, msg)
                raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": msg})
            return {"message": "Voice Clone Service is stopped"}
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail={"error": "voice clone service is not started"})


class TrainAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.post("/train/gpt/start")(self.train_gpt)
        self.router.delete("/train/gpt/stop")(self.train_gpt_stop)
        self.router.post("/train/sovits")(self.train_sovits)
        self.router.delete("/train/sovits/stop")(self.train_sovits_stop)

    async def train_gpt(self, params: GPTTrainParams, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        await async_start_session(do_train_gpt, str(uid), "TrainGPT", params=params)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "GPT training started", uid=str(uid))

    async def train_sovits(self, params: SovitsTrainParams, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})
        uid = uuid.uuid4()
        await async_start_session(do_train_sovits, str(uid), "TrainSovits", params=params)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "GPT training started", uid=str(uid))

    async def train_gpt_stop(self, uid: str):
        try:
            async_stop_session(uid, "TrainGPT")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "GPT training stopped")
        except Exception as e:
            logger.error(f"failed to stop GPT training: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop GPT training: {e}"})

    async def train_sovits_stop(self, uid: str):
        try:
            async_stop_session(uid, "TrainSovits")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Sovits training stopped")
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

    async def normalize(self, request: NormalizeParams, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        service = NormalizeService(request.output_dir)
        await async_start_session(service.normalize, str(uid), "Normalize")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Normalize started", uid=str(uid))

    async def normalize_stop(self, uid: str):
        try:
            async_stop_session(uid, "Normalize")
            return {"message": "Normalize stopped"}
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

        uid = uuid.uuid4()
        service = AudioService(source_dir=request.source_dir, output_dir=request.output_dir)
        await async_start_session(service.uvr5, str(uid), "AudioUVR5", model_name=request.model_name, audio_format=request.audio_format)

        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio UVR5 started", uid=str(uid))

    async def audio_slicer(self, request: AudioSlicerParams, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        service = AudioService(source_dir=request.source_dir, output_dir=request.output_dir)
        await async_start_session(service.slicer, str(uid), "AudioSlicer",
                                  threshold=request.threshold,
                                  min_length=request.min_length,
                                  min_interval=request.min_interval,
                                  hop_size=request.hop_size,
                                  max_silent_kept=request.max_silent_kept,
                                  normalize_max=request.normalize_max,
                                  alpha_mix=request.alpha_mix,
                                  )
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio Slicer started", uid=str(uid))

    async def audio_denoise(self, request: AudioDenoiseParams, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        service = AudioService(source_dir=request.source_dir, output_dir=request.output_dir)
        await async_start_session(service.denoise, str(uid), "AudioDenoise")
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio Denoise started", uid=str(uid))

    async def audio_asr(self, request: AudioASRParams, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        service = AudioService(source_dir=request.source_dir, output_dir=request.output_dir)
        await async_start_session(service.asr, str(uid), "AudioASR",
                                  asr_model=request.asr_model,
                                  model_size=request.model_size,
                                  language=request.language,
                                  precision=request.precision,
                                  )
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio ASR started", uid=str(uid))

    async def audio_uvr5_stop(self, uid: str):
        try:
            async_stop_session(uid, "AudioUVR5")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio UVR5 stopped")
        except Exception as e:
            logger.error(f"failed to stop Audio UVR5: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio UVR5: {e}"})

    async def audio_slicer_stop(self, uid: str):
        try:
            async_stop_session(uid, "AudioSlicer")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio Slicer stopped")
        except Exception as e:
            logger.error(f"failed to stop Audio Slicer: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio Slicer: {e}"})

    async def audio_denoise_stop(self, uid: str):
        try:
            async_stop_session(uid, "AudioDenoise")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio Denoise stopped")
        except Exception as e:
            logger.error(f"failed to stop Audio Denoise: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Audio Denoise: {e}"})

    async def audio_asr_stop(self, uid: str):
        try:
            async_stop_session(uid, "AudioASR")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Audio ASR stopped")
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

    async def easevoice(self, request: EaseVoiceRequest, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        await async_start_session(self._easevoice_easy, str(uid), "EaseVoice", request=request, uid=uid)
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "EaseVoice started", uid=str(uid))

    async def easevoice_stop(self, uid: str):
        try:
            async_stop_session(uid, "EaseVoice")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "EaseVoice stopped")
        except Exception as e:
            logger.error(f"failed to stop EaseVoice: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop EaseVoice: {e}"})

    async def _easevoice_easy(self, uid: str, request: EaseVoiceRequest):
        session_manager.update_session_info(uid, {
            "total_steps": 7,
            "current_step": 0,
            "progress": 0,
            "current_step_description": "Prepare for starting EaseVoice",
        })
        output_dir = os.path.join(request.source_dir, "output")
        audio_service = AudioService(source_dir=request.source_dir, output_dir=str(output_dir))
        resp = await audio_service.uvr5()
        response = self._check_response(uid, resp, "Audio UVR5")
        if response.status == ResponseStatus.FAILED:
            return response
        resp = await audio_service.slicer()
        response = self._check_response(uid, resp, "Audio Slicer")
        if response.status == ResponseStatus.FAILED:
            return response
        resp = await audio_service.denoise()
        response = self._check_response(uid, resp, "Audio Denoise")
        if response.status == ResponseStatus.FAILED:
            return response
        resp = await audio_service.asr()
        response = self._check_response(uid, resp, "Audio ASR")
        if response.status == ResponseStatus.FAILED:
            return response
        normalize_service = NormalizeService(processing_path=output_dir)
        resp = await normalize_service.normalize()
        response = self._check_response(uid, resp, "Normalize")
        if response.status == ResponseStatus.FAILED:
            return response
        normalize_path = response.data["normalize_path"]
        resp = await do_train_gpt(GPTTrainParams(train_input_dir=normalize_path))
        response = self._check_response(uid, resp, "Train GPT")
        if response.status == ResponseStatus.FAILED:
            return response
        resp = await do_train_sovits(SovitsTrainParams(train_input_dir=normalize_path))
        response = self._check_response(uid, resp, "Train Sovits")
        if response.status == ResponseStatus.FAILED:
            return response
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "EaseVoice completed successfully", data=response.data)

    @staticmethod
    def _check_response(uid: str, response: EaseVoiceResponse, step_name: str) -> EaseVoiceResponse:
        session = session_manager.get_session_info().get(uid, {})
        session_manager.update_session_info(uid, {
            "current_step": session.get("current_step") + 1,
        })
        if response.status == ResponseStatus.FAILED:
            session_manager.update_session_info(uid, {
                "current_step_description": f"Failed at step {session.get('current_step')} - {step_name}",
                "progress": 100,
            })
            return response

        session_manager.update_session_info(uid, {
            "current_step_description": f"Step {session.get('current_step')} - {step_name}",
            "progress": session.get("current_step") / session.get("total_steps") * 100,
        })
        return response


class TestAPI:
    def __init__(self):
        self.router = APIRouter()
        self._register_routes()

    def _register_routes(self):
        self.router.get("/test")(self.test)
        self.router.get("/test/status")(self.test_status)
        self.router.delete("/test/stop")(self.test_stop)

    async def test_stop(self, uid: str):
        try:
            async_stop_session(uid, "TEST")
            return EaseVoiceResponse(ResponseStatus.SUCCESS, "Test stopped")
        except Exception as e:
            logger.error(f"failed to stop Test: {e}")
            raise HTTPException(status_code=HTTPStatus.INTERNAL_SERVER_ERROR, detail={"error": f"failed to stop Test: {e}"})

    async def test_status(self):
        session_info = session_manager.get_session_info()
        return session_info

    async def test(self, background_tasks: BackgroundTasks):
        if session_manager.exist_running_session():
            raise HTTPException(status_code=HTTPStatus.CONFLICT, detail={"error": "There is an another task running."})

        uid = uuid.uuid4()
        background_tasks.add_task(self._do_test, uid=str(uid))
        return EaseVoiceResponse(ResponseStatus.SUCCESS, "Test started", uid=str(uid))

    def _do_test(self, uid: str):
        async_start_session(mock_long_running_task, uid, "TEST")


def mock_long_running_task():
    sleep(100)
    print("Long running task done")
    return "done"


# Initialize FastAPI and NamespaceService
app = FastAPI()

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

test_api = TestAPI()
app.include_router(test_api.router, prefix="/apis/v1")
