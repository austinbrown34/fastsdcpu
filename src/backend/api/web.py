import platform

import uvicorn
from frontend.webui.ui import get_web_ui
from frontend.webui.realtime_ui import demo
import gradio as gr
from backend.api.models.response import StableDiffusionResponse
from backend.models.device import DeviceInfo
from backend.base64_image import base64_image_to_pil, pil_image_to_base64_str
from backend.device import get_device_name
from backend.models.lcmdiffusion_setting import DiffusionTask, LCMDiffusionSetting
from constants import APP_VERSION, DEVICE, S3_BUCKET_NAME, S3_REGION, S3_FOLDER_PREFIX
from context import Context
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from models.interface_types import InterfaceType
from state import get_settings
from utils import upload_images_to_s3

app_settings = get_settings()
app = FastAPI(
    title="FastSD CPU",
    description="Fast stable diffusion on CPU",
    version=APP_VERSION,
    license_info={
        "name": "MIT",
        "identifier": "MIT",
    },
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)
print(app_settings.settings.lcm_diffusion_setting)

context = Context(InterfaceType.API_SERVER)


@app.get("/api/")
async def root():
    return {"message": "Welcome to FastSD CPU API"}


@app.get(
    "/api/info",
    description="Get system information",
    summary="Get system information",
)
async def info():
    device_info = DeviceInfo(
        device_type=DEVICE,
        device_name=get_device_name(),
        os=platform.system(),
        platform=platform.platform(),
        processor=platform.processor(),
    )
    return device_info.model_dump()


@app.get(
    "/api/config",
    description="Get current configuration",
    summary="Get configurations",
)
async def config():
    return app_settings.settings


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get(
    "/api/models",
    description="Get available models",
    summary="Get available models",
)
async def models():
    return {
        "lcm_lora_models": app_settings.lcm_lora_models,
        "stable_diffusion": app_settings.stable_diffsuion_models,
        "openvino_models": app_settings.openvino_lcm_models,
        "lcm_models": app_settings.lcm_models,
    }


@app.post(
    "/api/generate",
    description="Generate image(Text to image,Image to Image)",
    summary="Generate image(Text to image,Image to Image)",
)
async def generate(diffusion_config: LCMDiffusionSetting) -> StableDiffusionResponse:
    app_settings.settings.lcm_diffusion_setting = diffusion_config
    if diffusion_config.diffusion_task == DiffusionTask.image_to_image:
        app_settings.settings.lcm_diffusion_setting.init_image = base64_image_to_pil(
            diffusion_config.init_image
        )

    images = context.generate_text_to_image(app_settings.settings)
    
    if S3_BUCKET_NAME is None:
        image_urls = []
    else:
        images_with_format = [(img, "JPEG") for img in images]
        image_urls = upload_images_to_s3(
            images_with_format,
            S3_BUCKET_NAME,
            S3_FOLDER_PREFIX or "",
            S3_REGION or "us-east-1",
        )
    images_base64 = [pil_image_to_base64_str(img) for img in images]
    
    return StableDiffusionResponse(
        latency=round(context.latency, 2),
        image_urls=image_urls,
        images=images_base64,
    )

@app.get("/realtime")
async def redirect_realtime(request: Request):
    # 307 is a temporary redirect that preserves the HTTP method
    return RedirectResponse(url="/realtime/", status_code=307)

def setup_gradio_app(app_instance, path="/"):
    webui = get_web_ui()
    webui.queue()
    return gr.mount_gradio_app(app_instance, webui, path=path)


def setup_realtime_app(app_instance, path="/realtime"):
    demo.queue()
    return gr.mount_gradio_app(app_instance, demo, path=path)


def start_web_server(port: int = 8000):
    # Mount the realtime app first
    modified_app = setup_realtime_app(app)
    # Then mount the main web UI
    modified_app = setup_gradio_app(modified_app)
    
    uvicorn.run(
        modified_app,
        host="0.0.0.0",
        port=port,
    )