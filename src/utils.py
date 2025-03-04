from os import path, listdir
import platform
from typing import List
import boto3
import io
import uuid
from datetime import datetime
from typing import List, Tuple
from PIL import Image

def upload_images_to_s3(
    images: List[Tuple[Image.Image, str]], 
    bucket_name: str,
    folder_prefix: str = "uploads",
    region: str = None
) -> List[str]:
    """
    Upload a list of PIL images to an S3 bucket and return their URLs.
    
    Args:
        images: List of tuples containing (PIL_Image, format_string)
                Format string examples: 'JPEG', 'PNG', etc.
        bucket_name: Name of the S3 bucket
        folder_prefix: Optional folder path within bucket
        region: AWS region (if None, will use default from boto3 config)
        
    Returns:
        List of URLs to the uploaded images
    """
    s3_client = boto3.client('s3', region_name=region)
    uploaded_urls = []
    
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    for i, (image, format_str) in enumerate(images):
        # Generate unique key
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}-{i}-{unique_id}.{format_str.lower()}"
        object_key = f"{folder_prefix}/{filename}" if folder_prefix else filename
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=format_str)
        img_byte_arr.seek(0)
        
        # Upload to S3
        s3_client.upload_fileobj(
            img_byte_arr,
            bucket_name,
            object_key,
            ExtraArgs={
                'ContentType': f'image/{format_str.lower()}',
                'ACL': 'public-read'  # Makes the object publicly readable
            }
        )
        
        # Generate the URL
        if region:
            url = f"https://{bucket_name}.s3.{region}.amazonaws.com/{object_key}"
        else:
            # If no region specified, use region-less URL format
            url = f"https://{bucket_name}.s3.amazonaws.com/{object_key}"
            
        uploaded_urls.append(url)
    
    return uploaded_urls


def show_system_info():
    try:
        print(f"Running on {platform.system()} platform")
        print(f"OS: {platform.platform()}")
        print(f"Processor: {platform.processor()}")
    except Exception as ex:
        print(f"Error occurred while getting system information {ex}")


def get_models_from_text_file(file_path: str) -> List:
    models = []
    with open(file_path, "r") as file:
        lines = file.readlines()
    for repo_id in lines:
        if repo_id.strip() != "":
            models.append(repo_id.strip())
    return models


def get_image_file_extension(image_format: str) -> str:
    if image_format == "JPEG":
        return ".jpg"
    elif image_format == "PNG":
        return ".png"


def get_files_in_dir(root_dir: str) -> List:
    models = []
    models.append("None")
    for file in listdir(root_dir):
        if file.endswith((".gguf", ".safetensors")):
            models.append(path.join(root_dir, file))
    return models
