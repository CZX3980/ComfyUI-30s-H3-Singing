import base64
import gc
import functools
import io
import json
import shutil
import subprocess
import time
import uuid
import wave
import hashlib
import hmac
import datetime
from pathlib import Path
from urllib.parse import urljoin, urlencode

import numpy as np
import requests
from PIL import Image

try:
    import torch
except Exception:
    torch = None

try:
    from comfy.model_management import soft_empty_cache
except Exception:
    soft_empty_cache = None

import folder_paths
from comfy_api.latest import InputImpl

CLIP_TYPE = "H3_SINGING_CLIP"
KEY_TYPE = "H3_API_KEY"
CLOUD_KEY_TYPE = "CLOUD_COMFY_API"
AUTODL_KEY_TYPE = "AUTODL_API"
COMPSHARE_MEDIA_TYPE = "COMPSHARE_MEDIA_URLS"

H3_OPTIMIZER_SYSTEM = """You are a MiniMax H3 full-reference Ref2VA prompt rewriting expert. Return exactly one JSON object with only these fields in this order: subject_definitions (array), summary (string), retention_analysis (array), detailed_description (string), overall_soundscape (string), non_diegetic_music (string). No markdown, explanation, comments, or extra fields. The complete JSON, including keys, punctuation, and whitespace, MUST be no longer than 4900 characters. Shorten detailed_description first if needed, while preserving valid JSON and all six required fields. Write all six sections in English except original-language dialogue, lyrics, and visible text inside <d> tags. Use only official task prefixes: [keyframe completion], [reference generation], [video editing], [video continuation], [audio reuse], [audio reference], or [text to video].\n\nDefine every reference with stable labels <Subject N>, <Picture N>, <Video N>, and <Audio N>. Each subject definition is one line and states the label, source, role in the finished video, and concrete visible traits actually present; never invent traits. In retention_analysis write one line for every defined label, naming shot locations and using only fully_preserved, partially_preserved, attribute_transfer, weak_reference for visible references, or fully_copy, partially_copy, reference, weak_reference for audio. Do not use speaker ids in retention_analysis.\n\nDetailed_description begins with one or two English style-opening sentences before [Shot 1]. [Shot 1] has no timestamp; later shots use [Shot N] At MM:SS.mmm with timestamps below the target duration. Every shot specifies framing, subject position and appearance, environment and lighting, action/state, camera movement type and speed, current sound, and where each reference applies. Generation tasks contain 350-500 English words of information unless the 4900-character limit requires concise wording. Whenever a subject speaks, use <Subject N> (Sx) plus action or delivery plus <d>[Full English language name] exact dialogue or lyrics</d>. Assign speaker ids by first vocal event and reuse them. Simultaneous speakers use one composite id such as (S1,S2) and one <d> tag. Directly reused soundtrack lyrics use <Audio N> and no invented speaker id. Keep complete dialogue only in detailed_description; overall_soundscape contains ambience and physical sounds only; non_diegetic_music contains audience-only score details or N/A. Preserve labels and do not introduce new references after subject_definitions. Output JSON only."""
DEFAULT_PROMPT = "Use Image 1 as the singer. The singer performs naturally and lip-syncs precisely to Audio 1."

def _smart_cleanup():
    """Release Python and CUDA allocator caches without requiring CUDA."""
    gc.collect()
    if soft_empty_cache is not None:
        soft_empty_cache()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def _cleanup_after(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            _smart_cleanup()
    return wrapped

def _ffmpeg_path():
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError("FFmpeg is required. Install requirements.txt or add ffmpeg to PATH.") from exc

def _image_bytes(image):
    array = image[0].detach().cpu().clamp(0, 1).numpy()
    output = io.BytesIO()
    Image.fromarray((array * 255).round().astype(np.uint8)).save(output, format="PNG")
    return output.getvalue()

def _audio_to_wav(audio, max_duration=None):
    sample_rate = int(audio["sample_rate"])
    waveform = audio["waveform"].detach().cpu()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    duration = waveform.shape[-1] / sample_rate
    if duration < 1.8 or duration > 15.1:
        raise ValueError(f"Each H3 audio input must be 2-15 seconds; received {duration:.2f}s.")
    # Keep the encoded duration just below the provider's strict 15s limit.
    if max_duration is not None:
        max_samples = max(1, int(sample_rate * max_duration))
        waveform = waveform[..., :max_samples]
    frames = waveform.transpose(0, 1).contiguous().numpy()
    pcm = (np.clip(frames, -1, 1) * 32767).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()

def _upload(base_url, token, filename, data, content_type):
    response = requests.post(f"{base_url}/v1/files/upload", headers={"Authorization": f"Bearer {token}"}, data={"purpose": "video_generation_input"}, files={"file": (filename, data, content_type)}, timeout=600)
    if response.status_code >= 400:
        raise RuntimeError(f"MiniMax H3 file upload HTTP {response.status_code}: {response.text}")
    payload = response.json()
    base_resp = payload.get("base_resp") or {}
    if base_resp.get("status_code", 0) != 0:
        raise RuntimeError(base_resp.get("status_msg", str(base_resp)))
    file_info = payload.get("file") or {}
    if file_info.get("file_id") is not None:
        return f"mm_file://{file_info['file_id']}"
    if file_info.get("download_url"):
        return file_info["download_url"]
    raise RuntimeError(f"File upload returned no file_id or download_url: {payload}")

def _upload_temporary_public_media(filename, data, content_type):
    """Create a 72-hour public URL for CompShare's URL-only media fields."""
    errors = []
    services = (
        ("catbox", "https://litterbox.catbox.moe/resources/internals/api.php"),
        ("tmpfiles", "https://tmpfiles.org/api/v1/upload"),
    )
    for service, endpoint in services:
        for attempt in range(2):
            try:
                if service == "catbox":
                    response = requests.post(endpoint, data={"reqtype": "fileupload", "time": "72h"}, files={"fileToUpload": (filename, data, content_type)}, timeout=600)
                    url = response.text.strip()
                else:
                    response = requests.post(endpoint, files={"file": (filename, data, content_type)}, timeout=600)
                    payload = response.json()
                    url = str((payload.get("data") or {}).get("url", ""))
                    if url and "/dl/" not in url:
                        url = url.replace("tmpfiles.org/", "tmpfiles.org/dl/", 1)
                if response.status_code >= 400 or not url.startswith(("http://", "https://")):
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
                probe = requests.get(url, headers={"Range": "bytes=0-1023"}, stream=True, timeout=60, allow_redirects=True)
                status = probe.status_code
                probe.close()
                if status not in (200, 206):
                    raise RuntimeError(f"URL probe returned HTTP {status}")
                return url
            except Exception as exc:
                errors.append(f"{service} attempt {attempt + 1}: {exc}")
                if attempt == 0:
                    time.sleep(2)
    raise RuntimeError("Temporary public upload failed; CompShare cannot receive the media URL. " + " | ".join(errors))

def _cos_upload(cos, filename, data, content_type):
    """Upload a public-read object to Tencent COS using the documented v5 signature."""
    secret_id, secret_key = cos.get("secret_id", "").strip(), cos.get("secret_key", "").strip()
    bucket, region = cos.get("bucket", "").strip(), cos.get("region", "").strip()
    endpoint = cos.get("endpoint", "").strip().rstrip("/")
    if not all((secret_id, secret_key, bucket, region, endpoint)):
        raise ValueError("Tencent COS requires SecretId, SecretKey, Bucket, Region, and public Endpoint.")
    object_key = f"comfyui-h3/{uuid.uuid4().hex}_{filename}"
    path = "/" + "/".join(requests.utils.quote(part, safe="") for part in object_key.split("/"))
    host = endpoint.split("//", 1)[-1]
    now = int(time.time())
    key_time = f"{now};{now + 3600}"
    sign_key = hmac.new(secret_key.encode(), key_time.encode(), hashlib.sha1).hexdigest()
    headers = {"content-type": content_type, "host": host}
    header_list = ";".join(sorted(headers))
    # COS v5 canonical headers use ampersand-separated key/value pairs.
    format_headers = "&".join(f"{k}={requests.utils.quote(v, safe='')}" for k, v in sorted(headers.items()))
    http_string = f"put\n{path}\n\n{format_headers}\n"
    string_to_sign = f"sha1\n{key_time}\n{hashlib.sha1(http_string.encode()).hexdigest()}\n"
    signature = hmac.new(sign_key.encode(), string_to_sign.encode(), hashlib.sha1).hexdigest()
    authorization = f"q-sign-algorithm=sha1&q-ak={secret_id}&q-sign-time={key_time}&q-key-time={key_time}&q-header-list={header_list}&q-url-param-list=&q-signature={signature}"
    response = requests.put(endpoint + path, headers={**headers, "Authorization": authorization}, data=data, timeout=600)
    if response.status_code not in (200, 201, 204):
        raise RuntimeError(f"Tencent COS upload HTTP {response.status_code}: {response.text[:500]}")
    # Return a time-limited signed GET URL so third-party H3 workers can fetch
    # the object even when anonymous COS reads are filtered upstream.
    expires = now + 3600
    query = urlencode({"q-sign-algorithm": "sha1", "q-ak": secret_id, "q-sign-time": f"{now};{expires}", "q-key-time": f"{now};{expires}", "q-header-list": "", "q-url-param-list": ""})
    sign_key = hmac.new(secret_key.encode(), f"{now};{expires}".encode(), hashlib.sha1).hexdigest()
    http_string = f"get\n{path}\n\n\n"
    string_to_sign = f"sha1\n{now};{expires}\n{hashlib.sha1(http_string.encode()).hexdigest()}\n"
    signature = hmac.new(sign_key.encode(), string_to_sign.encode(), hashlib.sha1).hexdigest()
    return endpoint + path + "?" + query + f"&q-signature={signature}"

def _generate(base_url, token, image_urls, audio_url, prompt, resolution, ratio, watermark, optimize_prompt):
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"} for url in image_urls)
    content.append({"type": "audio_url", "audio_url": {"url": audio_url}, "role": "reference_audio"})
    payload = {"model": "MiniMax-H3", "content": content, "resolution": resolution, "duration": 15, "ratio": ratio, "aigc_watermark": False}
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json", "Accept": "application/json", "Idempotency-Key": f"h3-{uuid.uuid4().hex}"}
    response = requests.post(f"{base_url}/v2/video_generation", headers=headers, json=payload, timeout=600)
    if response.status_code >= 400:
        raise RuntimeError(f"MiniMax H3 video_generation HTTP {response.status_code}: {response.text}")
    task_id = response.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"Video creation returned no task_id: {response.text}. Reference URLs submitted: {image_urls + [audio_url]}")
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/v2/query/video_generation/{task_id}", headers={"Authorization": f"Bearer {token}"}, timeout=120)
        response.raise_for_status()
        task = response.json().get("task") or {}
        video_url = (task.get("content") or {}).get("url")
        if video_url:
            return video_url
        status = str(task.get("status", "")).lower()
        if status in {"failed", "fail", "cancelled", "canceled", "expired", "error"}:
            error = task.get("error") or {}
            raise RuntimeError(f"MiniMax H3 task {task_id} failed: {error.get('message', status)}")
        time.sleep(10)
    raise RuntimeError(f"MiniMax H3 task {task_id} timed out after 30 minutes.")

def _enhance_prompt(base_url, token, image_urls, audio_url, prompt, ratio):
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": url}, "role": "reference_image"} for url in image_urls)
    content.append({"type": "audio_url", "audio_url": {"url": audio_url}, "role": "reference_audio"})
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.post(f"{base_url}/v2/h3_context_ir", headers=headers, json={"model": "MiniMax-H3", "content": content, "duration": 15, "ratio": ratio}, timeout=600)
    if response.status_code >= 400:
        raise RuntimeError(f"H3-Context-IR HTTP {response.status_code}: {response.text}")
    task_id = response.json().get("task_id")
    if not task_id:
        raise RuntimeError(f"H3-Context-IR returned no task_id: {response.text}")
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        response = requests.get(f"{base_url}/v2/query/video_generation/{task_id}", headers={"Authorization": f"Bearer {token}"}, timeout=120)
        response.raise_for_status()
        task = response.json().get("task") or {}
        enhanced = (task.get("content") or {}).get("prompt")
        if enhanced:
            return enhanced
        status = str(task.get("status", "")).lower()
        if status in {"failed", "fail", "cancelled", "canceled", "expired", "error"}:
            error = task.get("error") or {}
            raise RuntimeError(f"H3-Context-IR task {task_id} failed: {error.get('message', status)}")
        time.sleep(10)
    raise RuntimeError(f"H3-Context-IR task {task_id} timed out after 30 minutes.")

def _download(url, destination):
    with requests.get(url, stream=True, timeout=600) as response:
        response.raise_for_status()
        with open(destination, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

def _data_uri(data, mime_type):
    return f"data:{mime_type};base64,{base64.b64encode(data).decode('ascii')}"

def _wav_to_mp3(wav_data):
    """Encode the short API reference segment as MP3 for broader provider compatibility."""
    temp_dir = Path(folder_paths.get_temp_directory()) / "h3_singing_uploads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    stem = uuid.uuid4().hex
    wav_path, mp3_path = temp_dir / f"{stem}.wav", temp_dir / f"{stem}.mp3"
    wav_path.write_bytes(wav_data)
    try:
        subprocess.run([_ffmpeg_path(), "-y", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)], check=True, capture_output=True)
        return mp3_path.read_bytes()
    finally:
        for path in (wav_path, mp3_path):
            try:
                path.unlink()
            except OSError:
                pass

def _fit_h3_prompt(prompt, limit=4900):
    """Keep optimized Ref2VA JSON within CompShare's 5000-character text limit."""
    prompt = str(prompt).strip()
    if len(prompt) <= limit:
        return prompt
    try:
        parsed = json.loads(prompt)
        detail = str(parsed.get("detailed_description", ""))
        parsed["detailed_description"] = detail[:max(500, limit // 2)]
        compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= limit:
            return compact
        parsed["retention_analysis"] = [str(x)[:350] for x in parsed.get("retention_analysis", [])]
        parsed["subject_definitions"] = [str(x)[:500] for x in parsed.get("subject_definitions", [])]
        parsed["summary"] = str(parsed.get("summary", ""))[:500]
        parsed["overall_soundscape"] = str(parsed.get("overall_soundscape", ""))[:400]
        parsed["non_diegetic_music"] = str(parsed.get("non_diegetic_music", ""))[:300]
        parsed["detailed_description"] = str(parsed.get("detailed_description", ""))[:1800]
        compact = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= limit:
            return compact
        # Keep the payload valid JSON even for unusually verbose optimizer output.
        fallback = {
            "subject_definitions": parsed.get("subject_definitions", []),
            "summary": parsed.get("summary", "[reference generation] A singer performs to the supplied reference audio."),
            "retention_analysis": parsed.get("retention_analysis", []),
            "detailed_description": str(parsed.get("detailed_description", ""))[:900],
            "overall_soundscape": parsed.get("overall_soundscape", "Concert ambience and physical stage sounds."),
            "non_diegetic_music": parsed.get("non_diegetic_music", "N/A"),
        }
        compact = json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))
        if len(compact) <= limit:
            return compact
        # Final deterministic valid payload; the API limit is more important than verbosity.
        fallback["detailed_description"] = "[Shot 1] A singer performs on a concert stage with precise lip-sync to the reference audio."
        return json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))[:limit] if len(json.dumps(fallback, ensure_ascii=False, separators=(",", ":"))) <= limit else json.dumps({"subject_definitions": [], "summary": "[reference generation] Concert singing performance.", "retention_analysis": [], "detailed_description": "[Shot 1] A singer performs with precise lip-sync to the reference audio.", "overall_soundscape": "Concert ambience.", "non_diegetic_music": "N/A"}, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, json.JSONDecodeError):
        return prompt[:limit]

def _find_cloud_video_url(value):
    if isinstance(value, str):
        return value if value.startswith(("http://", "https://", "/")) and (".mp4" in value or "/view" in value) else None
    if isinstance(value, dict):
        for key in ("video_url", "url", "download_url", "file_url"):
            if isinstance(value.get(key), str):
                candidate = value[key]
                if candidate.startswith(("http://", "https://", "/")):
                    return candidate
        for item in value.values():
            found = _find_cloud_video_url(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _find_cloud_video_url(item)
            if found:
                return found
    return None

def _history_video_url(result, base_url):
    task = next(iter(result.values()), {}) if isinstance(result, dict) else {}
    status = task.get("status") or {}
    if status.get("completed") is False or str(status.get("status_str", "")).lower() not in {"success", "completed", "done"}:
        return None
    outputs = task.get("outputs") or {}
    for output in outputs.values():
        if not isinstance(output, dict):
            continue
        for media_list in (output.get("gifs"), output.get("videos"), output.get("images")):
            if not isinstance(media_list, list):
                continue
            for media in media_list:
                if not isinstance(media, dict) or not media.get("filename"):
                    continue
                if media.get("filename", "").lower().endswith((".mp4", ".webm", ".mov", ".m4v")):
                    from urllib.parse import quote
                    filename = quote(str(media["filename"]))
                    subfolder = quote(str(media.get("subfolder", "")))
                    media_type = quote(str(media.get("type", "output")))
                    return f"{base_url}/api/comfy/view?filename={filename}&subfolder={subfolder}&type={media_type}"
    return None

def _cloud_get(url, params):
    """Retry transient cloud polling failures instead of failing a queued render."""
    last_error = None
    for attempt in range(4):
        try:
            return requests.get(url, params=params, timeout=(20, 120))
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Cloud result polling failed after 4 attempts: {last_error}") from last_error

def _generate_cloud_clip(config, image_values, audio_wav, prompt, width, height, steps, duration):
    input_values = {
        config["image_1_key"]: _data_uri(image_values[0], "image/png"),
        config["audio_key"]: _data_uri(audio_wav, "audio/wav"),
        config["prompt_key"]: prompt,
        "665:横竖对调": False,
        "665:预设分辨率": "自定义",
        "665:批量大小": 1,
        "665:自定义宽": width,
        "665:自定义高": height,
        "665:缩放倍数": "32",
        "728:steps": steps,
        config["duration_key"]: duration,
    }
    for key, image_data in zip((config["image_2_key"], config["image_3_key"]), image_values[1:]):
        if key and image_data is not None:
            input_values[key] = _data_uri(image_data, "image/png")
    response = requests.post(f"{config['base_url']}/api/workflow/generate", json={"workflow_id": config["workflow_id"], "input_values": input_values}, timeout=600)
    response.raise_for_status()
    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise RuntimeError(f"Cloud workflow returned no prompt_id: {response.text}")
    deadline = time.monotonic() + 1800
    use_history = False
    while time.monotonic() < deadline:
        if not use_history:
            response = _cloud_get(f"{config['base_url']}/api/workflow/result", {"prompt_id": prompt_id})
            if response.status_code == 404:
                use_history = True
                continue
            response.raise_for_status()
            result = response.json()
            video_url = _find_cloud_video_url(result)
        else:
            response = _cloud_get(f"{config['base_url']}/api/comfy/proxy/history", {"prompt_id": prompt_id})
            response.raise_for_status()
            result = response.json()
            video_url = _history_video_url(result, config["base_url"])
        if video_url:
            return urljoin(f"{config['base_url']}/", video_url)
        task = next(iter(result.values()), {}) if isinstance(result, dict) and use_history else result
        status_data = task.get("status") if isinstance(task, dict) else None
        status = str((status_data or {}).get("status_str", status_data) if isinstance(status_data, dict) else status_data or result.get("status", result.get("state", ""))).lower()
        if status in {"failed", "fail", "error", "cancelled", "canceled"}:
            raise RuntimeError(f"Cloud workflow {prompt_id} failed: {result}")
        time.sleep(5)
    raise RuntimeError(f"Cloud workflow {prompt_id} timed out after 30 minutes.")

class SplitAudioAt15Seconds:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",)}}
    RETURN_TYPES = ("AUDIO", "AUDIO")
    RETURN_NAMES = ("audio_0_15s", "audio_15_30s")
    FUNCTION = "split"
    CATEGORY = "MiniMaxH3/Utilities"
    def split(self, audio):
        rate = int(audio["sample_rate"])
        waveform = audio["waveform"]
        duration = waveform.shape[-1] / rate
        if duration < 30:
            raise ValueError(f"The input audio must be at least 30 seconds; received {duration:.2f}s.")
        if duration > 60:
            raise ValueError(f"The input audio must be no longer than 60 seconds; received {duration:.2f}s.")
        return ({"waveform": waveform[..., : rate * 15].clone(), "sample_rate": rate}, {"waveform": waveform[..., rate * 15 : rate * 30].clone(), "sample_rate": rate})

class MiniMaxH3ApiKey:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base_url": ("STRING", {"default": "https://metaso.cn/api/minimax"}), "token": ("STRING", {"default": "", "multiline": False}), "cos_secret_id": ("STRING", {"default": "", "multiline": False}), "cos_secret_key": ("STRING", {"default": "", "multiline": False}), "cos_bucket": ("STRING", {"default": ""}), "cos_region": ("STRING", {"default": "ap-guangzhou"}), "cos_endpoint": ("STRING", {"default": "https://czx-1471278445.cos.ap-guangzhou.myqcloud.com"})}}
    RETURN_TYPES = (KEY_TYPE,)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "make_key"
    CATEGORY = "MiniMaxH3/Direct"
    def make_key(self, base_url, token, cos_secret_id="", cos_secret_key="", cos_bucket="", cos_region="ap-guangzhou", cos_endpoint=""):
        base_url, token = base_url.strip().rstrip("/"), token.strip()
        if not base_url or not token:
            raise ValueError("base_url and token are required.")
        if "cp.compshare.cn" in base_url.lower() and not token.startswith("sk-ml-"):
            raise ValueError("CompShare API Key must start with sk-ml-. Please paste the model API key from the CompShare video studio.")
        cos = {"secret_id": cos_secret_id, "secret_key": cos_secret_key, "bucket": cos_bucket, "region": cos_region, "endpoint": cos_endpoint}
        return ({"base_url": base_url, "token": token, "cos": cos},)

class CloudComfyApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "base_url": ("STRING", {"default": "https://uu347213-786937da9ba7.westd.seetacloud.com:8443"}),
            "workflow_id": ("STRING", {"default": "U06-minimax_h3_lightX2v多图参考生视频V5 (2)"}),
            "image_1_key": ("STRING", {"default": "137:image"}),
            "audio_key": ("STRING", {"default": "719:audio"}),
            "prompt_key": ("STRING", {"default": "664:prompt"}),
            "image_2_key": ("STRING", {"default": ""}),
            "image_3_key": ("STRING", {"default": ""}),
            "duration_key": ("STRING", {"default": "132:value"}),
        }}
    RETURN_TYPES = (CLOUD_KEY_TYPE,)
    RETURN_NAMES = ("cloud_api",)
    FUNCTION = "configure"
    CATEGORY = "MiniMaxH3/Cloud"
    def configure(self, base_url, workflow_id, image_1_key, audio_key, prompt_key, image_2_key, image_3_key, duration_key):
        return ({"base_url": base_url.strip().rstrip("/"), "workflow_id": workflow_id.strip(), "image_1_key": image_1_key.strip(), "audio_key": audio_key.strip(), "prompt_key": prompt_key.strip(), "image_2_key": image_2_key.strip(), "image_3_key": image_3_key.strip(), "duration_key": duration_key.strip() or "132:value"},)

class AutoDLApi:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"base_url": ("STRING", {"default": "https://www.autodl.art/api/v1"}), "api_key": ("STRING", {"default": "", "multiline": False}), "model": ("STRING", {"default": "gpt-5.6-luna"})}}
    RETURN_TYPES = (AUTODL_KEY_TYPE,)
    RETURN_NAMES = ("autodl_api",)
    FUNCTION = "configure"
    CATEGORY = "MiniMaxH3/Prompt"
    def configure(self, base_url, api_key, model):
        base_url = base_url.strip().rstrip("/")
        if not base_url or not api_key.strip():
            raise ValueError("AutoDL base_url and api_key are required.")
        return ({"base_url": base_url, "api_key": api_key.strip(), "model": model.strip() or "gpt-5.6-luna"},)

class CompShareMediaUrls:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_1_url": ("STRING", {"default": ""}), "audio_url": ("STRING", {"default": ""})}, "optional": {"image_2_url": ("STRING", {"default": ""}), "image_3_url": ("STRING", {"default": ""})}}
    RETURN_TYPES = (COMPSHARE_MEDIA_TYPE,)
    RETURN_NAMES = ("media_urls",)
    FUNCTION = "configure"
    CATEGORY = "MiniMaxH3/CompShare"
    def configure(self, image_1_url, audio_url, image_2_url="", image_3_url=""):
        image_urls = [u.strip() for u in (image_1_url, image_2_url, image_3_url) if u.strip()]
        audio_url = audio_url.strip()
        if not image_urls or not audio_url or any(not u.startswith(("http://", "https://")) for u in [*image_urls, audio_url]):
            raise ValueError("CompShare requires accessible http(s) image and audio URLs.")
        return ({"image_urls": image_urls, "audio_url": audio_url},)

class AutoDLPromptOptimizer:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"autodl_api": (AUTODL_KEY_TYPE,), "user_prompt": ("STRING", {"default": "", "multiline": True})}, "optional": {"image_1": ("IMAGE",), "image_2": ("IMAGE",), "image_3": ("IMAGE",), "audio": ("AUDIO",)}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt_json",)
    FUNCTION = "optimize"
    CATEGORY = "MiniMaxH3/Prompt"
    @_cleanup_after
    def optimize(self, autodl_api, user_prompt, image_1=None, image_2=None, image_3=None, audio=None):
        _smart_cleanup()
        if not user_prompt.strip():
            raise ValueError("user_prompt is required for optimization.")
        user_content = [{"type": "text", "text": user_prompt}]
        for image in (image_1, image_2, image_3):
            if image is not None:
                user_content.append({"type": "image_url", "image_url": {"url": _data_uri(_image_bytes(image), "image/png")}})
        if audio is not None:
            audio_wav = _audio_to_wav(audio)
            user_content.append({"type": "input_audio", "input_audio": {"data": base64.b64encode(audio_wav).decode("ascii"), "format": "wav"}})
        response = requests.post(f"{autodl_api['base_url']}/chat/completions", headers={"Authorization": f"Bearer {autodl_api['api_key']}", "Content-Type": "application/json"}, json={"model": autodl_api["model"], "temperature": 0.2, "response_format": {"type": "json_object"}, "messages": [{"role": "system", "content": H3_OPTIMIZER_SYSTEM}, {"role": "user", "content": user_content}]}, timeout=600)
        if response.status_code >= 400:
            raise RuntimeError(f"AutoDL prompt optimization HTTP {response.status_code}: {response.text}")
        content = ((response.json().get("choices") or [{}])[0].get("message") or {}).get("content", "")
        content = content.strip().removeprefix("```json").removesuffix("```").strip()
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"AutoDL optimizer did not return valid JSON: {content[:500]}") from exc
        required = {"subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music"}
        if set(parsed) != required:
            raise RuntimeError(f"Optimizer JSON fields are invalid. Expected exactly: {sorted(required)}")
        _smart_cleanup()
        return (json.dumps(parsed, ensure_ascii=False),)

class CloudH3SingingClip15Seconds:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_1": ("IMAGE",), "audio": ("AUDIO",), "cloud_api": (CLOUD_KEY_TYPE,), "prompt": ("STRING", {"default": "", "multiline": True}), "duration": ("INT", {"default": 15, "min": 5, "max": 15}), "width": ("INT", {"default": 1344, "min": 256, "max": 4096, "step": 32}), "height": ("INT", {"default": 768, "min": 256, "max": 4096, "step": 32}), "steps": ("INT", {"default": 8, "min": 1, "max": 100})}, "optional": {"image_2": ("IMAGE",), "image_3": ("IMAGE",)}}
    RETURN_TYPES = (CLIP_TYPE, "STRING")
    RETURN_NAMES = ("clip", "segment_filename")
    FUNCTION = "generate"
    CATEGORY = "MiniMaxH3/Cloud"
    @_cleanup_after
    def generate(self, image_1, audio, cloud_api, prompt, duration, width, height, steps, image_2=None, image_3=None):
        _smart_cleanup()
        if not all((cloud_api["base_url"], cloud_api["workflow_id"], cloud_api["image_1_key"], cloud_api["audio_key"], cloud_api["prompt_key"])):
            raise ValueError("Cloud API configuration is incomplete.")
        wav = _audio_to_wav(audio)
        image_values = [_image_bytes(image) if image is not None else None for image in (image_1, image_2, image_3)]
        effective_prompt = prompt.strip()
        if not effective_prompt:
            raise ValueError("Optimized prompt is required; connect AutoDLPromptOptimizer output.")
        video_url = _generate_cloud_clip(cloud_api, image_values, wav, effective_prompt, width, height, steps, duration)
        clip_dir = Path(folder_paths.get_temp_directory()) / "cloud_h3_singing_clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_id = uuid.uuid4().hex
        raw_video, source_audio, segment = clip_dir / f"{clip_id}_raw.mp4", clip_dir / f"{clip_id}.wav", clip_dir / f"{clip_id}.mp4"
        _download(video_url, raw_video)
        source_audio.write_bytes(wav)
        subprocess.run([_ffmpeg_path(), "-y", "-i", str(raw_video), "-i", str(source_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(segment)], check=True, capture_output=True)
        _smart_cleanup()
        return ({"path": str(segment)}, segment.name)

class H3SingingClip15Seconds:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_1": ("IMAGE",), "audio": ("AUDIO",), "api_key": (KEY_TYPE,), "prompt": ("STRING", {"default": "", "multiline": True}), "optimize_prompt": ("BOOLEAN", {"default": False}), "resolution": (["768P", "2K"], {"default": "768P"}), "ratio": (["adaptive", "16:9", "4:3", "1:1", "3:4", "9:16", "21:9"], {"default": "adaptive"}), "watermark": ("BOOLEAN", {"default": False})}, "optional": {"image_2": ("IMAGE",), "image_3": ("IMAGE",), "sequence_gate": (CLIP_TYPE,), "compshare_media_urls": (COMPSHARE_MEDIA_TYPE,)}}
    RETURN_TYPES = (CLIP_TYPE, "STRING")
    RETURN_NAMES = ("clip", "segment_filename")
    FUNCTION = "generate"
    CATEGORY = "MiniMaxH3/Generation"
    @_cleanup_after
    def generate(self, image_1, audio, api_key, prompt, optimize_prompt, resolution, ratio, watermark, image_2=None, image_3=None, sequence_gate=None, compshare_media_urls=None):
        base_url, token = api_key["base_url"], api_key["token"]
        if "cp.compshare.cn" in base_url.lower():
            if resolution != "768P":
                raise ValueError("CompShare MiniMax H3 currently supports only 768P.")
            if watermark or optimize_prompt:
                raise ValueError("For CompShare, keep watermark and optimize_prompt disabled; use AutoDLPromptOptimizer.")
        source_wav = _audio_to_wav(audio)
        if "cp.compshare.cn" in base_url.lower():
            if compshare_media_urls is not None:
                image_urls = compshare_media_urls["image_urls"]
                audio_url = compshare_media_urls["audio_url"]
            else:
                cos = api_key.get("cos") or {}
                if not all(str(cos.get(k, "")).strip() for k in ("secret_id", "secret_key", "bucket", "region", "endpoint")):
                    raise ValueError("CompShare 公网版必须配置腾讯云 COS：请在 MiniMax H3 API Key 节点填写 COS SecretId、SecretKey、Bucket、Region 和 Endpoint。")
                uploader = lambda name, data, mime: _cos_upload(cos, name, data, mime)
                image_urls = [uploader(f"image_{index}.png", _image_bytes(image), "image/png") for index, image in enumerate((image_1, image_2, image_3), 1) if image is not None]
                api_wav = _audio_to_wav(audio, max_duration=14.8)
                api_mp3 = _wav_to_mp3(api_wav)
                audio_url = uploader("audio.mp3", api_mp3, "audio/mpeg")
        else:
            image_urls = [_upload(base_url, token, f"image_{index}.png", _image_bytes(image), "image/png") for index, image in enumerate((image_1, image_2, image_3), 1) if image is not None]
            api_wav = _audio_to_wav(audio, max_duration=14.8)
            audio_url = _upload(base_url, token, "audio.wav", api_wav, "audio/wav")
        effective_prompt = _fit_h3_prompt(prompt)
        if not effective_prompt:
            raise ValueError("Prompt is required; connect the optimized prompt instead of using a default prompt.")
        if optimize_prompt:
            effective_prompt = _enhance_prompt(base_url, token, image_urls, audio_url, effective_prompt, ratio)
        video_url = _generate(base_url, token, image_urls, audio_url, effective_prompt, resolution, ratio, watermark, False)
        clip_dir = Path(folder_paths.get_temp_directory()) / "h3_singing_clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        clip_id = uuid.uuid4().hex
        raw_video, source_audio, segment = clip_dir / f"{clip_id}_raw.mp4", clip_dir / f"{clip_id}.wav", clip_dir / f"{clip_id}.mp4"
        _download(video_url, raw_video)
        source_audio.write_bytes(source_wav)
        subprocess.run([_ffmpeg_path(), "-y", "-i", str(raw_video), "-i", str(source_audio), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(segment)], check=True, capture_output=True)
        return ({"path": str(segment)}, segment.name)

class ConcatenateH3SingingClips:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"first_clip": (CLIP_TYPE,), "second_clip": (CLIP_TYPE,)}}
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "output_filename")
    FUNCTION = "concatenate"
    CATEGORY = "MiniMaxH3/Utilities"
    OUTPUT_NODE = True
    @_cleanup_after
    def concatenate(self, first_clip, second_clip):
        _smart_cleanup()
        filename = f"h3_singing_30s_{int(time.time())}.mp4"
        output = Path(folder_paths.get_output_directory()) / filename
        subprocess.run([_ffmpeg_path(), "-y", "-i", first_clip["path"], "-i", second_clip["path"], "-filter_complex", "[0:v:0][0:a:0][1:v:0][1:a:0]concat=n=2:v=1:a=1[v][a]", "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-c:a", "aac", str(output)], check=True, capture_output=True)
        video = InputImpl.VideoFromFile(str(output))
        _smart_cleanup()
        return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}, "result": (video, filename)}

class PreviewH3SingingVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO",)}}
    RETURN_TYPES = ()
    FUNCTION = "preview"
    CATEGORY = "MiniMaxH3/Utilities"
    OUTPUT_NODE = True
    def preview(self, video):
        path = video.get_stream_source() if hasattr(video, "get_stream_source") else None
        if not path:
            raise ValueError("The merged video has no previewable file path.")
        if not isinstance(path, (str, Path)):
            raise ValueError("The merged video preview requires a file-backed video.")
        filename = Path(path).name
        output_dir = Path(folder_paths.get_output_directory()).resolve()
        resolved = Path(path).resolve()
        if resolved.parent != output_dir:
            raise ValueError("The merged video must be located in the ComfyUI output directory.")
        return {"ui": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]}}

NODE_CLASS_MAPPINGS = {"MiniMaxH3ApiKey": MiniMaxH3ApiKey, "CloudComfyApi": CloudComfyApi, "AutoDLApi": AutoDLApi, "CompShareMediaUrls": CompShareMediaUrls, "AutoDLPromptOptimizer": AutoDLPromptOptimizer, "SplitAudioAt15Seconds": SplitAudioAt15Seconds, "H3SingingClip15Seconds": H3SingingClip15Seconds, "CloudH3SingingClip15Seconds": CloudH3SingingClip15Seconds, "ConcatenateH3SingingClips": ConcatenateH3SingingClips, "PreviewH3SingingVideo": PreviewH3SingingVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3ApiKey": "MiniMax H3 API Key", "CloudComfyApi": "Cloud ComfyUI H3 API", "AutoDLApi": "AutoDL GPT-5.6-Luna API", "CompShareMediaUrls": "CompShare Media URLs", "AutoDLPromptOptimizer": "H3 Prompt Optimizer (GPT-5.6-Luna)", "SplitAudioAt15Seconds": "Split Audio (0-15s / 15-30s)", "H3SingingClip15Seconds": "MiniMax H3 Singing Clip (15s)", "CloudH3SingingClip15Seconds": "Cloud H3 Singing Clip (15s)", "ConcatenateH3SingingClips": "Concatenate H3 Singing Clips (30s)", "PreviewH3SingingVideo": "Preview H3 Singing Video"}
