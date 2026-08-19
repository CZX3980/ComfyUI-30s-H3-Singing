# ComfyUI 30s H3 Singing Node

Copy this folder to `ComfyUI/custom_nodes/ComfyUI-30s-H3-Singing`, then install:

```powershell
cd ComfyUI/custom_nodes/ComfyUI-30s-H3-Singing
..\..\python_embeded\python.exe -m pip install -r requirements.txt
```

Restart ComfyUI and import `workflows/h3_singing_30s.json`.

The node has two media inputs: one singer image and one audio file at least 30 seconds long. It creates both 15-second H3 jobs in parallel, restores the original 0-15s and 15-30s audio to the respective output videos, then writes the final MP4 to ComfyUI's `output` directory.

Leave `prompt` blank to use the built-in singing prompt. This is intentional: the upstream H3 API rejects an empty prompt.
