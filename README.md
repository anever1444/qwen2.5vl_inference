# qwen2.5vl_inference

Quick attempt at using Qwen 2.5 VL to caption a directory of mp4's. Options for using 3B (<=16GB) and 7B (>=24gb) models. Seems quite fast so long as you don't max out on vram, at which point it will offload as best it can but may oom.
```
git clone https://github.com/anever1444/qwen2.5vl_inference
cd qwen2.5vl_inference
python -m venv venv
venv\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python .\qwen_2.5vl_inference.py <directory_of_mp4s>
```

Captions will be saved in a matching .txt file. Existing captions will be overwritten.

This script will run in interactive mode by default, prompting for the model/res/fps settings. You can also specify them in the command-line, skipping the interactive prompts:

```
python .\qwen_2.5vl_inference.py <directory_of_mp4s> --model 3b --dimensions 336x336 --fps 1.0 --flash-attention
```

Flash attention is supported and helps a ton with speed and memory usage and seems to handle much longer videos. Typical flash/windows challenges, but you can try installing with:

```
pip install flash_attn --no-build-isolation
pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post9/triton-3.1.0-cp310-cp310-win_amd64.whl
```

Note: flash_attn can take up to two hours to build. For triton, switch cp310 to match your python version (python 3.11=cp311, python 3.12==cp312)
