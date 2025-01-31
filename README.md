# qwen2.5vl_inference

Quick attempt at using Qwen 2.5 VL to caption a directory of mp4's. Options for using 3B (<=16GB) and 7B (>=24gb) models. Seems quite fast so long as you don't max out on vram, at which point it will offload but should still complete albeit much slower.
```
git clone https://github.com/anever1444/qwen2.5vl_inference
cd qwen2.5vl_inference
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt
python .\qwen_2.5vl_inference.py <directory_of_mp4s>
```

Captions will be saved in a matching .txt file. Existing captions will be overwritten.
