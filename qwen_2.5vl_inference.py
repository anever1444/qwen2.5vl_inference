import sys
import os
import torch
import argparse
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
DEFAULT_WIDTH = 336
DEFAULT_HEIGHT = 336
DEFAULT_FPS = 1.0
DEFAULT_FLASH = False

MODEL_CHOICES = {
    "3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct"
}

def get_user_input(prompt, default):
    user_input = input(f"{prompt} (default: {default}): ").strip()
    return default if user_input == "" else user_input

def get_numeric_input(prompt, default, convert_func=float):
    while True:
        try:
            user_input = get_user_input(prompt, default)
            if isinstance(user_input, str):
                return convert_func(user_input)
            return user_input
        except ValueError:
            print(f"Please enter a valid number. Default is {default}")

def get_dimensions_input(prompt, default_width, default_height):
    default = f"{default_width}x{default_height}"
    while True:
        try:
            user_input = get_user_input(prompt, default)
            if user_input == default:
                return default_width, default_height
            
            if 'x' not in user_input.lower():
                raise ValueError
            
            width, height = user_input.lower().split('x')
            return int(width), int(height)
        except ValueError:
            print(f"Please enter dimensions in WxH format (e.g. {default})")

def get_yes_no_input(prompt, default=False):
    default_str = "[n]/y" if not default else "n/[y]"
    while True:
        user_input = input(f"{prompt} ({default_str}): ").strip().lower()
        if user_input == "":
            return default
        if user_input in ['y', 'yes']:
            return True
        if user_input in ['n', 'no']:
            return False
        print("Please enter 'y' or 'n'")

def select_model():
    print("\nSelect model:")
    print("3b. Qwen2.5-VL-3B-Instruct (default)")
    print("7b. Qwen2.5-VL-7B-Instruct")
    choice = get_user_input("Enter choice (3b/7b)", "3b")
    return MODEL_CHOICES.get(choice.lower(), DEFAULT_MODEL)

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def process_videos(directory, model_path, use_flash_attention, width, height, fps):
    model = None
    processor = None
    try:
        model_kwargs = {}
        print(f"\n📝 Using model: {model_path}")
        
        if use_flash_attention:
            print("🚀 Using Flash Attention 2.0 for inference")
            model_kwargs.update({
                'torch_dtype': torch.bfloat16,
                'attn_implementation': 'flash_attention_2'
            })
        else:
            print("⚡ Using default attention implementation")
        
        print(f"Dimensions: {width}x{height}")
        print(f"FPS: {fps}")
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(model_path)
        
        video_files = list(Path(directory).glob('*.mp4'))
        
        for video_file in video_files:
            print(f"Processing: {video_file.name}")
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": str(video_file),
                        "max_pixels": width * height,
                        "fps": fps,
                    },
                    {"type": "text", "text": "Describe this video in detail. When writing descriptions, focus on detailed, chronological descriptions of actions and scenes. Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph. Start directly with the action, and keep descriptions literal and precise. For best results, build your prompts using this structure: Start with main action in a single sentence. Add specific details about movements and gestures. Describe character and object appearances precisely. Include background and environment details.Specify camera angles and movements.Describe lighting and colors.Note any changes or sudden events."},
                ],
            }]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            
            inputs = processor(
                text=[text],
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
            inputs = inputs.to("cuda")
            generated_ids = model.generate(**inputs, max_new_tokens=256)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            output_caption = output_text[0]
            print(f"Caption for {video_file.name}:\n{output_caption}\n")
            
            output_file = video_file.with_suffix('.txt')
            with open(output_file, 'w') as f:
                f.write(output_caption)
                
            del inputs
            del generated_ids
            del generated_ids_trimmed
            del image_inputs
            del video_inputs
            cleanup_memory()
            
    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        cleanup_memory()

def parse_dimensions(dim_str):
    try:
        width, height = map(int, dim_str.lower().split('x'))
        return width, height
    except ValueError:
        raise argparse.ArgumentTypeError('Dimensions must be in format WxH (e.g., 336x336)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process videos using Qwen VL model')
    parser.add_argument('directory', help='Directory containing video files')
    parser.add_argument('--model', choices=['3b', '7b'], help='Model choice (3b: 3B-Instruct, 7b: 7B-Instruct)')
    parser.add_argument('--dimensions', type=parse_dimensions, help='Dimensions in WxH format (e.g., 336x336)')
    parser.add_argument('--fps', type=float, help='FPS for video processing')
    parser.add_argument('--flash-attention', action='store_true', help='Use Flash Attention 2.0')
    parser.add_argument('--no-interactive', action='store_true', help='Skip interactive prompts and use defaults for unspecified parameters')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.directory):
        print(f"Error: {args.directory} is not a valid directory")
        sys.exit(1)
    
    # Determine if we should run in interactive mode
    interactive_mode = not args.no_interactive and not all([
        args.model,
        args.dimensions,
        args.fps is not None,
        args.flash_attention is not None
    ])
    
    # Get model path
    if args.model:
        model_path = MODEL_CHOICES[args.model.lower()]
    elif interactive_mode:
        model_path = select_model()
    else:
        model_path = DEFAULT_MODEL
    
    # Get dimensions
    if args.dimensions:
        width, height = args.dimensions
    elif interactive_mode:
        width, height = get_dimensions_input("Enter dimensions (WxH)", DEFAULT_WIDTH, DEFAULT_HEIGHT)
    else:
        width, height = DEFAULT_WIDTH, DEFAULT_HEIGHT
    
    # Get FPS
    if args.fps is not None:
        fps = args.fps
    elif interactive_mode:
        fps = get_numeric_input("Enter FPS", DEFAULT_FPS, float)
    else:
        fps = DEFAULT_FPS
    
    # Get flash attention setting
    if args.flash_attention is not None:
        use_flash = args.flash_attention
    elif interactive_mode:
        use_flash = get_yes_no_input("Use Flash Attention 2.0?", DEFAULT_FLASH)
    else:
        use_flash = DEFAULT_FLASH
    
    process_videos(args.directory, model_path, use_flash, width, height, fps)