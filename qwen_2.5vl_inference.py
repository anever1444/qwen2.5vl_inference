import sys
import os
import torch
from pathlib import Path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_CHOICES = {
    "1": "Qwen/Qwen2.5-VL-7B-Instruct",
    "2": "Qwen/Qwen2.5-VL-3B-Instruct"
}

def select_model():
    print("\nSelect model:")
    print("1. Qwen2.5-VL-7B-Instruct")
    print("2. Qwen2.5-VL-3B-Instruct")
    choice = input("Enter choice (1/2): ").strip()
    while choice not in MODEL_CHOICES:
        choice = input("Invalid choice. Enter 1 or 2: ").strip()
    return MODEL_CHOICES[choice]

def cleanup_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def process_videos(directory, model_path):
    model = None
    processor = None
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
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
                        "max_pixels": 336 * 336,
                        "fps": 2.0,
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
                
            # Clean up CUDA memory after each video
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

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    if not os.path.isdir(directory):
        print(f"Error: {directory} is not a valid directory")
        sys.exit(1)
        
    model_path = select_model()
    process_videos(directory, model_path)