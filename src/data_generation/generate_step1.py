"""
Step1 Data Generation Script
Generate task proposals from scene images using Gemini 2.5 Pro
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image


class Step1Generator:
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.0-flash-exp",
        prompt_path: str = "step1.prompt"
    ):
        """
        Args:
            api_key: Google API key, if None will read from environment
            model: Model name to use (gemini-2.5-flash)
            prompt_path: Path to step1 prompt file
        """
        # Configure Gemini
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
        
        # Load prompt
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt = f.read()
    
    def generate_tasks(self, image_path: str) -> Dict:
        """
        Generate task proposals for a single image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing tasks or error information
        """
        try:
            # Load image
            image = Image.open(image_path)
            
            # Call Gemini
            response = self.model.generate_content(
                [self.prompt, image],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 500,
                }
            )
            
            # Parse response
            content = response.text
            tasks_data = json.loads(content)
            
            return {
                "success": True,
                "image_path": image_path,
                "tasks": tasks_data.get("tasks", []),
                "raw_response": content
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "image_path": image_path,
                "error": "JSON parsing failed",
                "raw_response": content if 'content' in locals() else None
            }
        except Exception as e:
            return {
                "success": False,
                "image_path": image_path,
                "error": str(e)
            }
    
    def batch_generate(
        self,
        image_dir: str,
        output_path: str,
        max_images: int = None
    ) -> List[Dict]:
        """
        Batch generate tasks for multiple images
        
        Args:
            image_dir: Directory containing images
            output_path: Path to save output JSON
            max_images: Maximum number of images to process
            
        Returns:
            List of generation results
        """
        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png"}
        image_files = [
            str(p) for p in Path(image_dir).rglob("*")
            if p.suffix.lower() in image_extensions
        ]
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} images to process")
        
        # Generate tasks
        results = []
        success_count = 0
        
        for image_path in tqdm(image_files, desc="Generating Step1 data"):
            result = self.generate_tasks(image_path)
            results.append(result)
            
            if result["success"]:
                success_count += 1
        
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\nGeneration complete:")
        print(f"  Total: {len(results)}")
        print(f"  Success: {success_count}")
        print(f"  Failed: {len(results) - success_count}")
        print(f"  Output saved to: {output_path}")
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Step1 task proposals")
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft_step1.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="step1.prompt",
        help="Path to step1 prompt file"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model to use (gemini-2.0-flash-exp or gemini-1.5-pro)"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = Step1Generator(
        model=args.model,
        prompt_path=args.prompt
    )
    
    # Run generation
    generator.batch_generate(
        image_dir=args.image_dir,
        output_path=args.output,
        max_images=args.max_images
    )


if __name__ == "__main__":
    main()

