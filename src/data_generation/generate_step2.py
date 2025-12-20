"""
Step2 Data Generation Script
Generate action plans from images and tasks using Gemini 2.5 Pro
"""

import os
import json
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
import google.generativeai as genai
from PIL import Image

from src.utils.logger import get_logger

logger = get_logger(__name__)



class Step2Generator:
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.0-flash-exp",
        prompt_path: str = "step2.prompt"
    ):
        """
        Args:
            api_key: Google API key, if None will read from environment
            model: Model name to use (gemini-2.5-flash)
            prompt_path: Path to step2 prompt file
        """
        # Configure Gemini
        genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model)
        
        # Load prompt template
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
    
    def generate_plan(self, image_path: str, task: str) -> Dict:
        """
        Generate action plan for a single image-task pair
        
        Args:
            image_path: Path to the image file
            task: Task instruction
            
        Returns:
            Dictionary containing plan or error information
        """
        try:
            # Replace task in prompt template
            prompt = self.prompt_template.replace(
                "{Instruction_From_Stage_A}",
                task
            )
            
            # Load image
            image = Image.open(image_path)
            
            # Call Gemini
            response = self.model.generate_content(
                [prompt, image],
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 800,
                }
            )
            
            # Parse response
            content = response.text
            plan_data = json.loads(content)
            
            return {
                "success": True,
                "image_path": image_path,
                "task": task,
                "reasoning": plan_data.get("reasoning", ""),
                "plan": plan_data.get("plan", []),
                "raw_response": content
            }
            
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "image_path": image_path,
                "task": task,
                "error": "JSON parsing failed",
                "raw_response": content if 'content' in locals() else None
            }
        except Exception as e:
            return {
                "success": False,
                "image_path": image_path,
                "task": task,
                "error": str(e)
            }
    
    def batch_generate_from_step1(
        self,
        step1_data_path: str,
        output_path: str,
        max_samples: int = None
    ) -> List[Dict]:
        """
        Batch generate plans from Step1 results
        
        Args:
            step1_data_path: Path to Step1 output JSON
            output_path: Path to save output JSON
            max_samples: Maximum number of samples to process
            
        Returns:
            List of generation results
        """
        # Load Step1 data
        with open(step1_data_path, "r", encoding="utf-8") as f:
            step1_results = json.load(f)
        
        # Filter successful results
        valid_results = [r for r in step1_results if r.get("success")]
        logger.info(f"Loaded {len(valid_results)} valid Step1 results")
        
        # Prepare image-task pairs
        pairs = []
        for result in valid_results:
            image_path = result["image_path"]
            tasks = result.get("tasks", [])
            for task in tasks:
                pairs.append({
                    "image_path": image_path,
                    "task": task
                })
        
        if max_samples:
            pairs = pairs[:max_samples]
        
        logger.info(f"Total {len(pairs)} image-task pairs to process")
        
        # Generate plans
        results = []
        success_count = 0
        
        for pair in tqdm(pairs, desc="Generating Step2 data"):
            result = self.generate_plan(
                image_path=pair["image_path"],
                task=pair["task"]
            )
            results.append(result)
            
            if result["success"]:
                success_count += 1
        
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nGeneration complete:")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Failed: {len(results) - success_count}")
        logger.info(f"  Output saved to: {output_path}")
        
        return results
    
    def batch_generate_from_pairs(
        self,
        pairs_path: str,
        output_path: str,
        max_samples: int = None
    ) -> List[Dict]:
        """
        Batch generate plans from predefined image-task pairs
        
        Args:
            pairs_path: Path to JSON file containing pairs
            output_path: Path to save output JSON
            max_samples: Maximum number of samples to process
            
        Returns:
            List of generation results
        """
        # Load pairs
        with open(pairs_path, "r", encoding="utf-8") as f:
            pairs = json.load(f)
        
        if max_samples:
            pairs = pairs[:max_samples]
        
        logger.info(f"Loaded {len(pairs)} image-task pairs to process")
        
        # Generate plans
        results = []
        success_count = 0
        
        for pair in tqdm(pairs, desc="Generating Step2 data"):
            result = self.generate_plan(
                image_path=pair["image_path"],
                task=pair["task"]
            )
            results.append(result)
            
            if result["success"]:
                success_count += 1
        
        # Save results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\nGeneration complete:")
        logger.info(f"  Total: {len(results)}")
        logger.info(f"  Success: {success_count}")
        logger.info(f"  Failed: {len(results) - success_count}")
        logger.info(f"  Output saved to: {output_path}")
        
        return results


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Step2 action plans")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file (Step1 results or image-task pairs)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft_step2.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="step2.prompt",
        help="Path to step2 prompt file"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.0-flash-exp",
        help="Gemini model to use (gemini-2.0-flash-exp or gemini-1.5-pro)"
    )
    parser.add_argument(
        "--input_type",
        type=str,
        default="step1",
        choices=["step1", "pairs"],
        help="Input type: 'step1' for Step1 results, 'pairs' for direct pairs"
    )
    
    args = parser.parse_args()
    
    # Create generator
    generator = Step2Generator(
        model=args.model,
        prompt_path=args.prompt
    )
    
    # Run generation
    if args.input_type == "step1":
        generator.batch_generate_from_step1(
            step1_data_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples
        )
    else:
        generator.batch_generate_from_pairs(
            pairs_path=args.input,
            output_path=args.output,
            max_samples=args.max_samples
        )


if __name__ == "__main__":
    main()

