import asyncio
import aiohttp
import base64
import json
import re
import shutil
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger(__name__)


MODEL1 = MODEL2 = "google/gemini-3-pro-preview"
OPENROUTER_API_KEY = "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"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

IMAGE_ROOT = Path("./bdv2")
PROCESSED_ROOT = Path("./processed")
TASK_OUT = Path("./tasks.jsonl")
ACTION_OUT = Path("./actions.jsonl")

SEM = asyncio.Semaphore(1)

TASK_SYSTEM_PROMPT = """
    # Role
You are the brain of a robot. Your task is to observe the image and propose only one reasonable and executable instructions for this scene.

# Constraints
1. Tasks can only involve moving, picking up, and placing objects.
2. Objects mentioned in the task must be clearly visible in the image.
3. Target locations mentioned in the task must also be visible in the image.
4. Output must be in JSON format.

# Output Format
{
  "task": [
    "Throw the cup on the black table into the trash can",
    "Put the bowl on the white table into the sink"
  ]
}
"""

ACTION_SYSTEM_PROMPT = """
# Role
You are an embodied AI planning expert.

# Input
- Image: [Input Image]
- Task: "{Instruction_From_Stage_A}"

# Action Space
1. Navigate(target): Move to a target location.
2. Pick(object): Pick up an object.
3. Place(receptacle): Place the held object.

# Requirements
1. Analyze the image to locate the object and target location mentioned in the task.
2. Generate a step-by-step action plan.
3. Provide your reasoning process in the "reasoning" field.
4. Output must be valid JSON.

# Output Format
{
  "reasoning": "I can see an apple on the white table and a basket on the floor. I need to first navigate to the white table, pick up the apple, then navigate to the basket and place the apple inside.",
  "plan": [
    "Navigate(White Table)",
    "Pick(Apple)",
    "Navigate(Basket)",
    "Place(Basket)"
  ]
}
"""



def encode_image(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")

async def call_openrouter(session, model, messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 2048  
    }
    async with session.post(OPENROUTER_URL, headers=headers, json=payload) as resp:
        data = await resp.json()
        try:
            ret = data["choices"][0]["message"]["content"]
        except:
            ret = None
            logger.info("return false")
            logger.info(data)
        return ret

async def safe_call_openrouter(session, model, messages):
    async with SEM:   
        return await call_openrouter(session, model, messages)

def parse_task(text):

    try:
        data = json.loads(text)
        if isinstance(data, dict) and isinstance(data.get("task"), str):
            return data["task"].strip()
    except:
        pass
    

    match = re.search(r'"task"\s*:\s*"([^"]+)"', text)
    return match.group(1).strip() if match else None

def parse_plan(text):

    try:
        data = json.loads(text)
        plan = data.get("plan")
        if isinstance(plan, list):
            return [str(step).strip() for step in plan]
    except:
        pass


    match = re.search(r'"plan"\s*:\s*\[(.*?)\]', text, re.S)
    if not match:
        return None

    steps = re.findall(r'"([^"]+)"', match.group(1))
    return steps if steps else None

def collect_first_images(root: Path):
    logger.info("collecting_img_path")
    image_ext = {".jpg", ".jpeg", ".png"}
    selected = []
    for sub in root.rglob("*"):
        if sub.is_dir():
            imgs = sorted(p for p in sub.iterdir() if p.suffix.lower() in image_ext)
            if imgs:
                first = imgs[0]
                if first.stem.endswith("_0"):
                    selected.append(first)
    logger.info("img_dir_sum",len(selected))
    return selected

async def process_image(session, image_path: Path):
    img_b64 = encode_image(image_path)
    image_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img_b64}"
            }
        }
    ]

    task_messages = [
        {"role": "system", "content": TASK_SYSTEM_PROMPT},
        {"role": "user", "content": image_content},
    ]
    logger.info("task_info_load_complete")

    task_raw = await safe_call_openrouter(session, MODEL1, task_messages)
    logger.info("task_send_succ")
    task = parse_task(task_raw)
    try:
        
        logger.info(task)
        logger.info("task_json_succ")
    except:
        logger.info("task_json_false")
        if task != None:
            
            logger.info(task)
        else:
            logger.info("task_is_none")
            logger.info(task_raw)
    if not task:
        return

    action_messages = [
        {"role": "system", "content": ACTION_SYSTEM_PROMPT},
        {"role": "user", "content": image_content},
        {"role": "user", "content": f"task: {task}"},
    ]
    logger.info("action_info_load_complete")

    action_raw = await safe_call_openrouter(session, MODEL2, action_messages)
    logger.info("action_send_succ")
    plan = parse_plan(action_raw)
    logger.info(plan[:20])
    if not plan:
        return

    # 计算处理后的路径（保持原有目录结构）
    rel_path = image_path.relative_to(IMAGE_ROOT)
    processed_img_path = PROCESSED_ROOT / rel_path
    processed_json_path = processed_img_path.with_suffix('.json')

    # 创建目录
    processed_img_path.parent.mkdir(parents=True, exist_ok=True)

    # 创建单独的 json 文件（方便检查）
    json_data = {
        "image": str(processed_img_path),
        "task": task,
        "plan": plan
    }
    processed_json_path.write_text(
        json.dumps(json_data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    # 追加到总的 jsonl 文件
    TASK_OUT.open("a", encoding="utf-8").write(json.dumps({
        "image": str(processed_img_path),
        "task": task
    }, ensure_ascii=False) + "\n")

    ACTION_OUT.open("a", encoding="utf-8").write(json.dumps({
        "image": str(processed_img_path),
        "task": task,
        "plan": plan
    }, ensure_ascii=False) + "\n")

    # 移动图片到 processed 目录
    shutil.move(str(image_path), str(processed_img_path))
    logger.info(f"Processed and moved: {image_path} -> {processed_img_path}")

async def main():
    images = collect_first_images(IMAGE_ROOT)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(process_image(session, img) for img in images))

asyncio.run(main())
