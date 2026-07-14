"""Minimal mem0 OSS server — same embedder as MAG (fastembed BGE-small 384d)"""
import os, json, logging, asyncio
from concurrent.futures import ThreadPoolExecutor
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("mem0-oss")

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Thread pool for blocking mem0 calls — prevents event loop blockage
_executor = ThreadPoolExecutor(max_workers=16)
# Semaphore limits concurrent LLM calls to avoid API rate limits
_add_sem = asyncio.Semaphore(8)

app = FastAPI()

class AddRequest(BaseModel):
    messages: list
    user_id: str
    observation_date: str | None = None
    timestamp: int | None = None

class SearchRequest(BaseModel):
    query: str
    user_id: str
    limit: int = 30

memory = None

@app.on_event("startup")
async def startup():
    global memory
    from mem0 import Memory
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("OPENAI_API_KEY must be set before starting mem0_oss_server.py")
    config = {
        'llm': {'provider': 'openai', 'config': {
            'model': os.getenv('OPENAI_MODEL', 'gpt-4o-mini'),
            'api_key': openai_api_key,
            'openai_base_url': os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
        }},
        'embedder': {'provider': 'fastembed', 'config': {
            'model': 'BAAI/bge-small-en-v1.5',
            'embedding_dims': 384,
        }},
        'vector_store': {'provider': 'qdrant', 'config': {
            'host': 'localhost', 'port': 6333,
            'embedding_model_dims': 384,
        }},
        'history_db_path': os.path.expanduser('~/.mem0/oss_history.db'),
    }
    memory = Memory.from_config(config_dict=config)
    logger.info("Mem0 OSS ready — fastembed BGE-small 384d")

@app.post("/memories")
async def add_memories(req: AddRequest):
    async with _add_sem:
        try:
            kwargs = {'user_id': req.user_id}
            if req.timestamp is not None:
                kwargs['timestamp'] = req.timestamp
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(_executor, lambda: memory.add(req.messages, **kwargs))
            return {"results": result.get("results", []) if isinstance(result, dict) else []}
        except Exception as e:
            raise HTTPException(500, str(e))

@app.post("/search")
async def search_memories(req: SearchRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(_executor, lambda: memory.search(req.query, filters={'user_id': req.user_id}, top_k=req.limit))
        return {"results": result.get("results", []) if isinstance(result, dict) else []}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/memories")
async def delete_memories(user_id: str):
    try:
        memory.delete_all(user_id=user_id)
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/health")
def health():
    return {"status": "ok", "embedder": "fastembed", "model": "BAAI/bge-small-en-v1.5"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
