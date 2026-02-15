
import argparse
import importlib
import json
import os
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=10, ge=1, le=512)


def parse_args():
    p = argparse.ArgumentParser(description="Start a server for the model.")
    p.add_argument("--model", choices=("gpt2", "smollm2"), required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode with hot-reload endpoint for attention code",
    )
    return p.parse_args()


def create_app(
    model_name: str, cache_dir: str | None = None, dev_mode: bool = False
) -> FastAPI:
    app = FastAPI(title="Transformers From Scratch API")

    # Store model in a dict so we can reassign it in reload endpoint
    state = {"model": None, "model_name": model_name, "cache_dir": cache_dir}

    if model_name == "gpt2":
        from models.gpt2 import GPT2

        state["model"] = GPT2(cache_dir=cache_dir)
    else:
        from models.smollm2 import SmolLM2

        state["model"] = SmolLM2(cache_dir=cache_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_name, "dev_mode": dev_mode}

    def build_prompt(raw_prompt: str) -> str:
        if model_name == "smollm2":
            messages = [{"role": "user", "content": raw_prompt}]
            return state["model"].tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return raw_prompt

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict[str, str]:
        try:
            prompt = build_prompt(req.prompt)
            text = state["model"].generate(prompt, max_tokens=req.max_tokens)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/generate/stream")
    def stream_generate(req: GenerateRequest) -> StreamingResponse:
        prompt = build_prompt(req.prompt)

        def event_stream():
            text = prompt
            eos_token = getattr(state["model"].tokenizer, "eos_token", None)
            try:
                for _ in range(req.max_tokens):
                    next_token = state["model"].next_token(text)
                    text += next_token
                    payload = json.dumps({"token": next_token})
                    yield f"data: {payload}\n\n"

                    if eos_token and next_token == eos_token:
                        break

                yield "data: [DONE]\n\n"
            except Exception as exc:
                payload = json.dumps({"error": str(exc)})
                yield f"data: {payload}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def main():
    args = parse_args()
    app = create_app(
        model_name=args.model, cache_dir=args.cache_dir, dev_mode=args.dev
    )
    
    if args.dev:
        print("🔥 Dev mode enabled!")
        print("   - Watching src/ for changes")
        print("   - Auto-reload on file save")
        print("   - Manual reload: POST /reload")
        print()
        
        # Set up file watching for auto-reload
        try:
            from watchfiles import awatch
            import asyncio
            import threading
            
            # Get the src directory path
            src_dir = Path(__file__).parent.parent.absolute()
            
            async def watch_and_reload():
                """Watch for file changes and trigger reload."""
                async for changes in awatch(src_dir):
                    # Filter to only Python files
                    py_changes = [
                        path for _, path in changes 
                        if path.endswith('.py')
                    ]
                    
                    if py_changes:
                        print(f"\n🔄 Detected changes in: {[os.path.basename(p) for p in py_changes]}")
                        print("   Reloading attention module...")
                        
                        try:
                            # Trigger the same reload logic
                            import utils.attention
                            importlib.reload(utils.attention)
                            
                            if args.model == "gpt2":
                                from models import gpt2
                                importlib.reload(gpt2)
                            else:
                                from models import smollm2
                                importlib.reload(smollm2)
                            
                            print("   ✅ Reload complete!\n")
                        except Exception as e:
                            print(f"   ❌ Reload failed: {e}\n")
            
            def start_watcher():
                """Start the file watcher in the background."""
                asyncio.run(watch_and_reload())
            
            # Start watcher in background thread
            watcher_thread = threading.Thread(target=start_watcher, daemon=True)
            watcher_thread.start()
            
        except ImportError:
            print("⚠️  watchfiles not installed. Auto-reload disabled.")
            print("   Install with: pip install watchfiles")
            print("   Or use: POST /reload for manual reload")
            print()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
