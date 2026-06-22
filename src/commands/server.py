import argparse
import json
import os
from pathlib import Path
from typing import Any
from jax import numpy as jnp

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(default=10, ge=1, le=512)


def parse_args():
    p = argparse.ArgumentParser(description="Start a server for the model.")
    p.add_argument("--arch", choices=("gpt2", "llama"), required=True)
    p.add_argument(
        "--repo",
        type=str,
        required=True,
        help="HuggingFace repo id for weights/tokenizer",
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--cache-dir", type=str, default=None)
    p.add_argument(
        "--dev",
        action="store_true",
        help="Enable dev mode with uvicorn auto-reload for Python files under src/",
    )
    return p.parse_args()


def create_app(
    model_name: str, repo_id: str, cache_dir: str | None = None, dev_mode: bool = False
) -> FastAPI:
    app = FastAPI(title="Transformers From Scratch API")

    # Store model in a dict so we can reassign it in reload endpoint
    state = {"model": None, "model_name": model_name, "cache_dir": cache_dir}

    if model_name == "gpt2":
        from models.gpt2 import GPT2

        state["model"] = GPT2(repo_id=repo_id, cache_dir=cache_dir)
    else:
        from models.llama import Llama

        state["model"] = Llama(repo_id=repo_id, cache_dir=cache_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_name, "dev_mode": dev_mode}

    def build_prompt(raw_prompt: str) -> jnp.ndarray:
        if model_name == "llama":
            messages = [{"role": "user", "content": raw_prompt}]
            return jnp.array(
                state["model"].tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
        return jnp.array(state["model"].tokenizer.encode(raw_prompt))

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict[str, str]:
        try:
            tokens = build_prompt(req.prompt)
            tokens = state["model"].generate(tokens, max_tokens=req.max_tokens)
            text = state["model"].tokenizer.decode(tokens, skip_special_tokens=True)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/generate/stream")
    def stream_generate(req: GenerateRequest) -> StreamingResponse:
        def event_stream():
            tokens = build_prompt(req.prompt)
            eos_token_id = state["model"].tokenizer.eos_token_id
            try:
                for _ in range(req.max_tokens):
                    next_token = state["model"].next_token(tokens)
                    tokens = jnp.concatenate([tokens, jnp.array([next_token])], axis=0)
                    payload = json.dumps(
                        {
                            "token": state["model"].tokenizer.decode(
                                [next_token], skip_special_tokens=True
                            )
                        }
                    )
                    yield f"data: {payload}\n\n"

                    if next_token == eos_token_id:
                        break

                yield "data: [DONE]\n\n"
            except Exception as exc:
                payload = json.dumps({"error": str(exc)})
                yield f"data: {payload}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def create_app_from_env() -> FastAPI:
    """Factory used by uvicorn reload workers."""
    model_name = os.environ["TRANSFORMERS_ARCH"]
    repo_id = os.environ["TRANSFORMERS_REPO"]
    cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR")
    dev_mode = os.environ.get("TRANSFORMERS_DEV_MODE") == "1"
    return create_app(
        model_name=model_name, repo_id=repo_id, cache_dir=cache_dir, dev_mode=dev_mode
    )


def main():
    args = parse_args()

    if args.dev:
        print("🔥 Dev mode enabled!")
        print("   - Watching src/ for changes")
        print("   - Auto-reload with uvicorn --reload")
        print()

        src_dir = Path(__file__).parent.parent.absolute()
        repo_root = src_dir.parent

        # Ensure reload worker processes can import "commands.server"
        existing_pythonpath = os.environ.get("PYTHONPATH")
        pythonpath_entries = [str(src_dir)]
        if existing_pythonpath:
            pythonpath_entries.append(existing_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

        os.environ["TRANSFORMERS_ARCH"] = args.arch
        os.environ["TRANSFORMERS_REPO"] = args.repo
        if args.cache_dir:
            os.environ["TRANSFORMERS_CACHE_DIR"] = args.cache_dir
        else:
            os.environ.pop("TRANSFORMERS_CACHE_DIR", None)
        os.environ["TRANSFORMERS_DEV_MODE"] = "1"

        uvicorn.run(
            "commands.server:create_app_from_env",
            host="0.0.0.0",
            port=args.port,
            reload=True,
            reload_dirs=[str(src_dir)],
            app_dir=str(repo_root),
            factory=True,
        )
        return

    app = create_app(
        model_name=args.arch,
        repo_id=args.repo,
        cache_dir=args.cache_dir,
        dev_mode=False,
    )
    uvicorn.run(app, host="0.0.0.0", port=args.port)
