
import argparse
import json
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
    return p.parse_args()


def create_app(model_name: str, cache_dir: str | None = None) -> FastAPI:
    app = FastAPI(title="Transformers From Scratch API")

    if model_name == "gpt2":
        from models.gpt2 import GPT2

        model = GPT2(cache_dir=cache_dir)
    else:
        from models.smollm2 import SmolLM2

        model = SmolLM2(cache_dir=cache_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": model_name}

    def build_prompt(raw_prompt: str) -> str:
        if model_name == "smollm2":
            messages = [{"role": "user", "content": raw_prompt}]
            return model.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        return raw_prompt

    @app.post("/generate")
    def generate(req: GenerateRequest) -> dict[str, str]:
        try:
            prompt = build_prompt(req.prompt)
            text = model.generate(prompt, max_tokens=req.max_tokens)
            return {"text": text}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/generate/stream")
    def stream_generate(req: GenerateRequest) -> StreamingResponse:
        prompt = build_prompt(req.prompt)

        def event_stream():
            text = prompt
            eos_token = getattr(model.tokenizer, "eos_token", None)
            try:
                for _ in range(req.max_tokens):
                    next_token = model.next_token(text)
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
    app = create_app(model_name=args.model, cache_dir=args.cache_dir)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
