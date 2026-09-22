from __future__ import annotations

import argparse
import base64
import json
import os
import random
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fmg_fly_connectome import FlyConnectomeRouter, RouteDecision


VERSION = "FMG-IMG-CONNECTOME-2.1"
DEFAULT_MODEL = "segmind/SSD-1B"
DEFAULT_A1111 = "http://127.0.0.1:7860"
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024
DEFAULT_STEPS = 50
DEFAULT_GUIDANCE = 9.0
DEFAULT_NEGATIVE = (
    "low quality, blurry, distorted, deformed anatomy, extra fingers, "
    "extra limbs, duplicate subject, watermark, signature, logo, text overlay"
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}



@dataclass(frozen=True)
class ImageRequest:
    prompt: str
    negative_prompt: str = DEFAULT_NEGATIVE
    steps: int = DEFAULT_STEPS
    guidance: float = DEFAULT_GUIDANCE
    seed: int = -1
    backend: str = "connectome"

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "ImageRequest":
        prompt = str(raw.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("prompt is required")
        # Legacy width/height inputs are intentionally ignored: the image
        # plane is a fixed 1024x1024 organ-level constant.
        steps = max(1, min(80, int(raw.get("steps", DEFAULT_STEPS))))
        guidance = max(0.0, min(20.0, float(raw.get("guidance", DEFAULT_GUIDANCE))))
        seed = int(raw.get("seed", -1))
        if seed < -1:
            seed = -1
        backend = str(raw.get("backend") or "connectome").strip().lower()
        if backend == "auto":
            backend = "connectome"
        if backend not in {"connectome", "a1111", "diffusers"}:
            raise ValueError(
                "backend must be connectome, a1111, or diffusers"
            )
        return cls(
            prompt=prompt,
            negative_prompt=str(
                raw.get("negative_prompt") or DEFAULT_NEGATIVE
            ),
            steps=steps,
            guidance=guidance,
            seed=seed,
            backend=backend,
        )


class BackendError(RuntimeError):
    pass


class A1111Backend:
    def __init__(self, base_url: str | None = None):
        raw = (
            base_url
            or os.environ.get("FMG_A1111_URL")
            or DEFAULT_A1111
        ).rstrip("/")
        parsed = urllib.parse.urlparse(raw)
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        ):
            raise ValueError("A1111 backend must be loopback-only")
        self.base_url = raw

    def _json(
        self,
        path: str,
        *,
        method: str = "GET",
        payload: dict | None = None,
        timeout: float = 5.0,
    ):
        data = (
            None
            if payload is None
            else json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            raise BackendError(
                f"A1111 request failed: {exc}"
            ) from exc

    def probe(self) -> dict[str, Any]:
        try:
            models = self._json(
                "/sdapi/v1/sd-models",
                timeout=1.5,
            )
            titles = []
            if isinstance(models, list):
                for row in models[:8]:
                    if isinstance(row, dict):
                        titles.append(
                            str(
                                row.get("title")
                                or row.get("model_name")
                                or ""
                            )
                        )
            return {
                "available": True,
                "backend": "a1111",
                "models": titles,
                "url": self.base_url,
            }
        except Exception as exc:
            return {
                "available": False,
                "backend": "a1111",
                "error": str(exc),
                "url": self.base_url,
            }

    def generate(
        self,
        req: ImageRequest,
        output_dir: Path,
    ) -> dict[str, Any]:
        seed = (
            req.seed
            if req.seed >= 0
            else random.SystemRandom().randint(
                0,
                2**31 - 1,
            )
        )
        payload = {
            "prompt": req.prompt,
            "negative_prompt": req.negative_prompt,
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "steps": req.steps,
            "cfg_scale": req.guidance,
            "seed": seed,
            "batch_size": 1,
            "n_iter": 1,
        }
        started = time.perf_counter()
        value = self._json(
            "/sdapi/v1/txt2img",
            method="POST",
            payload=payload,
            timeout=600,
        )
        images = (
            value.get("images")
            if isinstance(value, dict)
            else None
        )
        if not images:
            raise BackendError("A1111 returned no image")
        encoded = str(images[0]).split(",", 1)[-1]
        raw = base64.b64decode(encoded, validate=False)
        if not (
            raw.startswith(b"\x89PNG\r\n\x1a\n")
            or raw.startswith(b"\xff\xd8")
        ):
            raise BackendError(
                "A1111 returned an unsupported image payload"
            )
        ext = (
            ".png"
            if raw.startswith(b"\x89PNG")
            else ".jpg"
        )
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        name = (
            f"fmg_{int(time.time()*1000)}_"
            f"{seed}{ext}"
        )
        path = output_dir / name
        path.write_bytes(raw)
        return {
            "ok": True,
            "backend": "a1111",
            "seed": seed,
            "path": str(path),
            "name": name,
            "elapsed_s": round(
                time.perf_counter() - started,
                3,
            ),
        }


class DiffusersBackend:
    def __init__(self):
        self.model_id = os.environ.get(
            "FMG_IMAGE_MODEL",
            DEFAULT_MODEL,
        )
        self.checkpoint = os.environ.get(
            "FMG_IMAGE_CHECKPOINT",
            "",
        ).strip()
        self.offline = _env_bool(
            "FMG_IMAGE_OFFLINE",
            False,
        )
        self.low_vram = _env_bool(
            "FMG_IMAGE_LOW_VRAM",
            True,
        )
        self._pipe = None
        self._torch = None
        self._device = "unloaded"
        self._lock = threading.Lock()

    def probe(self) -> dict[str, Any]:
        try:
            import importlib.util

            missing = [
                x
                for x in ("torch", "diffusers", "PIL")
                if importlib.util.find_spec(x) is None
            ]
            if missing:
                return {
                    "available": False,
                    "backend": "diffusers",
                    "missing": missing,
                    "model": self.model_id,
                }
            return {
                "available": True,
                "backend": "diffusers",
                "loaded": self._pipe is not None,
                "device": self._device,
                "model": (
                    self.checkpoint
                    or self.model_id
                ),
                "offline": self.offline,
                "low_vram": self.low_vram,
            }
        except Exception as exc:
            return {
                "available": False,
                "backend": "diffusers",
                "error": str(exc),
            }

    def _load(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import (
                DiffusionPipeline,
                StableDiffusionXLPipeline,
            )
        except Exception as exc:
            raise BackendError(
                "Diffusers backend dependencies are missing. "
                "Run: pip install -r requirements-image.txt"
            ) from exc

        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif (
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
        ):
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        kwargs = {"torch_dtype": dtype}
        if self.offline:
            kwargs["local_files_only"] = True

        try:
            if self.checkpoint:
                pipe = StableDiffusionXLPipeline.from_single_file(
                    self.checkpoint,
                    **kwargs,
                )
            else:
                pipe = DiffusionPipeline.from_pretrained(
                    self.model_id,
                    **kwargs,
                )
        except Exception as exc:
            raise BackendError(
                f"Could not load image model: {exc}"
            ) from exc

        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        if device == "cuda" and self.low_vram:
            try:
                pipe.enable_model_cpu_offload()
                self._device = "cuda+cpu-offload"
            except Exception:
                pipe.to(device)
                self._device = device
        else:
            pipe.to(device)
            self._device = device

        self._pipe = pipe
        self._torch = torch

    def generate(
        self,
        req: ImageRequest,
        output_dir: Path,
    ) -> dict[str, Any]:
        with self._lock:
            self._load()
            torch = self._torch
            pipe = self._pipe
            if torch is None or pipe is None:
                raise BackendError(
                    "Diffusers pipeline unavailable"
                )

            seed = (
                req.seed
                if req.seed >= 0
                else random.SystemRandom().randint(
                    0,
                    2**31 - 1,
                )
            )
            gen_device = (
                "cpu"
                if self._device == "cuda+cpu-offload"
                else self._device
            )
            if gen_device not in {
                "cpu",
                "cuda",
                "mps",
            }:
                gen_device = "cpu"
            generator = torch.Generator(
                device=gen_device
            ).manual_seed(seed)

            started = time.perf_counter()
            try:
                out = pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt,
                    width=IMAGE_WIDTH,
                    height=IMAGE_HEIGHT,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance,
                    generator=generator,
                )
                image = out.images[0]
            except Exception as exc:
                raise BackendError(
                    f"Diffusers generation failed: {exc}"
                ) from exc

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )
            name = (
                f"fmg_{int(time.time()*1000)}_"
                f"{seed}.png"
            )
            path = output_dir / name
            image.save(path, format="PNG")
            return {
                "ok": True,
                "backend": "diffusers",
                "seed": seed,
                "path": str(path),
                "name": name,
                "device": self._device,
                "elapsed_s": round(
                    time.perf_counter() - started,
                    3,
                ),
            }


class FMGImageGenerator:
    def __init__(
        self,
        output_dir: str | Path | None = None,
    ):
        self.output_dir = Path(
            output_dir
            or os.environ.get(
                "FMG_IMAGE_OUTPUT",
                "runtime/images",
            )
        ).resolve()
        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.a1111 = A1111Backend()
        self.diffusers = DiffusersBackend()
        state_path = Path(
            os.environ.get(
                "FMG_CONNECTOME_STATE",
                str(
                    self.output_dir.parent
                    / "connectome"
                    / "state.json"
                ),
            )
        )
        self.connectome = FlyConnectomeRouter(
            state_path
        )
        self._generation_lock = threading.Lock()

    def _availability(self) -> dict[str, bool]:
        return {
            "a1111": bool(
                self.a1111.probe().get("available")
            ),
            "diffusers": bool(
                self.diffusers.probe().get("available")
            ),
        }

    def status(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "state": "ready",
            "controller": self.connectome.status(),
            "default_route": "fly-connectome",
            "default_model": DEFAULT_MODEL,
            "resolution": {"width": IMAGE_WIDTH, "height": IMAGE_HEIGHT, "policy": "fixed"},
            "quality_defaults": {"steps": DEFAULT_STEPS, "guidance": DEFAULT_GUIDANCE},
            "output_dir": str(self.output_dir),
            "backends": {
                "a1111": self.a1111.probe(),
                "diffusers": self.diffusers.probe(),
            },
            "distillation": {
                "mode": "pre-distilled-student",
                "student_model": DEFAULT_MODEL,
                "additional_distillation_required": False,
            },
            "constraints": {
                "qwen_used": False,
                "external_cloud_api_required": False,
                "lazy_model_loading": True,
            },
        }

    def _backend_for(self, name: str):
        if name == "a1111":
            return self.a1111
        if name == "diffusers":
            return self.diffusers
        raise BackendError(
            f"unknown image organ: {name}"
        )

    def _structural_verify(
        self,
        result: dict[str, Any],
    ) -> bool:
        try:
            path = Path(result["path"])
            raw = path.read_bytes()
            if len(raw) < 128:
                return False
            return (
                raw.startswith(b"\x89PNG\r\n\x1a\n")
                or raw.startswith(b"\xff\xd8")
            )
        except Exception:
            return False

    def _debug_override_decision(
        self,
        req: ImageRequest,
        backend: str,
    ) -> RouteDecision:
        availability = self._availability()
        if not availability.get(backend, False):
            raise BackendError(
                f"requested debug backend is unavailable: "
                f"{backend}"
            )
        base = self.connectome.route(
            asdict(req),
            availability,
        )
        return RouteDecision(
            selected_backend=backend,
            selected_mbon=(
                f"DEBUG-OVERRIDE::{backend}"
            ),
            pn_active=base.pn_active,
            kc_active=base.kc_active,
            kc_sparsity=base.kc_sparsity,
            scores=base.scores,
            available_backends=(
                base.available_backends
            ),
            retry_order=(backend,),
        )

    def generate(
        self,
        req: ImageRequest,
    ) -> dict[str, Any]:
        availability = self._availability()
        if req.backend in {
            "a1111",
            "diffusers",
        }:
            first = self._debug_override_decision(
                req,
                req.backend,
            )
        else:
            try:
                first = self.connectome.route(
                    asdict(req),
                    availability,
                )
            except RuntimeError as exc:
                raise BackendError(
                    str(exc)
                ) from exc

        attempted: list[dict[str, Any]] = []
        inhibited: set[str] = set()
        decision = first
        max_attempts = max(
            1,
            len(first.retry_order),
        )

        with self._generation_lock:
            for _ in range(max_attempts):
                backend = self._backend_for(
                    decision.selected_backend
                )
                try:
                    result = backend.generate(
                        req,
                        self.output_dir,
                    )
                    verified = (
                        self._structural_verify(result)
                    )
                    reward = (
                        0.82
                        if verified
                        else -0.55
                    )
                    self.connectome.reward(
                        decision,
                        reward,
                        structural_verified=verified,
                        elapsed_s=result.get(
                            "elapsed_s"
                        ),
                    )
                    attempted.append(
                        {
                            "backend": (
                                decision.selected_backend
                            ),
                            "ok": True,
                            "structural_verified": (
                                verified
                            ),
                        }
                    )
                    if not verified:
                        raise BackendError(
                            "generated artifact failed "
                            "structural verification"
                        )

                    result["request"] = asdict(req)
                    result["artifact_url"] = (
                        "/artifacts/"
                        + urllib.parse.quote(
                            result["name"]
                        )
                    )
                    result["version"] = VERSION
                    result["controller"] = (
                        "fly-connectome"
                    )
                    result["connectome_route"] = (
                        decision.as_dict()
                    )
                    result["attempts"] = attempted
                    return result
                except Exception as exc:
                    self.connectome.reward(
                        decision,
                        -0.85,
                        structural_verified=False,
                        elapsed_s=None,
                    )
                    attempted.append(
                        {
                            "backend": (
                                decision.selected_backend
                            ),
                            "ok": False,
                            "error": str(exc),
                        }
                    )
                    inhibited.add(
                        decision.selected_backend
                    )

                    if req.backend in {
                        "a1111",
                        "diffusers",
                    }:
                        raise BackendError(
                            str(exc)
                        ) from exc

                    try:
                        decision = (
                            self.connectome.route(
                                asdict(req),
                                availability,
                                inhibited=inhibited,
                            )
                        )
                    except RuntimeError:
                        break

        raise BackendError(
            "Connectome exhausted available "
            "image organs. "
            + json.dumps(
                attempted,
                ensure_ascii=False,
            )
        )


_UI = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FMG Image Generator</title>
<style>
body{font-family:system-ui;background:#0d1117;color:#e6edf3;
max-width:900px;margin:28px auto;padding:0 18px}
textarea,input,select,button{font:inherit}
textarea,input,select{box-sizing:border-box;width:100%;background:#161b22;
color:#e6edf3;border:1px solid #30363d;border-radius:8px;padding:10px}
textarea{min-height:110px}button{padding:10px 16px;margin-top:10px}
pre{white-space:pre-wrap;background:#161b22;padding:12px;border-radius:8px}
img{max-width:100%;border-radius:10px;margin-top:12px}
</style>
</head>
<body>
<h1>FMG Image Generator — Fly Connectome V2</h1>
<p>1024×1024 fixed · PN → sparse KC → MBON → lazy image organ → DAN reward</p>
<textarea id="prompt" placeholder="画像生成プロンプト"></textarea>
<p><label>Backend
<select id="backend">
<option value="connectome">fly-connectome</option>
<option value="a1111">A1111 debug override</option>
<option value="diffusers">Diffusers debug override</option>
</select></label></p>
<button onclick="generateImage()">生成</button>
<pre id="out">ready</pre>
<div id="image"></div>
<script>
async function generateImage(){
  const body={
    prompt:document.getElementById("prompt").value,
    steps:50,guidance:9,seed:-1,
    backend:document.getElementById("backend").value
  };
  const r=await fetch("/api/v1/image/generate",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify(body)
  });
  const j=await r.json();
  document.getElementById("out").textContent=JSON.stringify(j,null,2);
  const d=document.getElementById("image");
  d.innerHTML="";
  if(j.ok&&j.artifact_url){
    const i=document.createElement("img");
    i.src=j.artifact_url+"?t="+Date.now();
    d.appendChild(i);
  }
}
</script>
</body>
</html>
"""


GENERATOR: FMGImageGenerator | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = VERSION

    def _send_json(
        self,
        value: Any,
        status: int = 200,
    ):
        raw = json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
        ).encode("utf-8")
        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(raw)),
        )
        self.send_header(
            "Cache-Control",
            "no-store",
        )
        self.end_headers()
        self.wfile.write(raw)

    def _send_html(self, value: str):
        raw = value.encode("utf-8")
        self.send_response(200)
        self.send_header(
            "Content-Type",
            "text/html; charset=utf-8",
        )
        self.send_header(
            "Content-Length",
            str(len(raw)),
        )
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, fmt, *args):
        print("[FMG]", fmt % args)

    def do_GET(self):
        global GENERATOR
        if GENERATOR is None:
            self._send_json(
                {
                    "ok": False,
                    "error": (
                        "generator not initialized"
                    ),
                },
                500,
            )
            return

        path = urllib.parse.urlparse(
            self.path
        ).path
        if path == "/":
            self._send_html(_UI)
            return
        if path == "/api/v1/image/status":
            self._send_json(
                GENERATOR.status()
            )
            return
        if path.startswith("/artifacts/"):
            name = Path(
                urllib.parse.unquote(
                    path[len("/artifacts/") :]
                )
            ).name
            target = (
                GENERATOR.output_dir / name
            ).resolve()
            if (
                target.parent
                != GENERATOR.output_dir
                or not target.is_file()
            ):
                self.send_error(404)
                return
            raw = target.read_bytes()
            mime = (
                "image/jpeg"
                if target.suffix.lower()
                in {".jpg", ".jpeg"}
                else "image/png"
            )
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mime,
            )
            self.send_header(
                "Content-Length",
                str(len(raw)),
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_error(404)

    def do_POST(self):
        global GENERATOR
        if GENERATOR is None:
            self._send_json(
                {
                    "ok": False,
                    "error": (
                        "generator not initialized"
                    ),
                },
                500,
            )
            return

        path = urllib.parse.urlparse(
            self.path
        ).path
        if path != "/api/v1/image/generate":
            self.send_error(404)
            return
        try:
            size = int(
                self.headers.get(
                    "Content-Length",
                    "0",
                )
            )
            if size <= 0 or size > 1024 * 1024:
                raise ValueError(
                    "invalid request body size"
                )
            body = json.loads(
                self.rfile.read(size).decode(
                    "utf-8"
                )
            )
            if not isinstance(body, dict):
                raise ValueError(
                    "JSON body must be an object"
                )
            req = ImageRequest.from_mapping(
                body
            )
            self._send_json(
                GENERATOR.generate(req)
            )
        except ValueError as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                },
                400,
            )
        except BackendError as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                },
                503,
            )
        except Exception as exc:
            self._send_json(
                {
                    "ok": False,
                    "error": (
                        f"internal error: {exc}"
                    ),
                },
                500,
            )


def main():
    global GENERATOR
    ap = argparse.ArgumentParser(
        description=(
            "FMG fly-connectome "
            "image generator"
        )
    )
    ap.add_argument(
        "--host",
        default=os.environ.get(
            "FMG_IMAGE_HOST",
            "127.0.0.1",
        ),
    )
    ap.add_argument(
        "--port",
        type=int,
        default=int(
            os.environ.get(
                "FMG_IMAGE_PORT",
                "18765",
            )
        ),
    )
    ap.add_argument(
        "--output",
        default=os.environ.get(
            "FMG_IMAGE_OUTPUT",
            "runtime/images",
        ),
    )
    ap.add_argument(
        "--status",
        action="store_true",
    )
    args = ap.parse_args()

    GENERATOR = FMGImageGenerator(
        args.output
    )
    if args.status:
        print(
            json.dumps(
                GENERATOR.status(),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    server = ThreadingHTTPServer(
        (args.host, args.port),
        Handler,
    )
    print(
        f"{VERSION} listening on "
        f"http://{args.host}:{args.port}/"
    )
    print(
        "Controller: PN -> sparse KC -> "
        "MBON -> lazy organ -> DAN reward"
    )
    print(f"Default model: {DEFAULT_MODEL}")
    print(
        f"Artifacts: {GENERATOR.output_dir}"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
