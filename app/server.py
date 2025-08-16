import os
import uuid
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from torchvision.utils import save_image
from einops import repeat

from dalle_pytorch import DiscreteVAE, OpenAIDiscreteVAE, VQGanVAE, DALLE
from dalle_pytorch.tokenizer import tokenizer as default_tokenizer, HugTokenizer, YttmTokenizer, ChineseTokenizer


class GenerateRequest(BaseModel):
	text: str = Field(..., description="Text prompt")
	num_images: int = Field(4, ge=1, le=128)
	batch_size: int = Field(4, ge=1, le=64)
	top_k: float = Field(0.9, ge=0.0, le=1.0)
	gentxt: bool = False


class GenerateResponse(BaseModel):
	prompt: str
	count: int
	output_dir: str
	image_urls: List[str]


app = FastAPI(title="DALLE-pytorch Image Generator", version="1.0.0")

# CORS (optional, relaxed by default)
app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Static outputs dir
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", "outputs"))
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(OUTPUTS_DIR)), name="outputs")

# Global model state
_model_loaded = False
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_dalle: Optional[DALLE] = None
_tokenizer = None


def _bool_from_env(name: str, default: bool = False) -> bool:
	val = os.environ.get(name)
	if val is None:
		return default
	return str(val).lower() in {"1", "true", "yes", "y", "on"}


def _load_tokenizer_from_env():
	bpe_path = os.environ.get("BPE_PATH", None)
	if bpe_path:
		bpe_kind = os.environ.get("BPE_KIND", "yttm").lower()
		if bpe_kind == "hug":
			return HugTokenizer(bpe_path)
		else:
			return YttmTokenizer(bpe_path)
	if _bool_from_env("CHINESE", False):
		return ChineseTokenizer()
	return default_tokenizer


def _safe_dirname_from_text(text: str) -> str:
	name = "_".join(text.strip().split())
	if len(name) > 100:
		name = name[:100]
	return name


def _load_model_if_needed():
	global _model_loaded, _dalle, _tokenizer
	if _model_loaded:
		return

	dalle_path = os.environ.get("DALLE_WEIGHTS") or os.environ.get("DALLE_PATH")
	if not dalle_path:
		raise RuntimeError(
			"Environment variable DALLE_WEIGHTS (or DALLE_PATH) must point to a trained DALL-E checkpoint."
		)

	ckpt_path = Path(dalle_path)
	if not ckpt_path.exists():
		raise RuntimeError(f"DALL-E checkpoint not found at {ckpt_path}")

	map_location = _device if _device.type == "cpu" else None
	try:
		load_obj = torch.load(str(ckpt_path), map_location=map_location)
	except Exception:
		# retry CPU map
		load_obj = torch.load(str(ckpt_path), map_location="cpu")

	dalle_params = load_obj.pop("hparams")
	vae_params = load_obj.pop("vae_params")
	weights = load_obj.pop("weights")
	vae_class_name = load_obj.pop("vae_class_name", None)

	# VAE selection via env or checkpoint
	taming = _bool_from_env("TAMING", False)
	if taming:
		vqgan_model_path = os.environ.get("VQGAN_MODEL_PATH")
		vqgan_config_path = os.environ.get("VQGAN_CONFIG_PATH")
		if not vqgan_model_path or not vqgan_config_path:
			raise RuntimeError("TAMING enabled but VQGAN_MODEL_PATH or VQGAN_CONFIG_PATH not provided")
		vae = VQGanVAE(vqgan_model_path, vqgan_config_path)
	elif vae_params is not None:
		vae = DiscreteVAE(**vae_params)
	else:
		vae = OpenAIDiscreteVAE()

	if vae_class_name is not None and vae.__class__.__name__ != vae_class_name:
		raise RuntimeError(
			f"Checkpoint VAE class {vae_class_name} does not match instantiated {vae.__class__.__name__}."
		)

	_dalle = DALLE(vae=vae, **dalle_params).to(_device)
	_dalle.load_state_dict(weights)
	_dalle.eval()

	_tokenizer = _load_tokenizer_from_env()
	_model_loaded = True


@app.get("/")
async def index():
	return {
		"message": "DALLE-pytorch Image Generator API",
		"health": "/healthz",
		"generate": {
			"method": "POST",
			"path": "/generate",
			"body": {
				"text": "a red apple on a wooden table",
				"num_images": 4,
				"batch_size": 2,
				"top_k": 0.9
			}
		}
	}


@app.get("/healthz")
async def healthz():
	return {
		"status": "ok",
		"device": str(_device),
		"model_loaded": _model_loaded,
	}


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
	try:
		_load_model_if_needed()
	except Exception as e:
		raise HTTPException(status_code=500, detail=str(e))

	assert _dalle is not None

	texts = req.text.split("|")
	if len(texts) != 1:
		raise HTTPException(status_code=400, detail="Multiple prompts via '|' are not supported in the API. Provide a single prompt.")
	text = texts[0]

	image_urls: List[str] = []
	output_dir = OUTPUTS_DIR / f"{_safe_dirname_from_text(text)}_{uuid.uuid4().hex[:8]}"
	output_dir.mkdir(parents=True, exist_ok=True)

	with torch.no_grad():
		if req.gentxt:
			text_tokens, gen_texts = _dalle.generate_texts(_tokenizer, text=text, filter_thres=req.top_k)
			text = gen_texts[0]
		else:
			text_tokens = _tokenizer.tokenize([text], _dalle.text_seq_len).to(_device)

		text_tokens = repeat(text_tokens, '() n -> b n', b=req.num_images)

		outputs = []
		for text_chunk in text_tokens.split(req.batch_size):
			output = _dalle.generate_images(text_chunk, filter_thres=req.top_k)
			outputs.append(output)

		images = torch.cat(outputs, dim=0)

		for i, image in enumerate(images):
			out_path = output_dir / f"{i}.png"
			save_image(image, str(out_path), normalize=True)
			# URLs relative to server
			rel = out_path.relative_to(OUTPUTS_DIR)
			image_urls.append(f"/outputs/{rel.as_posix()}")

	# also save caption
	with open(output_dir / "caption.txt", "w") as f:
		f.write(text)

	return GenerateResponse(
		prompt=text,
		count=len(image_urls),
		output_dir=f"/outputs/{output_dir.relative_to(OUTPUTS_DIR).as_posix()}",
		image_urls=image_urls,
	)


if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)