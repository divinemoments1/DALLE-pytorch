# Image Creator API - Deployment

## Prerequisites
- Docker 24+
- (Optional) NVIDIA GPU with recent drivers
- (Optional) `docker compose` plugin or Docker Desktop
- A trained DALL-E checkpoint file

## Prepare data directory
```
mkdir -p data/checkpoints data/outputs
# copy your checkpoint
cp /path/to/your/dalle.pt data/checkpoints/dalle.pt
# optional tokenizer or vqgan files
# cp model.bpe data/tokenizer/model.bpe
# cp vqgan.ckpt data/checkpoints/vqgan.ckpt
# cp vqgan.yaml data/checkpoints/vqgan.yaml
```

## Build API image
```
docker build -f docker/api.Dockerfile -t dalle-api:local .
```

## Run with GPU (recommended)
```
docker run --rm -it \
  --gpus all \
  -p 8000:8000 \
  -e DALLE_WEIGHTS=/data/checkpoints/dalle.pt \
  -e OUTPUTS_DIR=/data/outputs \
  -v $(pwd)/data:/data \
  dalle-api:local
```

## Or with Compose
```
docker compose up --build
```

## Test
- Health: `curl http://localhost:8000/healthz`
- Generate:
```
curl -X POST http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"a red apple on a wooden table","num_images":4,"batch_size":2,"top_k":0.9}'
```
- Download an image: open any `image_urls` from the response in your browser.

## Environment variables
- `DALLE_WEIGHTS` (required): path in container to the checkpoint file
- `OUTPUTS_DIR` (default: `/data/outputs`)
- `TAMING` (optional): `true` to enable VQGAN
- `VQGAN_MODEL_PATH`, `VQGAN_CONFIG_PATH` (required if TAMING=true)
- `BPE_PATH` (optional): tokenizer model path
- `BPE_KIND` (optional): `yttm` (default) or `hug`
- `CHINESE` (optional): `true` for Chinese tokenizer