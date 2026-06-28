# From fine-tuned Mistral to Hugging Face to Ollama

Ollama runs models in GGUF format, not Hugging Face safetensors. So the pipeline is:
merge the adapter, push to HF, convert to GGUF, then load in Ollama. The whole
chain runs on your machine; only the upload step touches the internet.

```
QLoRA adapter ──merge──> fp16 model ──push──> Hugging Face
                              │
                              └──convert+quantize──> GGUF ──> Ollama
```

## Step 1: Merge the adapter into a standalone model

The LoRA adapter alone cannot be converted. Produce a merged fp16 model:

```bash
python finetune_mistral_qlora.py --train_file data/train.jsonl --merge_after
# result: ./mistral7b-qlora-out/merged   (full fp16 model, ~15 GB)
```

If you already trained without `--merge_after`, just rerun with the flag, or call
the `merge_adapter(...)` function in the script on the saved adapter.

## Step 2: Push to Hugging Face (optional but matches your plan)

Create a WRITE token at https://huggingface.co/settings/tokens, then:

```bash
export HF_TOKEN=hf_xxx
python push_to_hub.py --src ./mistral7b-qlora-out/merged --repo_id your-username/mistral7b-myfinetune --private
```

Use `--private` if the model was trained on anything sensitive (NOW-ISMS-AI-001).

## Step 3: Convert to GGUF and quantize (llama.cpp)

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
pip install -r requirements.txt
cmake -B build && cmake --build build --config Release   # builds llama-quantize

# convert the merged HF model to a 16-bit GGUF
python convert_hf_to_gguf.py /path/to/mistral7b-qlora-out/merged \
    --outfile mistral-finetuned-f16.gguf --outtype f16

# quantize to Q4_K_M (good size/quality balance, ~4 GB, runs well on the 3060)
./build/bin/llama-quantize mistral-finetuned-f16.gguf mistral-finetuned-Q4_K_M.gguf Q4_K_M
```

Put the resulting `.gguf` in a `gguf/` folder next to the Modelfile.

## Step 4a: Load in Ollama from a local GGUF (no HF needed)

```bash
ollama create my-mistral -f Modelfile
ollama run my-mistral
```

The provided `Modelfile` points `FROM` at `./gguf/mistral-finetuned-Q4_K_M.gguf` and
sets the Mistral instruct chat template plus stop tokens.

## Step 4b: Load in Ollama by pulling from Hugging Face

This matches your idea of "update in HF, download through Ollama". Upload the GGUF
to a HF repo, then pull it directly. No Modelfile needed.

```bash
# upload just the GGUF file(s)
python push_to_hub.py --src ./gguf --repo_id your-username/mistral7b-myfinetune-gguf

# pull and run straight from HF (Ollama matches the quant by tag)
ollama run hf.co/your-username/mistral7b-myfinetune-gguf:Q4_K_M
```

## Notes

- Repo sizes: merged fp16 model is ~15 GB; the Q4_K_M GGUF is ~4 GB. For Ollama you
  only need the GGUF repo, so uploading just the GGUF is faster and cheaper.
- Quantization choices: `Q4_K_M` (recommended default), `Q5_K_M` (a bit larger,
  slightly better quality), `Q8_0` (largest, near-fp16 quality).
- Alternative: recent Ollama can also import a safetensors model directly with
  `FROM /path/to/merged` in a Modelfile, but GGUF is the most reliable and portable.
```
