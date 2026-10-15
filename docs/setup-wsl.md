# Environment (Windows 11 + WSL2, RTX 50-series laptop GPU)

The repository lives on the Windows drive and is used from WSL2. The Python environment and the
Hugging Face cache stay on the Linux filesystem, where file access is fast.

## 1. WSL2 with Ubuntu (PowerShell)

```powershell
wsl -l -v                      # Ubuntu listed with VERSION 2? Then skip the next line
wsl --install -d Ubuntu-24.04  # reboot if asked, then create a Linux user
```

Do **not** install an NVIDIA driver inside Ubuntu: the Windows driver already exposes the GPU to WSL.

## 2. Everything else (Ubuntu)

```bash
cd /mnt/c/Users/<you>/Documents/bulls-bears-and-gamblers
bash setup/setup_wsl.sh
```

The script creates `~/.venvs/bbg`, installs PyTorch built for CUDA 12.8 (RTX 50-series GPUs need
it), installs the project, writes `requirements.lock.txt`, runs the unit tests and ends with
`scripts/check_gpu.py`, which must print `OK GPU ready`.

## 3. Reaching the llama.cpp server on Windows from WSL

The 35B model is served by a Windows build of llama.cpp (`llama-server`). The GPU has 8 GB: stop
any PyTorch job before starting the server.

The server listens on `127.0.0.1` only, so no other device on the network can reach it. WSL reaches
it through mirrored networking: create `%UserProfile%\.wslconfig` with

```ini
[wsl2]
networkingMode=mirrored
```

then run `wsl --shutdown` in PowerShell and reopen Ubuntu. Start the server with
`--host 127.0.0.1 --port 8080 --api-key <key>`, and from WSL:

```bash
export LLAMA_URL="http://127.0.0.1:8080/v1/chat/completions"
export LLAMA_API_KEY="..."        # the key given to llama-server, never committed
curl -s -H "Authorization: Bearer $LLAMA_API_KEY" "${LLAMA_URL%/chat/completions}/models"
```

Do not start the server on `0.0.0.0`: that address listens on the Wi-Fi and Ethernet adapters too,
where the API key travels in clear over HTTP.

Keep llama.cpp up to date (it parses third-party GGUF files) and check each GGUF file against the
SHA-256 listed in the pre-registration before loading it:
`Get-FileHash <file>.gguf -Algorithm SHA256` in PowerShell.
