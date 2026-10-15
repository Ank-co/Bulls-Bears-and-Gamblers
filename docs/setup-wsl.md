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

The 35B model is served by a Windows build of llama.cpp (`llama-server`, listening on
`0.0.0.0:8080` so that WSL and Docker containers can reach it). The GPU has 8 GB: stop any PyTorch
job before starting the server.

From WSL, Windows is reachable at the default gateway address:

```bash
export LLAMA_URL="http://$(ip route show default | awk '{print $3}'):8080/v1/chat/completions"
export LLAMA_API_KEY="..."        # the key given to llama-server, never committed
curl -s -H "Authorization: Bearer $LLAMA_API_KEY" "${LLAMA_URL%/chat/completions}/models"
```

`0.0.0.0` also listens on the Wi-Fi and Ethernet adapters, where the API key would travel in clear
over HTTP. Close the port on those adapters only (PowerShell as administrator); WSL and Docker go
through virtual adapters, which this rule does not touch:

```powershell
$phys = (Get-NetAdapter -Physical).Name
New-NetFirewallRule -DisplayName "Block llama-server from network" -Direction Inbound -Protocol TCP -LocalPort 8080 -Action Block -InterfaceAlias $phys
```

The rule lists the adapters present when it is created: run it again after adding a new one (a
dock, USB tethering).

If nothing else needs the server (no Docker container), a simpler option is to start it with
`--host 127.0.0.1` and enable mirrored networking in `%UserProfile%\.wslconfig`
(`[wsl2]` then `networkingMode=mirrored`, followed by `wsl --shutdown`); the default
`LLAMA_URL=http://127.0.0.1:8080/v1/chat/completions` then works. Docker Desktop does not support
mirrored networking well.

Keep llama.cpp up to date (it parses third-party GGUF files) and check each GGUF file against the
SHA-256 listed in the pre-registration before loading it:
`Get-FileHash <file>.gguf -Algorithm SHA256` in PowerShell.
