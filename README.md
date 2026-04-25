# DeepSeek2API-rs (ds2api-rs)

[![](https://img.shields.io/github/license/iidamie/deepseek2api.svg)](LICENSE)

A high-performance Rust port of the original DeepSeek2API. This project bridges the web-based DeepSeek chat interface to standard API formats. 

It supports high-speed streaming output, multi-turn conversations, DeepSeek-R1 "Deep Think" reasoning, multi-account rotation, and tool function calling. 

**Fully compatible with the OpenAI API format and includes preliminary support for the Anthropic (Claude) API format.**

## Table of Contents

* [Disclaimer](#disclaimer)
* [Features](#features)
* [Preparation](#preparation)
  * [Multi-Account Setup](#multi-account-setup)
* [Deployment](#deployment)
  * [Native (Cargo)](#native-cargo)
  * [Docker](#docker)
  * [Docker Compose](#docker-compose)
* [API Endpoints](#api-endpoints)
  * [Supported Models](#supported-models)
  * [OpenAI Chat Completions](#openai-chat-completions)
  * [Anthropic Messages](#anthropic-messages)
* [Notes](#notes)
  * [Nginx Reverse Proxy Optimization](#nginx-reverse-proxy-optimization)
* [Acknowledgements](#acknowledgements)

## Disclaimer

**Reverse-engineered APIs are inherently unstable. It is highly recommended to use the official DeepSeek API at https://platform.deepseek.com/ to avoid the risk of account bans.**

**This organization and its contributors do not accept any financial donations or commercial transactions. This project is purely for educational, research, and learning purposes!**

**STRICTLY FOR PERSONAL USE ONLY. Do not use this to provide public services or for commercial purposes to avoid putting pressure on official servers. Use at your own risk!**

## Features

* **Rust Rewite:** Built with Axum and Tokio for maximum performance and minimal footprint.
* **Native PoW Solver:** Uses `wasmtime` to efficiently solve DeepSeek's WebAssembly Proof-of-Work (PoW) challenges.
* **OpenAI Compatibility:** Drop-in replacement for OpenAI endpoints (`/v1/chat/completions`).
* **Anthropic Compatibility:** Translate Claude API requests directly to DeepSeek (`/anthropic/v1/messages`).
* **Web Search:** Support for DeepSeek's web search toggles.

## Preparation

You will need one or more registered DeepSeek accounts.

### Multi-Account Setup

Currently, a single account can only stream one response at a time. To handle concurrent requests, you can configure multiple accounts. The service will automatically rotate through available accounts for each request.

Create a `config.json` file in the root directory:

```json
{
  "bind_address": "0.0.0.0:5001",
  "cors_origins": ["*"],
  "keys": [
    "sk-your-custom-api-key-1",
    "sk-your-custom-api-key-2"
  ],
  "accounts": [
    {
      "email": "example1@example.com",
      "password": "password1",
      "token": ""
    },
    {
      "mobile": "12345678901",
      "password": "password3",
      "token": ""
    }
  ]
}
```
* `bind_address`: The host and port the server will bind to.
* `keys`: Your custom API authentication keys (used as Bearer tokens in your requests).
* `accounts`: A list of DeepSeek credentials. The system supports email/password or mobile/password logins. Tokens will be automatically fetched and refreshed.

*Note: The Wasm PoW file (`sha3_wasm_bg.7b9ca65ddd.wasm`) must be present in the root directory alongside the config.*

## Deployment

### Native (Cargo)

Ensure you have Rust and Cargo installed. 

```shell
git clone [https://github.com/your-username/ds2api-rs.git](https://github.com/your-username/ds2api-rs.git)
cd ds2api-rs
```

Ensure `config.json` and `sha3_wasm_bg.7b9ca65ddd.wasm` are in the current directory, then run:

```shell
cargo run --release
```

### Docker

If you have built a Docker image (e.g., `ds2api-rs:latest`), you can run it via Docker. Ensure your `config.json` is ready.

```shell
docker run -d -p 5001:5001 \
  -v "$(pwd)/config.json:/app/config.json" \
  --name deepseek2api \
  ds2api-rs:latest
```

View real-time logs:
```shell
docker logs -f deepseek2api
```

### Docker Compose

Clone the repository and prepare your `config.json`.

```shell
docker-compose up -d
```

## API Endpoints

### Supported Models

You can use the following model strings in your requests:

* **Standard Chat:** `deepseek-chat` or `deepseek-v3`
* **Deep Thinking (R1):** `deepseek-reasoner` or `deepseek-r1`
* **Chat + Web Search:** `deepseek-chat-search` or `deepseek-v3-search`
* **Deep Thinking + Web Search:** `deepseek-reasoner-search` or `deepseek-r1-search`

### OpenAI Chat Completions

**POST `/v1/chat/completions`**

Headers:
```http
Authorization: Bearer <YOUR_CUSTOM_KEY_FROM_CONFIG>
Content-Type: application/json
```

Request body:
```json
{
    "model": "deepseek-reasoner",
    "messages": [
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
    "stream": true
}
```

### Anthropic Messages

**POST `/anthropic/v1/messages`** (or `/claude/messages`)

*Note: Streaming is not yet supported for the Anthropic translation layer in this Rust implementation.*

Headers:
```http
Authorization: Bearer <YOUR_CUSTOM_KEY_FROM_CONFIG>
x-api-key: <YOUR_CUSTOM_KEY_FROM_CONFIG>
Content-Type: application/json
```

Request body:
```json
{
    "model": "claude-3-5-sonnet-20241022",
    "system": "You are a helpful assistant.",
    "messages": [
        {
            "role": "user",
            "content": "Who are you?"
        }
    ],
    "stream": false
}
```

## Notes

### Nginx Reverse Proxy Optimization

If you are using Nginx to reverse proxy this API, please add the following configuration to optimize Server-Sent Events (SSE) streaming output.

```nginx
# Disable proxy buffering. This ensures Nginx sends chunks from the backend to the client immediately.
proxy_buffering off;
# Enable chunked transfer encoding.
chunked_transfer_encoding on;
# Send data as soon as possible.
tcp_nopush on;
# Do not delay sending small packets.
tcp_nodelay on;
# Keep-alive timeout.
keepalive_timeout 120;
```

## Acknowledgements

This Rust port is based on the original Python implementation of `deepseek2api`. 
Special thanks to the [LLM-Red-Team/deepseek-free-api](https://github.com/LLM-Red-Team/deepseek-free-api) project, which provided foundational references for the reverse engineering implementation.
