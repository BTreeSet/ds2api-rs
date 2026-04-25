FROM rust:1.86-bookworm AS builder

WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/target/release/ds2api-rs /usr/local/bin/ds2api-rs
COPY config.json /app/config.json
COPY sha3_wasm_bg.7b9ca65ddd.wasm /app/sha3_wasm_bg.7b9ca65ddd.wasm

EXPOSE 5001
CMD ["/usr/local/bin/ds2api-rs"]
