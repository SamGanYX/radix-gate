# Multi-stage production build for RadixGate
FROM rust:slim AS builder
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock* ./
RUN mkdir -p src && echo "pub fn dummy() {}" > src/lib.rs && echo "fn main() {}" > src/main.rs
RUN cargo build --release || true

# Copy source tree and compile optimized release binary
COPY src ./src
COPY tests ./tests
RUN touch src/lib.rs src/main.rs && cargo build --release

# Minimal runtime image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY --from=builder /app/target/release/radix-gate /usr/local/bin/radix-gate

EXPOSE 8080 9090
HEALTHCHECK --interval=5s --timeout=3s --retries=3 CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/radix-gate"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--mock-workers", "4"]
