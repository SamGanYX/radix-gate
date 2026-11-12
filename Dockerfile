FROM rust:1.80-slim as builder
WORKDIR /app

# Cache dependencies
COPY Cargo.toml ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs && echo "pub fn dummy() {}" > src/lib.rs && cargo build --release || true

# Copy source tree and compile release binary
COPY src ./src
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY --from=builder /app/target/release/radix-gate /usr/local/bin/radix-gate

EXPOSE 8080 9090
HEALTHCHECK --interval=5s --timeout=3s --retries=3 CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/radix-gate"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8080", "--mock-workers", "4"]
