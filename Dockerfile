FROM python:3.12-slim

# Version from pyproject.toml - keep in sync
ARG VERSION=0.1.0
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.title="semantic-kinematics-mcp"
LABEL org.opencontainers.image.description="MCP server for semantic analysis tools"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY semantic_kinematics/ semantic_kinematics/

# Install the package (base dependencies, no GPU)
RUN pip install --no-cache-dir .

# Environment defaults
ENV PYTHONUNBUFFERED=1

# MCP runs over stdio
CMD ["semantic-kinematics-mcp"]
