# Additional API Triggers for GitHub Repository Re-indexing

## Overview

This document describes the API endpoints and webhooks for triggering re-indexing of GitHub repositories in Verba's RAG system. These endpoints allow external systems (including GitHub webhooks) to notify Verba when repository content has changed and needs to be re-indexed.

## Core Re-indexing Endpoints

### 1. Trigger Full Re-index

**Endpoint**: `POST /api/reindex`

Triggers a complete re-indexing of specified GitHub repositories.

```python
# Request Body
{
    "repositories": [
        {
            "url": "https://github.com/owner/repo",
            "branch": "main",  # Optional, defaults to main/master
            "paths": ["/docs", "/src"]  # Optional, specific paths to index
        }
    ],
    "config": {
        "chunker": "MarkdownChunker",  # Optional, chunker to use
        "embedder": "OpenAIEmbedder",  # Optional, embedder to use
        "force": true  # Force re-index even if content hasn't changed
    }
}

# Response
{
    "status": "accepted",
    "job_id": "reindex_12345",
    "repositories_queued": 1,
    "estimated_time": 120  # seconds
}
```

### 2. Incremental Update

**Endpoint**: `POST /api/update`

Updates only changed files in repositories.

```python
# Request Body
{
    "repository": "https://github.com/owner/repo",
    "changes": [
        {
            "file": "docs/README.md",
            "action": "modified"  # added, modified, deleted
        }
    ],
    "commit": "abc123def",  # Git commit hash
    "timestamp": "2024-01-15T10:30:00Z"
}

# Response
{
    "status": "processing",
    "job_id": "update_67890",
    "files_affected": 1
}
```

### 3. GitHub Webhook Handler

**Endpoint**: `POST /api/github/webhook`

Receives GitHub webhook events and triggers appropriate re-indexing.

```python
# Headers required
{
    "X-GitHub-Event": "push",
    "X-Hub-Signature-256": "sha256=..."  # HMAC signature for verification
}

# GitHub Push Event Payload (simplified)
{
    "ref": "refs/heads/main",
    "repository": {
        "full_name": "owner/repo",
        "clone_url": "https://github.com/owner/repo.git"
    },
    "commits": [
        {
            "id": "abc123",
            "added": ["file1.md"],
            "modified": ["file2.md"],
            "removed": ["file3.md"]
        }
    ]
}
```

### 4. Check Re-indexing Status

**Endpoint**: `GET /api/reindex/status/{job_id}`

```python
# Response
{
    "job_id": "reindex_12345",
    "status": "in_progress",  # pending, in_progress, completed, failed
    "progress": {
        "total_files": 150,
        "processed_files": 75,
        "percentage": 50
    },
    "errors": [],
    "started_at": "2024-01-15T10:30:00Z",
    "estimated_completion": "2024-01-15T10:32:00Z"
}
```

## Implementation in Verba

### Backend Implementation Location

The re-indexing endpoints are implemented in the FastAPI server:

```python
# goldenverba/server/api.py

from fastapi import FastAPI, BackgroundTasks, HTTPException
from goldenverba.server.types import ReindexRequest, ReindexStatus

@app.post("/api/reindex")
async def trigger_reindex(
    request: ReindexRequest,
    background_tasks: BackgroundTasks,
    verba_manager: VerbaManager = Depends(get_verba_manager)
):
    """Trigger repository re-indexing"""
    job_id = generate_job_id()
  
    # Queue background task
    background_tasks.add_task(
        reindex_repositories,
        job_id,
        request.repositories,
        request.config,
        verba_manager
    )
  
    return {
        "status": "accepted",
        "job_id": job_id,
        "repositories_queued": len(request.repositories)
    }

@app.post("/api/github/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    verba_manager: VerbaManager = Depends(get_verba_manager)
):
    """Handle GitHub webhook events"""
  
    # Verify webhook signature
    signature = request.headers.get("X-Hub-Signature-256")
    if not verify_github_signature(await request.body(), signature):
        raise HTTPException(status_code=401, detail="Invalid signature")
  
    event_type = request.headers.get("X-GitHub-Event")
    payload = await request.json()
  
    if event_type == "push":
        # Process push event
        job_id = process_github_push(payload, background_tasks, verba_manager)
        return {"status": "accepted", "job_id": job_id}
  
    return {"status": "ignored", "reason": f"Event type {event_type} not handled"}
```

### Re-indexing Task Implementation

```python
# goldenverba/server/tasks.py

async def reindex_repositories(
    job_id: str,
    repositories: list[dict],
    config: dict,
    verba_manager: VerbaManager
):
    """Background task for re-indexing repositories"""
  
    status_tracker = ReindexStatusTracker(job_id)
  
    try:
        for repo in repositories:
            # Clone or update repository
            repo_path = await clone_or_update_repo(repo["url"], repo.get("branch"))
          
            # Get files to index
            files = get_files_to_index(repo_path, repo.get("paths", []))
          
            # Create documents from files
            documents = []
            for file_path in files:
                doc = create_document_from_file(file_path)
                documents.append(doc)
          
            # Process through RAG pipeline
            chunker = verba_manager.chunker_manager.get(config.get("chunker", "RecursiveChunker"))
            embedder = verba_manager.embedding_manager.get(config.get("embedder", "OpenAIEmbedder"))
          
            # Chunk documents
            documents = await chunker.chunk(config, documents, embedder)
          
            # Generate embeddings
            documents = await embedder.embed(config, documents)
          
            # Store in Weaviate
            await verba_manager.weaviate_manager.batch_insert_documents(documents)
          
            status_tracker.update_progress(repo["url"], "completed")
  
    except Exception as e:
        status_tracker.mark_failed(str(e))
        raise
```

## GitHub Integration Configuration

### Setting Up GitHub Webhooks

1. In your GitHub repository, go to Settings → Webhooks
2. Add webhook with URL: `https://your-verba-instance.com/api/github/webhook`
3. Set content type to `application/json`
4. Generate and set a secret token
5. Select events: Push, Pull Request (merged), Release

### Environment Configuration

```bash
# .env file
GITHUB_WEBHOOK_SECRET=your_webhook_secret_here
GITHUB_ACCESS_TOKEN=ghp_your_personal_access_token  # For private repos
REINDEX_BATCH_SIZE=50
REINDEX_WORKER_THREADS=4
```

### Client Configuration

The frontend can configure repositories through the settings interface:

```javascript
// Frontend configuration example
const repoConfig = {
    repositories: [
        {
            url: "https://github.com/owner/repo1",
            branch: "main",
            autoIndex: true,
            indexFrequency: "on_push",  // on_push, hourly, daily
            filePatterns: ["*.md", "*.py", "*.js"],
            excludePatterns: ["test/*", "*.test.js"]
        }
    ],
    indexingSettings: {
        chunker: "MarkdownChunker",
        chunkSize: 1000,
        chunkOverlap: 200,
        embedder: "OpenAIEmbedder"
    }
};
```

## Webhook Security

### Signature Verification

```python
import hmac
import hashlib

def verify_github_signature(payload: bytes, signature: str) -> bool:
    """Verify GitHub webhook signature"""
    if not signature:
        return False
  
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "").encode()
    expected_signature = "sha256=" + hmac.new(
        secret,
        payload,
        hashlib.sha256
    ).hexdigest()
  
    return hmac.compare_digest(expected_signature, signature)
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/reindex")
@limiter.limit("10/hour")  # Limit to 10 reindex requests per hour
async def trigger_reindex(...):
    # Implementation
```

## Testing Webhooks Locally

For local development, use ngrok to expose your local server:

```bash
# Install ngrok
brew install ngrok  # macOS
# or download from https://ngrok.com

# Expose local server
ngrok http 8000

# Use the HTTPS URL provided by ngrok in GitHub webhook settings
```

### Manual Testing

```bash
# Trigger re-index manually
curl -X POST http://localhost:8000/api/reindex \
  -H "Content-Type: application/json" \
  -d '{
    "repositories": [
      {
        "url": "https://github.com/owner/repo",
        "branch": "main"
      }
    ],
    "config": {
      "force": true
    }
  }'

# Check status
curl http://localhost:8000/api/reindex/status/reindex_12345
```

## Error Handling

### Common Error Responses

```json
// Repository not accessible
{
    "error": "REPO_ACCESS_DENIED",
    "message": "Cannot access repository. Check permissions and authentication.",
    "repository": "https://github.com/owner/private-repo"
}

// Rate limit exceeded
{
    "error": "RATE_LIMIT_EXCEEDED",
    "message": "GitHub API rate limit exceeded. Retry after 3600 seconds.",
    "retry_after": 3600
}

// Invalid configuration
{
    "error": "INVALID_CONFIG",
    "message": "Invalid chunker specified: UnknownChunker",
    "valid_options": ["TokenChunker", "SentenceChunker", "RecursiveChunker"]
}
```

## Monitoring and Logging

### Metrics to Track

- Re-indexing request rate
- Average processing time per repository
- Number of documents/chunks created
- Embedding generation time
- Weaviate insertion rate
- Error rate by type

### Logging Configuration

```python
import logging
from goldenverba.server.helpers import LoggerManager

logger = LoggerManager("ReindexAPI")

# Log important events
await logger.info(f"Re-indexing started for {repo_url}")
await logger.warn(f"Rate limit approaching for GitHub API")
await logger.error(f"Failed to index repository: {error}")
```

## See Also

- [RAG_DEV.md](./RAG_DEV.md) - Complete RAG development guide
- [DEVELOPMENT.md](./DEVELOPMENT.md) - GitHub integration development guide
- [API.md](./API.md) - Complete API documentation
