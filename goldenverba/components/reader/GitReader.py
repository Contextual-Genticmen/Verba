import aiohttp
import os
import urllib
import base64

from wasabi import msg

from goldenverba.components.document import Document
from goldenverba.components.interfaces import Reader
from goldenverba.server.types import FileConfig
from goldenverba.components.reader.BasicReader import BasicReader
from goldenverba.components.util import get_environment

from goldenverba.components.types import InputConfig


class GitReader(Reader):
    """
    The GitReader downloads files from GitHub or GitLab and ingests them into Weaviate.
    """

    def __init__(self):
        super().__init__()
        self.name = "Git"
        self.type = "URL"
        self.description = (
            "Downloads and ingests all files from a GitHub or GitLab Repo."
        )
        self.config = {
            "Platform": InputConfig(
                type="dropdown",
                value="GitHub",
                description="Select the Git platform",
                values=["GitHub", "GitLab"],
            ),
            "Owner": InputConfig(
                type="text",
                value="",
                description="Enter the repo owner (GitHub) or group/user (GitLab)",
                values=[],
            ),
            "Name": InputConfig(
                type="text",
                value="",
                description="Enter the repo name",
                values=[],
            ),
            "Branch": InputConfig(
                type="text",
                value="main",
                description="Enter the branch name",
                values=[],
            ),
            "Path": InputConfig(
                type="text",
                value="",
                description="Enter the path or leave it empty to import all",
                values=[],
            ),
        }

        if os.getenv("GITHUB_TOKEN") is None and os.getenv("GITLAB_TOKEN") is None:
            self.config["Git Token"] = InputConfig(
                type="password",
                value="",
                description="You can set your GitHub/GitLab Token here if you haven't set it up as environment variable `GITHUB_TOKEN` or `GITLAB_TOKEN`",
                values=[],
            )

    async def load(self, config: dict, fileConfig: FileConfig) -> list[Document]:
        documents = []
        platform = config["Platform"].value
        token = self.get_token(config, platform)

        reader = BasicReader()

        if platform == "GitHub":
            owner = config["Owner"].value
            name = config["Name"].value
            branch = config["Branch"].value
            path = config["Path"].value
            fetch_url = f"https://api.github.com/repos/{owner}/{name}/git/trees/{branch}?recursive=1"
            docs = await self.fetch_docs_github(fetch_url, path, token, reader)
        else:  # GitLab
            owner = config["Owner"].value
            name = config["Name"].value
            project_id = urllib.parse.quote(f"{owner}/{name}", safe="")
            branch = config["Branch"].value
            path = config["Path"].value
            fetch_url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/tree?ref={branch}&path={path}&per_page=100"
            docs = await self.fetch_docs_gitlab(fetch_url, token, reader)

        msg.info(f"Fetched {len(docs)} document paths from {fetch_url}")

        for _file in docs:
            try:
                if platform == "GitHub":
                    content, link, size, extension = await self.download_file_github(
                        owner, name, _file, branch, token
                    )
                else:
                    content, link, size, extension = await self.download_file_gitlab(
                        owner, name, _file, branch, token
                    )

                if content:
                    # Extract file information for enhanced metadata
                    file_directory = os.path.dirname(_file) if os.path.dirname(_file) else "root"
                    file_basename = os.path.basename(_file)
                    file_name_without_ext = os.path.splitext(file_basename)[0]
                    
                    # Determine file category based on extension and path
                    file_category = self.categorize_file(_file, extension)
                    
                    # Create comprehensive metadata for RAG context
                    enhanced_metadata = {
                        # Git repository context
                        "git_platform": platform,
                        "repository_owner": owner,
                        "repository_name": name,
                        "repository_full_name": f"{owner}/{name}",
                        "branch": branch,
                        "commit_ref": branch,  # Could be enhanced with actual commit SHA
                        
                        # File location and structure
                        "file_path": _file,
                        "file_directory": file_directory,
                        "file_basename": file_basename,
                        "file_name": file_name_without_ext,
                        "file_extension": extension,
                        "file_category": file_category,
                        "file_size_bytes": size,
                        
                        # Content classification
                        "content_type": self.get_content_type(extension),
                        "is_documentation": self.is_documentation_file(_file, extension),
                        "is_source_code": self.is_source_code_file(extension),
                        "is_configuration": self.is_configuration_file(_file, extension),
                        
                        # RAG-specific metadata
                        "source_type": "git_repository",
                        "document_hierarchy": self.get_hierarchy_level(_file),
                        
                        # URLs for reference
                        "source_url": link,
                        "repository_url": f"https://{platform.lower()}.com/{owner}/{name}",
                        "raw_url": self.get_raw_url(platform, owner, name, branch, _file),
                        
                        # Import metadata
                        "imported_from": "GitReader",
                        "import_timestamp": fileConfig.fileID,  # Using fileID as timestamp
                    }
                    
                    new_file_config = FileConfig(
                        fileID=fileConfig.fileID,
                        filename=_file,
                        isURL=False,
                        overwrite=fileConfig.overwrite,
                        extension=extension,
                        source=link,
                        content=content,
                        labels=fileConfig.labels,
                        rag_config=fileConfig.rag_config,
                        file_size=size,
                        status=fileConfig.status,
                        status_report=fileConfig.status_report,
                        metadata=enhanced_metadata,
                    )
                    document = await reader.load(config, new_file_config)
                    documents.append(document[0])
            except Exception as e:
                raise Exception(f"Couldn't load or retrieve {_file}: {str(e)}")

        return documents

    def get_token(self, config: dict, platform: str) -> str:
        env_var = "GITHUB_TOKEN" if platform == "GitHub" else "GITLAB_TOKEN"
        return get_environment(
            config, "Git Token", env_var, f"No {platform} Token detected"
        )

    async def fetch_docs_github(
        self, url: str, folder: str, token: str, reader: Reader
    ) -> list[str]:
        headers = self.get_headers(token, "GitHub")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return [
                    item["path"]
                    for item in data["tree"]
                    if item["path"].startswith(folder)
                    and any(item["path"].endswith(ext) for ext in reader.extension)
                ]

    async def fetch_docs_gitlab(self, url: str, token: str, reader: Reader) -> list:
        headers = self.get_headers(token, "GitLab")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                return [
                    item["path"]
                    for item in data
                    if item["type"] == "blob"
                    and any(item["path"].endswith(ext) for ext in reader.extension)
                ]

    async def download_file_github(
        self, owner: str, name: str, path: str, branch: str, token: str
    ) -> tuple[str, str, int, str]:
        url = (
            f"https://api.github.com/repos/{owner}/{name}/contents/{path}?ref={branch}"
        )
        headers = self.get_headers(token, "GitHub")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                content_b64 = data["content"]
                link = data["html_url"]
                size = data["size"]
                extension = os.path.splitext(path)[1][1:]
                return content_b64, link, size, extension

    async def download_file_gitlab(
        self, owner: str, name: str, file_path: str, branch: str, token: str
    ) -> tuple[str, str, int, str]:
        project_id = urllib.parse.quote(f"{owner}/{name}", safe="")
        url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/files/{urllib.parse.quote(file_path, safe='')}/raw?ref={branch}"
        headers = {"PRIVATE-TOKEN": token}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    content = await response.read()
                    content_b64 = base64.b64encode(content).decode("utf-8")
                    size = len(content)
                    extension = os.path.splitext(file_path)[1][1:]
                    link = (
                        f"https://gitlab.com/{owner}/{name}/-/blob/{branch}/{file_path}"
                    )
                    return content_b64, link, size, extension
                else:
                    raise Exception(
                        f"Failed to download file: {response.status} {await response.text()}"
                    )

    def get_headers(self, token: str, platform: str) -> dict:
        if platform == "GitHub":
            return {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json",
            }
        else:  # GitLab
            return {
                "Authorization": f"Bearer {token}",
            }

    def categorize_file(self, file_path: str, extension: str) -> str:
        """Categorize file based on path and extension for better RAG context."""
        file_path_lower = file_path.lower()
        extension_lower = extension.lower()
        
        # Documentation files
        if any(doc_indicator in file_path_lower for doc_indicator in ['readme', 'doc', 'wiki', 'guide', 'tutorial']):
            return "documentation"
        
        # Configuration files
        if any(config_indicator in file_path_lower for config_indicator in ['.github', '.vscode', 'config', 'settings']):
            return "configuration"
        
        # Test files
        if any(test_indicator in file_path_lower for test_indicator in ['test', 'spec', '__test__']):
            return "test"
        
        # Source code by extension
        if extension_lower in ['py', 'js', 'ts', 'java', 'cpp', 'c', 'go', 'rs', 'php', 'rb']:
            return "source_code"
        
        # Data files
        if extension_lower in ['json', 'yaml', 'yml', 'xml', 'csv', 'sql']:
            return "data"
        
        # Build/Deploy files
        if extension_lower in ['dockerfile', 'makefile'] or 'docker' in file_path_lower:
            return "deployment"
        
        return "other"

    def get_content_type(self, extension: str) -> str:
        """Determine content type for RAG processing optimization."""
        extension_lower = extension.lower()
        
        # Text-based content types
        text_extensions = ['md', 'txt', 'rst', 'adoc']
        code_extensions = ['py', 'js', 'ts', 'java', 'cpp', 'c', 'go', 'rs', 'php', 'rb', 'html', 'css']
        data_extensions = ['json', 'yaml', 'yml', 'xml', 'csv']
        config_extensions = ['conf', 'cfg', 'ini', 'toml']
        
        if extension_lower in text_extensions:
            return "text_document"
        elif extension_lower in code_extensions:
            return "source_code"
        elif extension_lower in data_extensions:
            return "structured_data"
        elif extension_lower in config_extensions:
            return "configuration"
        else:
            return "unknown"

    def is_documentation_file(self, file_path: str, extension: str) -> bool:
        """Check if file is likely documentation for enhanced RAG retrieval."""
        doc_extensions = ['md', 'txt', 'rst', 'adoc']
        doc_keywords = ['readme', 'doc', 'guide', 'tutorial', 'manual', 'wiki', 'help']
        
        return (extension.lower() in doc_extensions or 
                any(keyword in file_path.lower() for keyword in doc_keywords))

    def is_source_code_file(self, extension: str) -> bool:
        """Check if file is source code for code-specific RAG processing."""
        code_extensions = ['py', 'js', 'ts', 'tsx', 'jsx', 'java', 'cpp', 'c', 'h', 
                          'go', 'rs', 'php', 'rb', 'swift', 'kt', 'scala', 'html', 'css']
        return extension.lower() in code_extensions

    def is_configuration_file(self, file_path: str, extension: str) -> bool:
        """Check if file is configuration for infrastructure context."""
        config_extensions = ['json', 'yaml', 'yml', 'toml', 'ini', 'conf', 'cfg']
        config_keywords = ['config', 'settings', '.env', 'dockerfile', 'makefile', '.github']
        
        return (extension.lower() in config_extensions or 
                any(keyword in file_path.lower() for keyword in config_keywords))

    def get_hierarchy_level(self, file_path: str) -> str:
        """Determine document hierarchy for RAG context organization."""
        depth = len(file_path.split('/')) - 1
        
        if depth == 0:
            return "root"
        elif depth == 1:
            return "top_level"
        elif depth <= 3:
            return "mid_level"
        else:
            return "deep_nested"

    def get_raw_url(self, platform: str, owner: str, name: str, branch: str, file_path: str) -> str:
        """Generate raw content URL for direct file access."""
        if platform == "GitHub":
            return f"https://raw.githubusercontent.com/{owner}/{name}/{branch}/{file_path}"
        else:  # GitLab
            project_id = urllib.parse.quote(f"{owner}/{name}", safe="")
            encoded_path = urllib.parse.quote(file_path, safe='')
            return f"https://gitlab.com/{owner}/{name}/-/raw/{branch}/{file_path}"
