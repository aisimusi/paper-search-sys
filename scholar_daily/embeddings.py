"""Configurable local embeddings. Model libraries are imported only on encode()."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re


@dataclass(frozen=True)
class EmbeddingConfig:
    model_id: str = "intfloat/multilingual-e5-small"
    revision: str = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    dimension: int = 384
    max_length: int = 512
    batch_size: int = 4
    device: str = "cpu"
    dtype: str = "float32"
    query_prefix: str = "query: "
    document_prefix: str = "passage: "
    cache_folder: str | None = None
    local_files_only: bool = True

    def __post_init__(self):
        for name in ("model_id", "revision", "device"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(name + " 必须为非空字符串")
        for name in ("dimension", "max_length", "batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(name + " 必须为正整数")
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("不支持的 Embedding dtype")
        if not isinstance(self.query_prefix, str) or not isinstance(self.document_prefix, str):
            raise ValueError("Embedding 前缀必须为字符串")
        if self.cache_folder is not None and not isinstance(self.cache_folder, str):
            raise ValueError("cache_folder 必须为路径字符串或 None")
        if type(self.local_files_only) is not bool:
            raise ValueError("local_files_only 必须为布尔值")

    def signature(self) -> dict:
        """Identity of vector generation, independent of cache storage location."""
        value = asdict(self)
        value.pop("cache_folder")
        value.pop("local_files_only")
        value["normalization"] = "l2-float32-v1"
        value["encoder"] = "sentence-transformers; model-defined pooling; explicit prefixes"
        value["libraries"] = {}
        for name in ("sentence-transformers", "transformers", "torch"):
            try:
                value["libraries"][name] = version(name)
            except PackageNotFoundError:
                value["libraries"][name] = None
        local = Path(self.model_id)
        if local.is_dir():
            # Pin local weights too; compute once per DenseRetriever instance.
            files = {}
            for path in sorted(local.rglob("*")):
                if path.is_file() and path.suffix in {".json", ".safetensors", ".bin", ".model", ".txt"}:
                    with path.open("rb") as source:
                        fingerprint = hashlib.sha256()
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            fingerprint.update(block)
                        files[path.relative_to(local).as_posix()] = fingerprint.hexdigest()
            if not files:
                raise ValueError("本地 Embedding 模型目录为空")
            value["local_files"] = files
        return value


class SentenceTransformerEmbedder:
    """encode(texts, query=bool) is also the injectable embedding test interface."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            cfg = self.config
            if not Path(cfg.model_id).is_dir() and not re.fullmatch(r"[0-9a-f]{40}", cfg.revision):
                raise ValueError("远程 Embedding 模型 revision 必须固定为完整 commit SHA")
            model = SentenceTransformer(
                cfg.model_id, revision=cfg.revision, device=cfg.device,
                cache_folder=cfg.cache_folder, local_files_only=cfg.local_files_only,
                trust_remote_code=False, model_kwargs={"torch_dtype": cfg.dtype},
            )
            if model.get_embedding_dimension() != cfg.dimension:
                raise ValueError("模型输出维度与 dimension 配置不一致")
            if cfg.max_length > model.max_seq_length:
                raise ValueError("max_length 超过模型声明的输入长度限制")
            model.max_seq_length = cfg.max_length
            self._model = model
        return self._model

    def encode(self, texts: list[str], *, query: bool = False):
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding 输入必须为非空文本")
        prefix = self.config.query_prefix if query else self.config.document_prefix
        return self._load().encode(
            [prefix + text for text in texts], batch_size=self.config.batch_size,
            convert_to_numpy=True, normalize_embeddings=False, prompt="",
            show_progress_bar=False,
        )
