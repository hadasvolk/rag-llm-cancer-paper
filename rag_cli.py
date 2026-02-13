#!/usr/bin/env python3
"""CLI for running MOAlmanac RAG-LLM with hybrid retrieval (Azure OpenAI).

This script reads credentials/config from a `.env` file (default: `.env`) and
assumes an Azure OpenAI *OpenAI-compatible* endpoint.

Required in `.env`:
    - AZURE_OPENAI_API_KEY=...
    - AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
        OR AZURE_OPENAI_BASE_URL=https://<resource>.openai.azure.com/openai/v1/

Optional in `.env`:
    - AZURE_OPENAI_DEPLOYMENT=<chat deployment name>
    - AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding deployment name>

Example:
    python rag_cli.py \
        --context_db fda \
        --context_db_type structured \
        --strategy 5 \
        --temp 0.0 \
        --max_tokens 2048 \
        --query "What FDA-approved therapies are available for ...?"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import faiss
import requests

from openai import OpenAI

from context_retriever.entity_prediction import extract_entities
from context_retriever.entity_prediction import db_extract_entities
from context_retriever.hybrid_search import retrieve_context_hybrid
from llm.inference import run_llm
from utils.embedding import index_context_db
from utils.flatten_statement import flatten_statements, extract_biomarker_info, extract_therapy_info
from utils.prompt import get_prompt


DEFAULT_EMBED_MODEL = "text-embedding-3-small"
DEFAULT_NUM_VEC = 25


def _get_remote_version() -> str:
    agents = requests.get("https://api.moalmanac.org/agents", timeout=60).json()
    return agents["service"]["last_updated"]


def _subset_statements_by_org(statements: list[dict], org: str) -> list[dict]:
    subset: list[dict] = []
    for statement in statements:
        reported = (statement.get("reportedIn") or [{}])[0]
        org_id = (reported.get("organization") or {}).get("id")
        agent_id = (reported.get("agent") or {}).get("id")
        if org_id == org or agent_id == org:
            subset.append(statement)
    return subset


def _load_disease_modifiers() -> list[str]:
    path = Path("data/latest_db/disease_modifiers__2025-09-04.json")
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_clinical_modifier(raw_cancer_type: str, standardized_cancer_type: str, modifiers: list[str]) -> Optional[str]:
    raw_lower = (raw_cancer_type or "").lower()
    std_lower = (standardized_cancer_type or "").lower()
    extracted = [m for m in modifiers if m in raw_lower and m not in std_lower]
    if not extracted:
        return None
    return max(extracted, key=len)


def _build_context_strings(
    statements: list[dict],
    *,
    db_type: str,
    modifiers: list[str],
) -> tuple[list[str], list[dict]]:
    """Return (contexts, entity_rows) where entity_rows contain fields needed for db_extract_entities."""
    contexts: list[str] = []
    entity_rows: list[dict] = []

    for stmt in statements:
        standardized_cancer = (
            stmt.get("proposition", {})
            .get("conditionQualifier", {})
            .get("name", "Unknown cancer")
        )
        raw_cancer = (stmt.get("indication", {}) or {}).get("raw_cancer_type") or standardized_cancer
        modifier = _extract_clinical_modifier(raw_cancer, standardized_cancer, modifiers)
        if modifier:
            modified_standardized_cancer = f"{modifier} {standardized_cancer.lower()}"
        else:
            modified_standardized_cancer = standardized_cancer.lower()

        biomarker_list = extract_biomarker_info(stmt)["list"]
        biomarker_text = ", ".join([b for b in biomarker_list if isinstance(b, str) and b])
        therapy_info = extract_therapy_info(stmt)["list"]
        drug_list = therapy_info["drugList"]
        therapy_text = " + ".join([d for d in drug_list if isinstance(d, str) and d])

        # Build contexts
        if db_type == "structured":
            basic = (
                f"if a patient with {modified_standardized_cancer} has {biomarker_text.lower()}, "
                f"one recommended therapy is {therapy_text.lower()}."
            )
            summary, row = flatten_statements(stmt)
            therapy_type = row.get("therapy_type") or []
            therapy_strategy = row.get("therapy_strategy") or []
            therapy_type_str = " + ".join([t for t in therapy_type if isinstance(t, str) and t])
            therapy_strategy_str = " + ".join([t for t in therapy_strategy if isinstance(t, str) and t])
            indication = (row.get("indication") or "").strip()
            approval_url = (row.get("approval_url") or "").strip()

            contexts.append(
                f"{basic} therapy type: {therapy_type_str.lower()}. "
                f"therapy strategy: {therapy_strategy_str.lower()}. "
                f"indication: {indication.lower()} approval url: {approval_url}"
            )
        elif db_type == "unstructured":
            summary, _ = flatten_statements(stmt)
            contexts.append(summary)
        else:
            raise ValueError("context_db_type must be structured or unstructured")

        entity_rows.append(
            {
                "modified_standardized_cancer": modified_standardized_cancer,
                "biomarker": biomarker_text,
            }
        )

    return contexts, entity_rows


def build_assets_if_missing(
    *,
    db: str,
    db_type: str,
    embed_name: str,
    client: OpenAI,
    force_rebuild: bool,
    version: Optional[str] = None,
) -> str:
    """Build local FAISS+JSON context assets and entity cache for fda/ema.

    Returns the version string used.
    """
    if db not in {"fda", "ema"}:
        raise ValueError("Auto-build is supported only for fda/ema")

    used_version = version or _get_remote_version()
    index_path, ctx_path = _cache_paths(
        output_dir="data/latest_db/indexes",
        embed_name=embed_name,
        db=db,
        name=f"{db_type}_context",
        version=used_version,
    )
    entity_path = Path("context_retriever/entities") / f"moalmanac_{db}_ner_entities__{used_version}.json"

    if not force_rebuild and os.path.exists(index_path) and os.path.exists(ctx_path) and entity_path.exists():
        return used_version

    all_statements = requests.get("https://api.moalmanac.org/statements", timeout=120).json()["data"]
    statements = _subset_statements_by_org(all_statements, db)
    if not statements:
        raise RuntimeError(f"No statements found for organization '{db}'")

    modifiers = _load_disease_modifiers()
    contexts, entity_rows = _build_context_strings(statements, db_type=db_type, modifiers=modifiers)

    # Build FAISS index (uses Azure embeddings via the provided client)
    index = index_context_db(contexts, client, embed_name)
    faiss.write_index(index, index_path)
    with open(ctx_path, "w", encoding="utf-8") as f:
        json.dump(contexts, f)

    # Build entity cache for hybrid BM25/fuzzy layer
    entity_cache = [db_extract_entities(r) for r in entity_rows]
    entity_path.parent.mkdir(parents=True, exist_ok=True)
    with open(entity_path, "w", encoding="utf-8") as f:
        json.dump(entity_cache, f)

    return used_version


def _parse_version_from_asset_name(filename: str) -> Optional[str]:
    # Expected shapes:
    #   <embed>_<db>_<name>__<version>.faiss
    #   <embed>_<db>_<name>__<version>.json
    if "__" not in filename:
        return None
    tail = filename.split("__", 1)[1]
    # strip extension
    if "." in tail:
        tail = tail.rsplit(".", 1)[0]
    return tail or None


def _find_latest_context_assets(embed_name: str, db: str, db_type: str) -> Optional[tuple[str, str, str]]:
    """Return (faiss_path, json_path, version) for newest matching assets, or None."""
    base_dir = Path("data/latest_db/indexes")
    name = f"{db_type}_context"
    pattern = f"{embed_name}_{db}_{name}__*.faiss"
    candidates = list(base_dir.glob(pattern))
    if not candidates:
        return None

    version_to_paths: dict[str, tuple[Path, Path]] = {}
    for faiss_path in candidates:
        version = _parse_version_from_asset_name(faiss_path.name)
        if not version:
            continue
        json_path = faiss_path.with_suffix(".json")
        if json_path.exists():
            version_to_paths[version] = (faiss_path, json_path)

    if not version_to_paths:
        return None

    latest_version = sorted(version_to_paths.keys())[-1]
    faiss_path, json_path = version_to_paths[latest_version]
    return str(faiss_path), str(json_path), latest_version


def _find_latest_entity_asset(db: str) -> Optional[tuple[str, str]]:
    """Return (entity_json_path, version) for newest matching entity cache, or None."""
    base_dir = Path("context_retriever/entities")
    if db in {"fda", "ema"}:
        pattern = f"moalmanac_{db}_ner_entities__*.json"
    elif db == "civic":
        path = base_dir / "civic_ner_entities__2025-10-01.json"
        return (str(path), "2025-10-01") if path.exists() else None
    else:
        return None

    candidates = list(base_dir.glob(pattern))
    if not candidates:
        return None

    versioned: list[tuple[str, Path]] = []
    for p in candidates:
        version = _parse_version_from_asset_name(p.name)
        if version:
            versioned.append((version, p))
    if not versioned:
        return None

    versioned.sort(key=lambda t: t[0])
    latest_version, latest_path = versioned[-1]
    return str(latest_path), latest_version


def _load_dotenv(env_file: str) -> dict[str, str]:
    """Minimal .env loader (no external dependency).

    Supports lines like KEY=VALUE and ignores blank lines/comments.
    """
    if not os.path.exists(env_file):
        raise FileNotFoundError(
            f"Missing {env_file}. Create it or pass --env_file. "
            "This CLI expects Azure OpenAI settings in a .env file."
        )

    values: dict[str, str] = {}
    with open(env_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].lstrip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                values[key] = value
    return values


def _get_env(values: dict[str, str], key: str) -> Optional[str]:
    return values.get(key) or os.environ.get(key)


def _azure_base_url(values: dict[str, str]) -> str:
    base_url = _get_env(values, "AZURE_OPENAI_BASE_URL")
    if base_url:
        normalized = base_url.strip()
        normalized = normalized.replace("/openai/v1/openai/v1/", "/openai/v1/")
        normalized = normalized.replace("/openai/v1/openai/v1", "/openai/v1")
        return normalized.rstrip("/") + "/"

    endpoint = _get_env(values, "AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "Azure base URL not configured. Set either AZURE_OPENAI_BASE_URL or AZURE_OPENAI_ENDPOINT in your .env"
        )
    endpoint = endpoint.strip()
    if "/openai/v1" in endpoint:
        return endpoint.rstrip("/") + "/"
    return endpoint.rstrip("/") + "/openai/v1/"


def _read_local_db_version(cache_file: str = "db_version_cache.json") -> str:
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Missing {cache_file}. This repo uses it to decide which local DB version to load. "
            "If you cloned a partial copy, re-add it or run the DB sync pipeline to generate it."
        )
    with open(cache_file, "r", encoding="utf-8") as f:
        payload = json.load(f)
    version = payload.get("version")
    if not version:
        raise ValueError(f"{cache_file} exists but has no 'version' field")
    return version


def _cache_paths(output_dir: str, embed_name: str, db: str, name: str, version: str) -> tuple[str, str]:
    # Mirror utils.context_db._cache_paths without importing utils.context_db
    os.makedirs(output_dir, exist_ok=True)
    return (
        f"{output_dir}/{embed_name}_{db}_{name}__{version}.faiss",
        f"{output_dir}/{embed_name}_{db}_{name}__{version}.json",
    )


def load_context_local(
    version: str,
    db: str,
    db_type: str,
    embed_name: str,
    *,
    allow_fallback_latest: bool = True,
) -> tuple[list[str], faiss.Index, str]:
    """Load context strings and FAISS index from disk.

    Expects assets under `data/latest_db/indexes/`.
    """
    if db in {"fda", "ema"}:
        v = version
    elif db == "civic":
        v = "2025-10-01"  # civic assets in this repo are pinned to this snapshot
    else:
        raise ValueError("context_db must be one of: fda, ema, civic")

    index_path, ctx_path = _cache_paths(
        output_dir="data/latest_db/indexes",
        embed_name=embed_name,
        db=db,
        name=f"{db_type}_context",
        version=v,
    )

    if not os.path.exists(index_path) or not os.path.exists(ctx_path):
        if allow_fallback_latest and db in {"fda", "ema"}:
            latest = _find_latest_context_assets(embed_name=embed_name, db=db, db_type=db_type)
            if latest is not None:
                index_path, ctx_path, v = latest
            else:
                missing = [p for p in (index_path, ctx_path) if not os.path.exists(p)]
                raise FileNotFoundError(
                    "Missing local retrieval assets:\n"
                    + "\n".join(f"- {p}" for p in missing)
                    + "\n\nNo matching assets found to fallback to. Ensure `data/latest_db/indexes/` contains "
                    f"{embed_name}_{db}_{db_type}_context__<version>.faiss/.json"
                )
        else:
            missing = [p for p in (index_path, ctx_path) if not os.path.exists(p)]
            raise FileNotFoundError(
                "Missing local retrieval assets:\n"
                + "\n".join(f"- {p}" for p in missing)
                + "\n\nThis CLI only loads prebuilt assets. Ensure `data/latest_db/indexes/` contains "
                "the requested context DB + index for your version."
            )

    with open(ctx_path, "r", encoding="utf-8") as f:
        context = json.load(f)
    index = faiss.read_index(index_path)
    return context, index, v


def load_db_entities_local(version: str, db: str, *, allow_fallback_latest: bool = True) -> tuple[list[dict[str, list[str]]], str]:
    base_path = Path("context_retriever/entities")
    if db in {"fda", "ema"}:
        path = base_path / f"moalmanac_{db}_ner_entities__{version}.json"
    elif db == "civic":
        path = base_path / "civic_ner_entities__2025-10-01.json"
    else:
        raise ValueError("context_db must be one of: fda, ema, civic")

    if not path.exists():
        if allow_fallback_latest and db in {"fda", "ema"}:
            latest = _find_latest_entity_asset(db)
            if latest is None:
                raise FileNotFoundError(f"Missing entity cache: {path}")
            latest_path, latest_version = latest
            path = Path(latest_path)
            version = latest_version
        else:
            raise FileNotFoundError(f"Missing entity cache: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f), version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAG-LLM with hybrid retrieval (CLI).")

    parser.add_argument(
        "--env_file",
        default=".env",
        help="Path to a .env file containing Azure OpenAI settings (default: .env)",
    )

    parser.add_argument(
        "--model",
        default=None,
        help="Chat deployment name (Azure). If omitted, reads AZURE_OPENAI_DEPLOYMENT from .env.",
    )
    parser.add_argument(
        "--embed_model",
        default=None,
        help=(
            "Embedding deployment name (Azure). If omitted, reads AZURE_OPENAI_EMBEDDING_DEPLOYMENT from .env. "
            f"Must match the FAISS index prefix on disk; fallback default is {DEFAULT_EMBED_MODEL}."
        ),
    )

    parser.add_argument("--context_db", choices=["fda", "ema", "civic"], required=True)
    parser.add_argument("--context_db_type", choices=["structured", "unstructured"], default="structured")

    parser.add_argument(
        "--db_version",
        default=None,
        help=(
            "Override the local DB version (e.g. 2025-09-04). "
            "If omitted, uses db_version_cache.json for fda/ema and will fallback to the newest available local assets if missing."
        ),
    )

    parser.add_argument(
        "--build_if_missing",
        action="store_true",
        help="If local FDA/EMA retrieval assets are missing, build them by calling the MOAlmanac API + Azure embeddings.",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Rebuild retrieval assets even if they already exist.",
    )

    parser.add_argument("--strategy", type=int, choices=list(range(0, 8)), default=5)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2025)

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Free-form user query. If omitted, reads from stdin.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print resolved Azure base_url + deployment names before running",
    )

    return parser.parse_args()


def _read_query(query: Optional[str]) -> str:
    if query is not None and query.strip():
        return query.strip()
    # stdin fallback
    import sys

    text = sys.stdin.read().strip()
    if not text:
        raise ValueError("No query provided. Pass --query or pipe text via stdin.")
    return text


def main() -> None:
    args = parse_args()

    user_query = _read_query(args.query)

    env_values = _load_dotenv(args.env_file)
    api_key = _get_env(env_values, "AZURE_OPENAI_API_KEY") or _get_env(env_values, "OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing AZURE_OPENAI_API_KEY in .env (or OPENAI_API_KEY as a fallback)")

    base_url = _azure_base_url(env_values)
    model = args.model or _get_env(env_values, "AZURE_OPENAI_DEPLOYMENT") or "gpt-4o"
    embed_model = (
        args.embed_model
        or _get_env(env_values, "AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        or DEFAULT_EMBED_MODEL
    )
    if args.debug:
        print(base_url, model, embed_model)
    client = OpenAI(api_key=api_key, base_url=base_url)

    # Resolve version for local assets
    local_version = _read_local_db_version()
    requested_version = args.db_version or local_version

    # Load local DB assets (with fallback-to-latest for fda/ema)
    try:
        context_chunks, index, resolved_version = load_context_local(
            version=requested_version,
            db=args.context_db,
            db_type=args.context_db_type,
            embed_name=embed_model,
        )
    except FileNotFoundError:
        if args.build_if_missing and args.context_db in {"fda", "ema"}:
            built_version = build_assets_if_missing(
                db=args.context_db,
                db_type=args.context_db_type,
                embed_name=embed_model,
                client=client,
                force_rebuild=args.force_rebuild,
                version=None,
            )
            context_chunks, index, resolved_version = load_context_local(
                version=built_version,
                db=args.context_db,
                db_type=args.context_db_type,
                embed_name=embed_model,
                allow_fallback_latest=False,
            )
        else:
            raise

    # Load DB entities + extract query entities (hybrid retrieval)
    db_entity, entity_version = load_db_entities_local(version=resolved_version, db=args.context_db)
    user_entities = extract_entities(user_query)

    retrieval = retrieve_context_hybrid(
        user_entities=user_entities,
        db_entities=db_entity,
        user_query=user_query,
        db_context=context_chunks,
        index=index,
        client=client,
        model_embed=embed_model,
        num_vec=DEFAULT_NUM_VEC,
    )

    retrieved_context_text = "\n".join(retrieval.top_contexts)

    query_prompt = get_prompt(args.strategy, user_query)
    input_prompt = (
        "Context information is below.\n"
        "---------------------\n"
        f"{retrieved_context_text}\n"
        "---------------------\n"
        f"{query_prompt}\n"
    )

    output, _ = run_llm(
        input_prompt=input_prompt,
        client=client,
        model_type="gpt",
        model=model,
        max_len=args.max_tokens,
        temp=args.temp,
        random_seed=args.seed,
    )

    if output is None:
        raise SystemExit(2)

    # Print pretty JSON if possible; otherwise print raw text.
    try:
        parsed = json.loads(output)
    except Exception:
        print(output)
        return

    print(json.dumps(parsed, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
