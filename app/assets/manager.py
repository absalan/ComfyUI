import os
import mimetypes
from typing import Sequence

from app.database.db import create_session
from app.assets.api import schemas_out
from app.assets.database.queries import (
    asset_exists_by_hash,
    fetch_asset_info_asset_and_tags,
    fetch_asset_info_and_asset,
    touch_asset_info_by_id,
    list_cache_states_by_asset_id,
    list_asset_infos_page,
    list_tags_with_usage,
    pick_best_live_path,
)


def _safe_sort_field(requested: str | None) -> str:
    if not requested:
        return "created_at"
    v = requested.lower()
    if v in {"name", "created_at", "updated_at", "size", "last_access_time"}:
        return v
    return "created_at"


def asset_exists(asset_hash: str) -> bool:
    with create_session() as session:
        return asset_exists_by_hash(session, asset_hash=asset_hash)

def list_assets(
    *,
    include_tags: Sequence[str] | None = None,
    exclude_tags: Sequence[str] | None = None,
    name_contains: str | None = None,
    metadata_filter: dict | None = None,
    limit: int = 20,
    offset: int = 0,
    sort: str = "created_at",
    order: str = "desc",
    owner_id: str = "",
) -> schemas_out.AssetsList:
    sort = _safe_sort_field(sort)
    order = "desc" if (order or "desc").lower() not in {"asc", "desc"} else order.lower()

    with create_session() as session:
        infos, tag_map, total = list_asset_infos_page(
            session,
            owner_id=owner_id,
            include_tags=include_tags,
            exclude_tags=exclude_tags,
            name_contains=name_contains,
            metadata_filter=metadata_filter,
            limit=limit,
            offset=offset,
            sort=sort,
            order=order,
        )

    summaries: list[schemas_out.AssetSummary] = []
    for info in infos:
        asset = info.asset
        tags = tag_map.get(info.id, [])
        summaries.append(
            schemas_out.AssetSummary(
                id=info.id,
                name=info.name,
                asset_hash=asset.hash if asset else None,
                size=int(asset.size_bytes) if asset else None,
                mime_type=asset.mime_type if asset else None,
                tags=tags,
                preview_url=f"/api/assets/{info.id}/content",
                created_at=info.created_at,
                updated_at=info.updated_at,
                last_access_time=info.last_access_time,
            )
        )

    return schemas_out.AssetsList(
        assets=summaries,
        total=total,
        has_more=(offset + len(summaries)) < total,
    )

def get_asset(
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> schemas_out.AssetDetail:
    with create_session() as session:
        res = fetch_asset_info_asset_and_tags(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not res:
            raise ValueError(f"AssetInfo {asset_info_id} not found")
        info, asset, tag_names = res
        preview_id = info.preview_id

    return schemas_out.AssetDetail(
        id=info.id,
        name=info.name,
        asset_hash=asset.hash if asset else None,
        size=int(asset.size_bytes) if asset and asset.size_bytes is not None else None,
        mime_type=asset.mime_type if asset else None,
        tags=tag_names,
        user_metadata=info.user_metadata or {},
        preview_id=preview_id,
        created_at=info.created_at,
        last_access_time=info.last_access_time,
    )

def resolve_asset_content_for_download(
    *,
    asset_info_id: str,
    owner_id: str = "",
) -> tuple[str, str, str]:
    with create_session() as session:
        pair = fetch_asset_info_and_asset(session, asset_info_id=asset_info_id, owner_id=owner_id)
        if not pair:
            raise ValueError(f"AssetInfo {asset_info_id} not found")

        info, asset = pair
        states = list_cache_states_by_asset_id(session, asset_id=asset.id)
        abs_path = pick_best_live_path(states)
        if not abs_path:
            raise FileNotFoundError

        touch_asset_info_by_id(session, asset_info_id=asset_info_id)
        session.commit()

    ctype = asset.mime_type or mimetypes.guess_type(info.name or abs_path)[0] or "application/octet-stream"
    download_name = info.name or os.path.basename(abs_path)
    return abs_path, ctype, download_name

def list_tags(
    prefix: str | None = None,
    limit: int = 100,
    offset: int = 0,
    order: str = "count_desc",
    include_zero: bool = True,
    owner_id: str = "",
) -> schemas_out.TagsList:
    limit = max(1, min(1000, limit))
    offset = max(0, offset)

    with create_session() as session:
        rows, total = list_tags_with_usage(
            session,
            prefix=prefix,
            limit=limit,
            offset=offset,
            include_zero=include_zero,
            order=order,
            owner_id=owner_id,
        )

    tags = [schemas_out.TagUsage(name=name, count=count, type=tag_type) for (name, tag_type, count) in rows]
    return schemas_out.TagsList(tags=tags, total=total, has_more=(offset + len(tags)) < total)
