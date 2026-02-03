# Re-export public API from query modules
# Maintains backward compatibility with old flat queries.py imports

from app.assets.database.queries.asset import (
    asset_exists_by_hash,
    get_asset_by_hash,
)

from app.assets.database.queries.asset_info import (
    asset_info_exists_for_asset_id,
    get_asset_info_by_id,
    list_asset_infos_page,
    fetch_asset_info_asset_and_tags,
    fetch_asset_info_and_asset,
    touch_asset_info_by_id,
    create_asset_info_for_existing_asset,
    replace_asset_info_metadata_projection,
    ingest_fs_asset,
    update_asset_info_full,
    delete_asset_info_by_id,
    set_asset_info_preview,
)

from app.assets.database.queries.cache_state import (
    list_cache_states_by_asset_id,
    pick_best_live_path,
    prune_orphaned_assets,
    fast_db_consistency_pass,
)

from app.assets.database.queries.tags import (
    ensure_tags_exist,
    get_asset_tags,
    set_asset_info_tags,
    add_tags_to_asset_info,
    remove_tags_from_asset_info,
    add_missing_tag_for_asset_id,
    remove_missing_tag_for_asset_id,
    list_tags_with_usage,
)

__all__ = [
    # asset.py
    "asset_exists_by_hash",
    "get_asset_by_hash",
    # asset_info.py
    "asset_info_exists_for_asset_id",
    "get_asset_info_by_id",
    "list_asset_infos_page",
    "fetch_asset_info_asset_and_tags",
    "fetch_asset_info_and_asset",
    "touch_asset_info_by_id",
    "create_asset_info_for_existing_asset",
    "replace_asset_info_metadata_projection",
    "ingest_fs_asset",
    "update_asset_info_full",
    "delete_asset_info_by_id",
    "set_asset_info_preview",
    # cache_state.py
    "list_cache_states_by_asset_id",
    "pick_best_live_path",
    "prune_orphaned_assets",
    "fast_db_consistency_pass",
    # tags.py
    "ensure_tags_exist",
    "get_asset_tags",
    "set_asset_info_tags",
    "add_tags_to_asset_info",
    "remove_tags_from_asset_info",
    "add_missing_tag_for_asset_id",
    "remove_missing_tag_for_asset_id",
    "list_tags_with_usage",
]
