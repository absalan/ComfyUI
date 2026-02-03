# Re-export public API from query modules
# Pure atomic database queries only - no business logic or orchestration

from app.assets.database.queries.asset import (
    asset_exists_by_hash,
    get_asset_by_hash,
    upsert_asset,
)

from app.assets.database.queries.asset_info import (
    asset_info_exists_for_asset_id,
    get_asset_info_by_id,
    insert_asset_info,
    get_or_create_asset_info,
    update_asset_info_timestamps,
    list_asset_infos_page,
    fetch_asset_info_asset_and_tags,
    fetch_asset_info_and_asset,
    touch_asset_info_by_id,
    replace_asset_info_metadata_projection,
    delete_asset_info_by_id,
    set_asset_info_preview,
)

from app.assets.database.queries.cache_state import (
    CacheStateRow,
    list_cache_states_by_asset_id,
    upsert_cache_state,
    delete_cache_states_outside_prefixes,
    get_orphaned_seed_asset_ids,
    delete_assets_by_ids,
    get_cache_states_for_prefixes,
    bulk_set_needs_verify,
    delete_cache_states_by_ids,
    delete_orphaned_seed_asset,
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
    "upsert_asset",
    # asset_info.py
    "asset_info_exists_for_asset_id",
    "get_asset_info_by_id",
    "insert_asset_info",
    "get_or_create_asset_info",
    "update_asset_info_timestamps",
    "list_asset_infos_page",
    "fetch_asset_info_asset_and_tags",
    "fetch_asset_info_and_asset",
    "touch_asset_info_by_id",
    "replace_asset_info_metadata_projection",
    "delete_asset_info_by_id",
    "set_asset_info_preview",
    # cache_state.py
    "CacheStateRow",
    "list_cache_states_by_asset_id",
    "upsert_cache_state",
    "delete_cache_states_outside_prefixes",
    "get_orphaned_seed_asset_ids",
    "delete_assets_by_ids",
    "get_cache_states_for_prefixes",
    "bulk_set_needs_verify",
    "delete_cache_states_by_ids",
    "delete_orphaned_seed_asset",
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
