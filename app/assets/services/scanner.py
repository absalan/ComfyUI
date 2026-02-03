import contextlib
import logging
import os
import time

import folder_paths
from app.assets.database.bulk_ops import seed_from_paths_batch
from app.assets.database.queries import (
    add_missing_tag_for_asset_id,
    ensure_tags_exist,
    remove_missing_tag_for_asset_id,
    delete_cache_states_outside_prefixes,
    get_orphaned_seed_asset_ids,
    delete_assets_by_ids,
    get_cache_states_for_prefixes,
    bulk_set_needs_verify,
    delete_cache_states_by_ids,
    delete_orphaned_seed_asset,
)
from app.assets.helpers import (
    collect_models_files,
    compute_relative_filename,
    fast_asset_file_check,
    get_name_and_tags_from_asset_path,
    list_tree,
    prefixes_for_root,
    RootType,
)
from app.database.db import create_session, dependencies_available


def prune_orphaned_assets(session, valid_prefixes: list[str]) -> int:
    """Prune cache states outside valid prefixes, then delete orphaned seed assets.

    Args:
        session: Database session
        valid_prefixes: List of absolute directory prefixes that are valid

    Returns:
        Number of orphaned assets deleted
    """
    delete_cache_states_outside_prefixes(session, valid_prefixes)
    orphan_ids = get_orphaned_seed_asset_ids(session)
    return delete_assets_by_ids(session, orphan_ids)


def reconcile_cache_states_for_root(
    session,
    root: RootType,
    collect_existing_paths: bool = False,
    update_missing_tags: bool = False,
) -> set[str] | None:
    """Reconcile cache states with filesystem for a root.

    - Toggle needs_verify per state using fast mtime/size check
    - For hashed assets with at least one fast-ok state in this root: delete stale missing states
    - For seed assets with all states missing: delete Asset and its AssetInfos
    - Optionally add/remove 'missing' tags based on fast-ok in this root
    - Optionally return surviving absolute paths

    Args:
        session: Database session
        root: Root type to scan
        collect_existing_paths: If True, return set of surviving file paths
        update_missing_tags: If True, update 'missing' tags based on file status

    Returns:
        Set of surviving absolute paths if collect_existing_paths=True, else None
    """
    prefixes = prefixes_for_root(root)
    if not prefixes:
        return set() if collect_existing_paths else None

    rows = get_cache_states_for_prefixes(session, prefixes)

    by_asset: dict[str, dict] = {}
    for row in rows:
        acc = by_asset.get(row.asset_id)
        if acc is None:
            acc = {"hash": row.asset_hash, "size_db": row.size_bytes, "states": []}
            by_asset[row.asset_id] = acc

        fast_ok = False
        try:
            exists = True
            fast_ok = fast_asset_file_check(
                mtime_db=row.mtime_ns,
                size_db=acc["size_db"],
                stat_result=os.stat(row.file_path, follow_symlinks=True),
            )
        except FileNotFoundError:
            exists = False
        except OSError:
            exists = False

        acc["states"].append({
            "sid": row.state_id,
            "fp": row.file_path,
            "exists": exists,
            "fast_ok": fast_ok,
            "needs_verify": row.needs_verify,
        })

    to_set_verify: list[int] = []
    to_clear_verify: list[int] = []
    stale_state_ids: list[int] = []
    survivors: set[str] = set()

    for aid, acc in by_asset.items():
        a_hash = acc["hash"]
        states = acc["states"]
        any_fast_ok = any(s["fast_ok"] for s in states)
        all_missing = all(not s["exists"] for s in states)

        for s in states:
            if not s["exists"]:
                continue
            if s["fast_ok"] and s["needs_verify"]:
                to_clear_verify.append(s["sid"])
            if not s["fast_ok"] and not s["needs_verify"]:
                to_set_verify.append(s["sid"])

        if a_hash is None:
            if states and all_missing:
                delete_orphaned_seed_asset(session, aid)
            else:
                for s in states:
                    if s["exists"]:
                        survivors.add(os.path.abspath(s["fp"]))
            continue

        if any_fast_ok:
            for s in states:
                if not s["exists"]:
                    stale_state_ids.append(s["sid"])
            if update_missing_tags:
                with contextlib.suppress(Exception):
                    remove_missing_tag_for_asset_id(session, asset_id=aid)
        elif update_missing_tags:
            with contextlib.suppress(Exception):
                add_missing_tag_for_asset_id(session, asset_id=aid, origin="automatic")

        for s in states:
            if s["exists"]:
                survivors.add(os.path.abspath(s["fp"]))

    delete_cache_states_by_ids(session, stale_state_ids)
    bulk_set_needs_verify(session, to_set_verify, value=True)
    bulk_set_needs_verify(session, to_clear_verify, value=False)

    return survivors if collect_existing_paths else None


def seed_assets(roots: tuple[RootType, ...], enable_logging: bool = False) -> None:
    """Scan the given roots and seed the assets into the database."""
    if not dependencies_available():
        if enable_logging:
            logging.warning("Database dependencies not available, skipping assets scan")
        return

    t_start = time.perf_counter()
    created = 0
    skipped_existing = 0
    orphans_pruned = 0
    paths: list[str] = []

    try:
        existing_paths: set[str] = set()
        for r in roots:
            try:
                with create_session() as sess:
                    survivors = reconcile_cache_states_for_root(
                        sess,
                        r,
                        collect_existing_paths=True,
                        update_missing_tags=True,
                    )
                    sess.commit()
                if survivors:
                    existing_paths.update(survivors)
            except Exception as e:
                logging.exception("fast DB scan failed for %s: %s", r, e)

        try:
            with create_session() as sess:
                all_prefixes = [
                    os.path.abspath(p) for r in roots for p in prefixes_for_root(r)
                ]
                orphans_pruned = prune_orphaned_assets(sess, all_prefixes)
                sess.commit()
        except Exception as e:
            logging.exception("orphan pruning failed: %s", e)

        if "models" in roots:
            paths.extend(collect_models_files())
        if "input" in roots:
            paths.extend(list_tree(folder_paths.get_input_directory()))
        if "output" in roots:
            paths.extend(list_tree(folder_paths.get_output_directory()))

        specs: list[dict] = []
        tag_pool: set[str] = set()
        for p in paths:
            abs_p = os.path.abspath(p)
            if abs_p in existing_paths:
                skipped_existing += 1
                continue
            try:
                stat_p = os.stat(abs_p, follow_symlinks=False)
            except OSError:
                continue
            if not stat_p.st_size:
                continue
            name, tags = get_name_and_tags_from_asset_path(abs_p)
            specs.append({
                "abs_path": abs_p,
                "size_bytes": stat_p.st_size,
                "mtime_ns": getattr(stat_p, "st_mtime_ns", int(stat_p.st_mtime * 1_000_000_000)),
                "info_name": name,
                "tags": tags,
                "fname": compute_relative_filename(abs_p),
            })
            for t in tags:
                tag_pool.add(t)

        if not specs:
            return

        with create_session() as sess:
            if tag_pool:
                ensure_tags_exist(sess, tag_pool, tag_type="user")
            result = seed_from_paths_batch(sess, specs=specs, owner_id="")
            created += result["inserted_infos"]
            sess.commit()

    finally:
        if enable_logging:
            logging.info(
                "Assets scan(roots=%s) completed in %.3fs (created=%d, skipped_existing=%d, orphans_pruned=%d, total_seen=%d)",
                roots,
                time.perf_counter() - t_start,
                created,
                skipped_existing,
                orphans_pruned,
                len(paths),
            )
