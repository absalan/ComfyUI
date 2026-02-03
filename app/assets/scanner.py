import time
import logging
import os

import folder_paths
from app.database.db import create_session, dependencies_available
from app.assets.helpers import (
    collect_models_files, compute_relative_filename, fast_asset_file_check, get_name_and_tags_from_asset_path,
    list_tree, prefixes_for_root, escape_like_prefix,
    RootType
)
from app.assets.database.queries import (
    add_missing_tag_for_asset_id,
    ensure_tags_exist,
    remove_missing_tag_for_asset_id,
    prune_orphaned_assets,
    fast_db_consistency_pass,
)
from app.assets.database.bulk_ops import seed_from_paths_batch


def seed_assets(roots: tuple[RootType, ...], enable_logging: bool = False) -> None:
    """
    Scan the given roots and seed the assets into the database.
    """
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
                    survivors: set[str] = fast_db_consistency_pass(
                        sess,
                        r,
                        prefixes_for_root_fn=prefixes_for_root,
                        escape_like_prefix_fn=escape_like_prefix,
                        fast_asset_file_check_fn=fast_asset_file_check,
                        add_missing_tag_fn=add_missing_tag_for_asset_id,
                        remove_missing_tag_fn=remove_missing_tag_for_asset_id,
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
                orphans_pruned = prune_orphaned_assets(sess, roots, prefixes_for_root)
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
            # skip empty files
            if not stat_p.st_size:
                continue
            name, tags = get_name_and_tags_from_asset_path(abs_p)
            specs.append(
                {
                    "abs_path": abs_p,
                    "size_bytes": stat_p.st_size,
                    "mtime_ns": getattr(stat_p, "st_mtime_ns", int(stat_p.st_mtime * 1_000_000_000)),
                    "info_name": name,
                    "tags": tags,
                    "fname": compute_relative_filename(abs_p),
                }
            )
            for t in tags:
                tag_pool.add(t)
        # if no file specs, nothing to do
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



