from __future__ import annotations

from aiohttp import web

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from comfy_api.latest._node_replace import NodeReplace


class NodeReplaceManager:
    """Manages node replacement registrations."""

    def __init__(self):
        self._replacements: dict[str, list[NodeReplace]] = {}

    def register(self, node_replace: NodeReplace):
        """Register a node replacement mapping."""
        self._replacements.setdefault(node_replace.old_node_id, []).append(node_replace)

    def get_replacement(self, old_node_id: str) -> list[NodeReplace] | None:
        """Get replacements for an old node ID."""
        return self._replacements.get(old_node_id)

    def has_replacement(self, old_node_id: str) -> bool:
        """Check if a replacement exists for an old node ID."""
        return old_node_id in self._replacements

    def as_dict(self):
        """Serialize all replacements to dict."""
        return {
            k: [v.as_dict() for v in v_list]
            for k, v_list in self._replacements.items()
        }

    def add_routes(self, routes):
        @routes.get("/node_replacements")
        async def get_node_replacements(request):
            return web.json_response(self.as_dict())
