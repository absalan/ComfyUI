from __future__ import annotations

from typing import Any


class NodeReplace:
    """
    Defines a possible node replacement, mapping inputs and outputs of the old node to the new node.

    Also supports assigning specific values to the input widgets of the new node.
    """
    def __init__(self,
        new_node_id: str,
        old_node_id: str,
        old_widget_ids: list[str] | None=None,
        input_mapping: list[InputMap] | None=None,
        output_mapping: list[OutputMap] | None=None,
    ):
        self.new_node_id = new_node_id
        self.old_node_id = old_node_id
        self.old_widget_ids = old_widget_ids
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping

    def as_dict(self):
        """Create serializable representation of the node replacement."""
        return {
            "new_node_id": self.new_node_id,
            "old_node_id": self.old_node_id,
            "old_widget_ids": self.old_widget_ids,
            "input_mapping": [m.as_dict() for m in self.input_mapping] if self.input_mapping else None,
            "output_mapping": [m.as_dict() for m in self.output_mapping] if self.output_mapping else None,
        }


class InputMap:
    """
    Map inputs of node replacement.

    Use InputMap.OldId or InputMap.SetValue for mapping purposes.
    """
    class _Assign:
        def __init__(self, assign_type: str):
            self.assign_type = assign_type

        def as_dict(self):
            return {
                "assign_type": self.assign_type,
            }

    class OldId(_Assign):
        """Connect the input of the old node with given id to new node when replacing."""
        def __init__(self, old_id: str):
            super().__init__("old_id")
            self.old_id = old_id

        def as_dict(self):
            return super().as_dict() | {
                "old_id": self.old_id,
            }

    class SetValue(_Assign):
        """Use the given value for the input of the new node when replacing; assumes input is a widget."""
        def __init__(self, value: Any):
            super().__init__("set_value")
            self.value = value

        def as_dict(self):
            return super().as_dict() | {
                "value": self.value,
            }

    def __init__(self, new_id: str, assign: OldId | SetValue):
        self.new_id = new_id
        self.assign = assign

    def as_dict(self):
        return {
            "new_id": self.new_id,
            "assign": self.assign.as_dict(),
        }


class OutputMap:
    """Map outputs of node replacement via indexes, as that's how outputs are stored."""
    def __init__(self, new_idx: int, old_idx: int):
        self.new_idx = new_idx
        self.old_idx = old_idx

    def as_dict(self):
        return {
            "new_idx": self.new_idx,
            "old_idx": self.old_idx,
        }
