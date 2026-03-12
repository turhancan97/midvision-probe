from __future__ import annotations

import gradio_client.utils as gr_client_utils


def patch_gradio_schema_parser() -> None:
    """Handle boolean JSON schema nodes emitted by newer pydantic versions."""
    if getattr(gr_client_utils, "_sparrta_bool_schema_patch", False):
        return

    original_get_type = gr_client_utils.get_type
    original_json_schema_to_python_type = gr_client_utils._json_schema_to_python_type

    def patched_get_type(schema):
        if isinstance(schema, bool):
            return {}
        return original_get_type(schema)

    def patched_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            return "Any"
        return original_json_schema_to_python_type(schema, defs)

    gr_client_utils.get_type = patched_get_type
    gr_client_utils._json_schema_to_python_type = patched_json_schema_to_python_type
    gr_client_utils._sparrta_bool_schema_patch = True

