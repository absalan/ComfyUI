import os
import re
import logging
from contextlib import contextmanager
from typing import TypedDict, Generator

import numpy as np
import torch

import nodes
from comfy_api.latest import ComfyExtension, io, ui
from comfy.cli_args import args
from typing_extensions import override
from utils.install_util import get_missing_requirements_message


class SizeModeInput(TypedDict):
    size_mode: str
    width: int
    height: int


MAX_IMAGES = 5      # u_image0-4
MAX_UNIFORMS = 5    # u_float0-4, u_int0-4
MAX_OUTPUTS = 4     # fragColor0-3 (MRT)

logger = logging.getLogger(__name__)

try:
    import moderngl
except ImportError as e:
    raise RuntimeError(f"ModernGL is not available.\n{get_missing_requirements_message()}") from e

# Default NOOP fragment shader that passes through the input image unchanged
# For multiple outputs, use: layout(location = 0) out vec4 fragColor0; etc.
DEFAULT_FRAGMENT_SHADER = """#version 300 es
precision highp float;

uniform sampler2D u_image0;
uniform vec2 u_resolution;

in vec2 v_texCoord;
layout(location = 0) out vec4 fragColor0;

void main() {
    fragColor0 = texture(u_image0, v_texCoord);
}
"""


# Simple vertex shader for full-screen quad
VERTEX_SHADER = """#version 330

in vec2 in_position;
in vec2 in_texcoord;

out vec2 v_texCoord;

void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    v_texCoord = in_texcoord;
}
"""


def _convert_es_to_desktop_glsl(source: str) -> str:
    """Convert GLSL ES 3.00 shader to desktop GLSL 3.30 for ModernGL compatibility."""
    return re.sub(r'#version\s+300\s+es', '#version 330', source)


def _create_software_gl_context() -> moderngl.Context:
    original_env = os.environ.get("LIBGL_ALWAYS_SOFTWARE")
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    try:
        ctx = moderngl.create_standalone_context(require=330)
        logger.info(f"Created software-rendered OpenGL context: {ctx.info['GL_RENDERER']}")
        return ctx
    finally:
        if original_env is None:
            os.environ.pop("LIBGL_ALWAYS_SOFTWARE", None)
        else:
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = original_env


def _create_gl_context(force_software: bool = False) -> moderngl.Context:
    if force_software:
        try:
            return _create_software_gl_context()
        except Exception as e:
            raise RuntimeError(
                "Failed to create software-rendered OpenGL context.\n"
                "Ensure Mesa/llvmpipe is installed for software rendering support."
            ) from e

    # Try hardware rendering first, fall back to software
    try:
        ctx = moderngl.create_standalone_context(require=330)
        logger.info(f"Created OpenGL context: {ctx.info['GL_RENDERER']}")
        return ctx
    except Exception as hw_error:
        logger.warning(f"Hardware OpenGL context creation failed: {hw_error}")
        logger.info("Attempting software rendering fallback...")
        try:
            return _create_software_gl_context()
        except Exception as sw_error:
            raise RuntimeError(
                f"Failed to create OpenGL context.\n"
                f"Hardware error: {hw_error}\n\n"
                f"Possible solutions:\n"
                f"1. Install GPU drivers with OpenGL 3.3+ support\n"
                f"2. Install Mesa for software rendering (Linux: apt install libgl1-mesa-dri)\n"
                f"3. On headless servers, ensure virtual framebuffer (Xvfb) or EGL is available"
            ) from sw_error


def _image_to_texture(ctx: moderngl.Context, image: np.ndarray) -> moderngl.Texture:
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1

    components = min(channels, 4)

    image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)

    # Flip vertically for OpenGL coordinate system (origin at bottom-left)
    image_uint8 = np.ascontiguousarray(np.flipud(image_uint8))

    texture = ctx.texture((width, height), components, image_uint8.tobytes())
    texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
    texture.repeat_x = False
    texture.repeat_y = False

    return texture


def _texture_to_image(fbo: moderngl.Framebuffer, attachment: int = 0, channels: int = 4) -> np.ndarray:
    width, height = fbo.size

    data = fbo.read(components=channels, attachment=attachment)
    image = np.frombuffer(data, dtype=np.uint8).reshape((height, width, channels))

    image = np.ascontiguousarray(np.flipud(image))

    return image.astype(np.float32) / 255.0


def _compile_shader(ctx: moderngl.Context, fragment_source: str) -> moderngl.Program:
    # Convert user's GLSL ES 3.00 fragment shader to desktop GLSL 3.30 for ModernGL
    fragment_source = _convert_es_to_desktop_glsl(fragment_source)

    try:
        program = ctx.program(
            vertex_shader=VERTEX_SHADER,
            fragment_shader=fragment_source,
        )
        return program
    except Exception as e:
        raise RuntimeError(
            "Fragment shader compilation failed.\n\n"
            "Make sure your shader:\n"
            "1. Uses #version 300 es (WebGL 2.0 compatible)\n"
            "2. Has valid GLSL ES 3.00 syntax\n"
            "3. Includes 'precision highp float;' after version\n"
            "4. Uses 'out vec4 fragColor' instead of gl_FragColor\n"
            "5. Declares uniforms correctly (e.g., uniform sampler2D u_image0;)"
        ) from e


def _render_shader(
    ctx: moderngl.Context,
    program: moderngl.Program,
    width: int,
    height: int,
    textures: list[moderngl.Texture],
    uniforms: dict[str, int | float],
) -> list[np.ndarray]:
    # Create output textures
    output_textures = []
    for _ in range(MAX_OUTPUTS):
        tex = ctx.texture((width, height), 4)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        output_textures.append(tex)

    fbo = ctx.framebuffer(color_attachments=output_textures)

    # Full-screen quad vertices (position + texcoord)
    vertices = np.array([
        # Position (x, y), Texcoord (u, v)
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,
        -1.0, 1.0, 0.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())
    vao = ctx.vertex_array(
        program,
        [(vbo, '2f 2f', 'in_position', 'in_texcoord')],
    )

    try:
        # Bind textures
        for i, texture in enumerate(textures):
            texture.use(i)
            uniform_name = f'u_image{i}'
            if uniform_name in program:
                program[uniform_name].value = i

        # Set uniforms
        if 'u_resolution' in program:
            program['u_resolution'].value = (float(width), float(height))

        for name, value in uniforms.items():
            if name in program:
                program[name].value = value

        # Render
        fbo.use()
        fbo.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(moderngl.TRIANGLE_STRIP)

        # Read results from all attachments
        results = []
        for i in range(MAX_OUTPUTS):
            results.append(_texture_to_image(fbo, attachment=i, channels=4))
        return results
    finally:
        vao.release()
        vbo.release()
        for tex in output_textures:
            tex.release()
        fbo.release()


def _prepare_textures(
    ctx: moderngl.Context,
    image_list: list[torch.Tensor],
    batch_idx: int,
) -> list[moderngl.Texture]:
    textures = []
    for img_tensor in image_list[:MAX_IMAGES]:
        img_idx = min(batch_idx, img_tensor.shape[0] - 1)
        img_np = img_tensor[img_idx].cpu().numpy()
        textures.append(_image_to_texture(ctx, img_np))
    return textures


def _prepare_uniforms(int_list: list[int], float_list: list[float]) -> dict[str, int | float]:
    uniforms: dict[str, int | float] = {}
    for i, val in enumerate(int_list[:MAX_UNIFORMS]):
        uniforms[f'u_int{i}'] = int(val)
    for i, val in enumerate(float_list[:MAX_UNIFORMS]):
        uniforms[f'u_float{i}'] = float(val)
    return uniforms


def _release_textures(textures: list[moderngl.Texture]) -> None:
    for texture in textures:
        texture.release()


@contextmanager
def _gl_context(force_software: bool = False) -> Generator[moderngl.Context, None, None]:
    ctx = _create_gl_context(force_software)
    try:
        yield ctx
    finally:
        ctx.release()


@contextmanager
def _shader_program(ctx: moderngl.Context, fragment_source: str) -> Generator[moderngl.Program, None, None]:
    program = _compile_shader(ctx, fragment_source)
    try:
        yield program
    finally:
        program.release()


@contextmanager
def _textures_context(
    ctx: moderngl.Context,
    image_list: list[torch.Tensor],
    batch_idx: int,
) -> Generator[list[moderngl.Texture], None, None]:
    textures = _prepare_textures(ctx, image_list, batch_idx)
    try:
        yield textures
    finally:
        _release_textures(textures)


class GLSLShader(io.ComfyNode):

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Create autogrow templates
        image_template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image"),
            prefix="image",
            min=1,
            max=MAX_IMAGES,
        )

        float_template = io.Autogrow.TemplatePrefix(
            io.Float.Input("float", default=0.0),
            prefix="u_float",
            min=0,
            max=MAX_UNIFORMS,
        )

        int_template = io.Autogrow.TemplatePrefix(
            io.Int.Input("int", default=0),
            prefix="u_int",
            min=0,
            max=MAX_UNIFORMS,
        )

        return io.Schema(
            node_id="GLSLShader",
            display_name="GLSL Shader",
            category="image/shader",
            description=(
                f"Apply GLSL fragment shaders to images. "
                f"Inputs: u_image0-{MAX_IMAGES-1} (sampler2D), u_resolution (vec2), "
                f"u_float0-{MAX_UNIFORMS-1}, u_int0-{MAX_UNIFORMS-1}. "
                f"Outputs: layout(location = 0-{MAX_OUTPUTS-1}) out vec4 fragColor0-{MAX_OUTPUTS-1}."
            ),
            inputs=[
                io.String.Input(
                    "fragment_shader",
                    default=DEFAULT_FRAGMENT_SHADER,
                    multiline=True,
                    tooltip="GLSL fragment shader source code (GLSL ES 3.00 / WebGL 2.0 compatible)",
                ),
                io.DynamicCombo.Input(
                    "size_mode",
                    options=[
                        io.DynamicCombo.Option(
                            "from_input",
                            [],  # No extra inputs - uses first input image dimensions
                        ),
                        io.DynamicCombo.Option(
                            "custom",
                            [
                                io.Int.Input("width", default=512, min=1, max=nodes.MAX_RESOLUTION),
                                io.Int.Input("height", default=512, min=1, max=nodes.MAX_RESOLUTION),
                            ],
                        ),
                    ],
                    tooltip="Output size: 'from_input' uses first input image dimensions, 'custom' allows manual size",
                ),
                io.Autogrow.Input("images", template=image_template),
                io.Autogrow.Input("floats", template=float_template),
                io.Autogrow.Input("ints", template=int_template),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE0"),
                io.Image.Output(display_name="IMAGE1"),
                io.Image.Output(display_name="IMAGE2"),
                io.Image.Output(display_name="IMAGE3"),
            ],
        )

    @classmethod
    def execute(
        cls,
        fragment_shader: str,
        size_mode: SizeModeInput,
        images: io.Autogrow.Type,
        floats: io.Autogrow.Type = None,
        ints: io.Autogrow.Type = None,
        **kwargs,
    ) -> io.NodeOutput:
        image_list = [v for v in images.values() if v is not None]
        float_list = [v if v is not None else 0.0 for v in floats.values()] if floats else []
        int_list = [v if v is not None else 0 for v in ints.values()] if ints else []

        if not image_list:
            raise ValueError("At least one input image is required")

        # Determine output dimensions
        if size_mode["size_mode"] == "custom":
            out_width, out_height = size_mode["width"], size_mode["height"]
        else:
            out_height, out_width = image_list[0].shape[1], image_list[0].shape[2]

        batch_size = image_list[0].shape[0]
        uniforms = _prepare_uniforms(int_list, float_list)

        with _gl_context(force_software=args.cpu) as ctx:
            with _shader_program(ctx, fragment_shader) as program:
                # Collect outputs for each render target across all batches
                all_outputs: list[list[torch.Tensor]] = [[] for _ in range(MAX_OUTPUTS)]

                for b in range(batch_size):
                    with _textures_context(ctx, image_list, b) as textures:
                        results = _render_shader(ctx, program, out_width, out_height, textures, uniforms)
                        for i, result in enumerate(results):
                            all_outputs[i].append(torch.from_numpy(result))

                # Stack batches for each output
                output_values = []
                for i in range(MAX_OUTPUTS):
                    output_batch = torch.stack(all_outputs[i], dim=0)
                    output_values.append(output_batch)

                return io.NodeOutput(*output_values, ui=cls._build_ui_output(image_list, output_values[0]))

    @classmethod
    def _build_ui_output(cls, image_list: list[torch.Tensor], output_batch: torch.Tensor) -> dict[str, list]:
        """Build UI output with input and output images for client-side shader execution."""
        combined_inputs = torch.cat(image_list, dim=0)
        input_images_ui = ui.ImageSaveHelper.save_images(
            combined_inputs,
            filename_prefix="GLSLShader_input",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        output_images_ui = ui.ImageSaveHelper.save_images(
            output_batch,
            filename_prefix="GLSLShader_output",
            folder_type=io.FolderType.temp,
            cls=None,
            compress_level=1,
        )

        return {"input_images": input_images_ui, "images": output_images_ui}


class GLSLExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [GLSLShader]


async def comfy_entrypoint() -> GLSLExtension:
    return GLSLExtension()
