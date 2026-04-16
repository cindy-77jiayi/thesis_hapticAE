"""Gradio UI for the 3-image OpenAI -> semantic -> PC -> local VAE flow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from src.llm import DEFAULT_OPENAI_MODEL
from src.pipelines.semantic_reconstruction import create_session_run_dir, run_three_image_reconstruction


PipelineRunner = Callable[..., dict[str, Any]]


def run_three_image_generation(
    *,
    image_paths: list[str | None],
    notes: str,
    frozen_manifest: str,
    model: str,
    outputs_dir: str,
    pipeline_runner: PipelineRunner = run_three_image_reconstruction,
) -> dict[str, Any]:
    """Validate inputs and run one 3-image reconstruction session."""
    cleaned_paths = [str(path) for path in image_paths if path not in (None, "")]
    if len(cleaned_paths) != 3:
        raise gr.Error("Please upload exactly 3 images before generating.")
    if not frozen_manifest:
        raise gr.Error("Please provide a frozen manifest path before generating.")

    run_dir = create_session_run_dir(outputs_dir)
    return pipeline_runner(
        image_paths=cleaned_paths,
        notes=notes or "",
        run_dir=run_dir,
        frozen_manifest_path=frozen_manifest,
        model=model,
    )


def build_demo(
    *,
    frozen_manifest: str,
    model: str = DEFAULT_OPENAI_MODEL,
    outputs_dir: str = "outputs/vae_llm_ui",
    pipeline_runner: PipelineRunner = run_three_image_reconstruction,
) -> gr.Blocks:
    """Build the local end-to-end validation UI."""
    outputs_root = str(Path(outputs_dir).resolve())

    def _generate(
        image_1: str | None,
        image_2: str | None,
        image_3: str | None,
        notes: str,
    ) -> tuple[str, str, str, str, str, str, str, str]:
        result = run_three_image_generation(
            image_paths=[image_1, image_2, image_3],
            notes=notes,
            frozen_manifest=frozen_manifest,
            model=model,
            outputs_dir=outputs_dir,
            pipeline_runner=pipeline_runner,
        )
        segment = result["segments"][0]
        return (
            json.dumps(segment["semantic_controls"], indent=2, ensure_ascii=False),
            json.dumps(segment["rationale"], indent=2, ensure_ascii=False),
            json.dumps(segment["pc_vector"], indent=2, ensure_ascii=False),
            result["generated_wav"],
            result["waveform_image"],
            result["generated_wav"],
            result["waveform_image"],
            result["run_dir"],
        )

    frozen_manifest_label = str(Path(frozen_manifest).resolve()) if frozen_manifest else "(missing)"

    with gr.Blocks(title="3-Image Semantic to VAE Reconstruction") as demo:
        gr.Markdown(
            f"""
            # 3-Image Semantic to VAE Reconstruction
            Upload 3 ordered images and run the full local pipeline:
            OpenAI semantic controls -> PC vector -> local VAE waveform reconstruction.

            Frozen manifest: `{frozen_manifest_label}`
            Outputs root: `{outputs_root}`
            """
        )

        with gr.Row():
            image_1 = gr.Image(label="image_1", type="filepath")
            image_2 = gr.Image(label="image_2", type="filepath")
            image_3 = gr.Image(label="image_3", type="filepath")
        notes = gr.Textbox(label="optional notes", lines=3, placeholder="Optional timing or tactile notes")
        generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            semantic_controls = gr.Code(label="semantic_controls", language="json")
            rationale = gr.Code(label="rationale", language="json")
        pc_vector = gr.Code(label="pc_vector", language="json")

        with gr.Row():
            audio_output = gr.Audio(label="generated wav", type="filepath")
            waveform_output = gr.Image(label="waveform image", type="filepath")
        with gr.Row():
            generated_wav_path = gr.Textbox(label="generated wav path", interactive=False)
            waveform_image_path = gr.Textbox(label="waveform image path", interactive=False)
        run_dir_state = gr.Textbox(label="run directory", interactive=False)

        generate_btn.click(
            _generate,
            inputs=[image_1, image_2, image_3, notes],
            outputs=[
                semantic_controls,
                rationale,
                pc_vector,
                audio_output,
                waveform_output,
                generated_wav_path,
                waveform_image_path,
                run_dir_state,
            ],
        )

    return demo
