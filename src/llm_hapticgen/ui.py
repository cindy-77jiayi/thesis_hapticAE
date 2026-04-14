"""Gradio UI for the 3-image OpenAI -> HapticGen validation flow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gradio as gr

from .artifacts import create_session_run_dir, import_colab_outputs, save_generation_artifacts
from .openai_hapticgen import DEFAULT_HAPTICGEN_OPENAI_MODEL, OpenAIHapticGenPromptClient


def build_demo(
    *,
    model: str = DEFAULT_HAPTICGEN_OPENAI_MODEL,
    outputs_dir: str = "outputs/hapticgen_llm_ui",
) -> gr.Blocks:
    """Build the validation UI."""
    client = OpenAIHapticGenPromptClient(model=model)

    def _generate(
        image_1: str | None,
        image_2: str | None,
        image_3: str | None,
        notes: str,
    ) -> tuple[str, str, str, str, str]:
        image_paths = [image_1, image_2, image_3]
        if any(path in (None, "") for path in image_paths):
            raise gr.Error("Please upload exactly 3 images before generating.")

        run_dir = create_session_run_dir(outputs_dir)
        result = client.infer_prompt(image_paths=[str(path) for path in image_paths], notes=notes or "")
        artifact_paths = save_generation_artifacts(
            run_dir=run_dir,
            image_paths=[str(path) for path in image_paths],
            notes=notes or "",
            openai_payload={
                "visual_summary": result["visual_summary"],
                "haptic_prompt": result["haptic_prompt"],
                "negative_prompt": result["negative_prompt"],
                "rationale": result["rationale"],
                "response_text": result["response_text"],
            },
        )
        return (
            result["visual_summary"],
            result["haptic_prompt"],
            result["negative_prompt"],
            result["rationale"],
            artifact_paths["run_dir"],
        )

    def _import_results(
        run_dir: str,
        generated_wav: str | None,
        waveform_image: str | None,
    ) -> tuple[str | None, str | None, str]:
        if not run_dir:
            raise gr.Error("Generate a prompt first so the UI can create a run directory.")
        metadata = import_colab_outputs(
            run_dir=run_dir,
            generated_wav=generated_wav,
            waveform_image=waveform_image,
        )
        status = "Imported Colab outputs into local run directory."
        return metadata["generated_wav"], metadata["waveform_image"], status

    with gr.Blocks(title="3-Image OpenAI to HapticGen Validation") as demo:
        gr.Markdown(
            """
            # 3-Image OpenAI to HapticGen Validation
            Upload 3 ordered images, generate a HapticGen-ready text prompt with OpenAI,
            then import the `wav` and waveform image generated in Colab.
            """
        )

        with gr.Row():
            image_1 = gr.Image(label="image_1", type="filepath")
            image_2 = gr.Image(label="image_2", type="filepath")
            image_3 = gr.Image(label="image_3", type="filepath")
        notes = gr.Textbox(label="optional notes", lines=3, placeholder="Optional timing or tactile notes")
        generate_btn = gr.Button("Generate", variant="primary")

        with gr.Row():
            visual_summary = gr.Textbox(label="OpenAI summary", lines=4)
            haptic_prompt = gr.Textbox(label="HapticGen prompt", lines=4)
        with gr.Row():
            negative_prompt = gr.Textbox(label="negative prompt", lines=2)
            rationale = gr.Textbox(label="rationale", lines=4)
        run_dir_state = gr.Textbox(label="run directory", interactive=False)

        generate_btn.click(
            _generate,
            inputs=[image_1, image_2, image_3, notes],
            outputs=[
                visual_summary,
                haptic_prompt,
                negative_prompt,
                rationale,
                run_dir_state,
            ],
        )

        gr.Markdown("## Import Colab outputs")
        generated_wav_upload = gr.File(label="Upload generated wav", file_types=[".wav"], type="filepath")
        waveform_image_upload = gr.Image(label="Upload waveform image", type="filepath")
        import_btn = gr.Button("Import Colab result")
        audio_output = gr.Audio(label="generated wav")
        waveform_output = gr.Image(label="waveform image")
        import_status = gr.Textbox(label="import status", interactive=False)

        import_btn.click(
            _import_results,
            inputs=[run_dir_state, generated_wav_upload, waveform_image_upload],
            outputs=[audio_output, waveform_output, import_status],
        )

    return demo
