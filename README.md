# ComfyUI Image Detail Contrast

A ComfyUI node for softening or boosting local detail contrast in images and video frames. Negative values soften crunchy textures; positive values enhance fine detail.

## Installation

From your `ComfyUI/custom_nodes` directory:

```sh
git clone https://github.com/D3voz/comfyui-image-detail-contrast.git
```

Restart ComfyUI, then add **Image Detail Contrast** from `image/filters`.

The node uses the PyTorch installation already provided by ComfyUI. No additional Python packages or model downloads are required.

## Inputs

| Input | Default | Description |
| --- | --- | --- |
| `image` | — | An image or batch of video frames. |
| `detail_contrast` | `0.0` | `0` leaves the input unchanged; negative values soften detail and positive values boost it. Range: `-1.0` to `2.0`. |
| `blur_radius` | `24` | Radius of the Gaussian filter used to separate the base image from fine detail. |
| `blur_sigma` | `8.0` | Gaussian filter sigma. |
| `temporal_mode` | `per_frame` | Process frames independently, or use `temporal_smooth` to average the blurred base with adjacent frames. |
| `chunk_size` | `8` | Frames processed together. Lower values reduce temporary processing memory. |

The output is an `IMAGE` with the same shape as the input. Processed values are clamped to `[0, 1]`. Temporal smoothing uses adjacent batch entries as neighbouring video frames, so use `per_frame` for batches of unrelated images.

## Example

For a softer video, start with `detail_contrast = -0.2`, `blur_radius = 24`, `blur_sigma = 8.0`, and `temporal_mode = per_frame`. Connect the image output of your video loader to this node, then connect the result to your video writer. Route the original audio directly from the loader to the writer.

The node is used by Openjutsu's **Fix colours** workflow. Its workflow ID is `ImageDetailContrast`.

## License

MIT. See [LICENSE](LICENSE).
