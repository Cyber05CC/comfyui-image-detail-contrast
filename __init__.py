import torch
import torch.nn.functional as F

import comfy.model_management


def gaussian_blur(x, radius, sigma):
    k = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-(k * k) / (2.0 * sigma * sigma))
    k = k / k.sum()
    c = x.shape[1]
    x = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.pad(x, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)


class ImageDetailContrast:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "detail_contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 2.0, "step": 0.01, "tooltip": "Negative softens local detail contrast, positive boosts it."}),
                "blur_radius": ("INT", {"default": 24, "min": 1, "max": 256, "step": 1}),
                "blur_sigma": ("FLOAT", {"default": 8.0, "min": 0.1, "max": 128.0, "step": 0.1}),
                "temporal_mode": (["per_frame", "temporal_smooth"], {"tooltip": "temporal_smooth averages the blurred base over neighbouring frames to reduce flicker on video."}),
                "chunk_size": ("INT", {"default": 8, "min": 1, "max": 1024, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"
    CATEGORY = "image/filters"

    def apply(self, image, detail_contrast, blur_radius, blur_sigma, temporal_mode, chunk_size):
        if detail_contrast == 0.0:
            return (image,)

        device = comfy.model_management.get_torch_device()
        temporal = temporal_mode == "temporal_smooth"
        frames = image.shape[0]
        out = torch.empty_like(image)

        for start in range(0, frames, chunk_size):
            end = min(start + chunk_size, frames)
            # one extra frame on each side so temporal smoothing is seamless across chunks
            lo = max(start - 1, 0) if temporal else start
            hi = min(end + 1, frames) if temporal else end
            x = image[lo:hi].to(device).movedim(-1, 1)
            base = gaussian_blur(x, blur_radius, blur_sigma)
            if temporal and base.shape[0] > 1:
                prev = torch.cat((base[:1], base[:-1]))
                nxt = torch.cat((base[1:], base[-1:]))
                base = 0.5 * base + 0.25 * (prev + nxt)
            x = x[start - lo:start - lo + end - start]
            base = base[start - lo:start - lo + end - start]
            res = torch.lerp(base, x, 1.0 + detail_contrast).clamp_(0.0, 1.0)
            out[start:end] = res.movedim(1, -1).to(out.device)

        return (out,)


NODE_CLASS_MAPPINGS = {
    "ImageDetailContrast": ImageDetailContrast,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDetailContrast": "Image Detail Contrast",
}
