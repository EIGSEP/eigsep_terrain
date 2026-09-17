import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

class TiledSkyProbSegFormer:
    """
    Produces a full-resolution P(sky) map by running SegFormer on overlapping tiles
    and stitching results with smooth blending to avoid seams.
    """
    def __init__(self, model_id="nvidia/segformer-b0-finetuned-ade-512-512", device="cpu"):
        self.proc = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        ).to(device).eval()
        self.device = device

        id2label = self.model.config.id2label
        self.sky_id = next((i for i, lab in id2label.items() if lab.lower() == "sky"), None)
        self.tree_id = next((i for i, lab in id2label.items() if lab.lower() == "tree"), None)
        if self.sky_id is None:
            raise ValueError("Model has no 'sky' label.")
        if self.tree_id is None:
            raise ValueError("Model has no 'tree' label.")

        # Try to disable resizing so we can feed native-resolution tiles
        # (Some processors accept these kwargs; if not, we handle by passing do_resize=False in call.)
        if hasattr(self.proc, "do_resize"):
            self.proc.do_resize = False

    @staticmethod
    def _hann2d(h, w, eps=1e-6):
        wy = np.hanning(h)
        wx = np.hanning(w)
        win = np.outer(wy, wx).astype(np.float32)
        win = np.maximum(win, eps)  # avoid division by ~0 at borders
        return win

    @torch.inference_mode()
    def p_sky_tiled(
        self,
        filename,
        tile=1024,
        overlap=256,
        batch=1,
        max_tile_dim=None,
    ):
        """
        Args:
          tile: tile size in pixels (square tiles). Increase until you hit VRAM limits.
          overlap: overlap size in pixels. Helps reduce edge artifacts.
          batch: tiles per forward pass (increase if VRAM allows).
          max_tile_dim: optional cap on tile dims (e.g. 1024) if you pass non-square crops elsewhere.

        Returns:
          psky_full: (H, W) float32 in [0,1]
        """
        img = Image.open(filename).convert("RGB")
        H, W = img.size[1], img.size[0]

        step = tile - overlap
        if step <= 0:
            raise ValueError("overlap must be < tile")

        # Accumulators for blended stitching
        sky_acc = np.zeros((H, W), dtype=np.float32)
        tree_acc = np.zeros((H, W), dtype=np.float32)
        wacc = np.zeros((H, W), dtype=np.float32)

        # Precompute blending window (will be cropped for edge tiles)
        win_full = self._hann2d(tile, tile)

        # Collect tiles for batching
        tiles = []
        metas = []  # (y0, y1, x0, x1, win_crop)

        # generate tile coordinates that cover the whole image
        ys = list(range(0, max(1, H - tile + 1), step))
        xs = list(range(0, max(1, W - tile + 1), step))
        if ys[-1] != H - tile:
            ys.append(max(0, H - tile))
        if xs[-1] != W - tile:
            xs.append(max(0, W - tile))

        for y0 in ys:
            for x0 in xs:
                y1 = min(H, y0 + tile)
                x1 = min(W, x0 + tile)

                # Crop tile from original
                crop = img.crop((x0, y0, x1, y1))

                # If the tile is smaller at borders, we keep it smaller (no padding needed),
                # but we’ll use a matching cropped Hann window.
                th, tw = (y1 - y0), (x1 - x0)
                win = win_full[:th, :tw].copy()

                # Optional dimension cap (rarely needed if tile is already set)
                if max_tile_dim is not None and (th > max_tile_dim or tw > max_tile_dim):
                    crop = crop.resize((min(tw, max_tile_dim), min(th, max_tile_dim)))
                    # If you resize here, you must also adjust stitching coords—so prefer not to use this.

                tiles.append(crop)
                metas.append((y0, y1, x0, x1, win))

                if len(tiles) == batch:
                    self._process_batch(tiles, metas, sky_acc, tree_acc, wacc)
                    tiles, metas = [], []

        if tiles:
            self._process_batch(tiles, metas, sky_acc, tree_acc, wacc)

        psky_full = sky_acc / np.maximum(wacc, 1e-6)
        psky_full = np.clip(psky_full, 0.0, 1.0)
        ptree_full = tree_acc / np.maximum(wacc, 1e-6)
        ptree_full = np.clip(ptree_full, 0.0, 1.0)
        return psky_full, ptree_full

    @torch.inference_mode()
    def _process_batch(self, tiles, metas, sky_acc, tree_acc, wacc):
        # Processor call: explicitly request no resizing if supported
        try:
            inputs = self.proc(images=tiles, return_tensors="pt", do_resize=False)
        except TypeError:
            # Older processors may not accept do_resize in call; rely on self.proc.do_resize=False if available.
            inputs = self.proc(images=tiles, return_tensors="pt")

        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        # Convert logits -> probs -> select sky channel (still low-res per tile)
        probs = torch.softmax(out.logits, dim=1)              # (B, C, h, w)
        psky_lr = probs[:, self.sky_id, :, :]             # (B, h, w)
        ptree_lr = probs[:, self.tree_id, :, :]             # (B, h, w)

        # Stitch each tile back at full tile resolution and blend
        for i, (y0, y1, x0, x1, win) in enumerate(metas):
            th, tw = (y1 - y0), (x1 - x0)

            # Upsample per-tile psky to tile pixel resolution on GPU (small, safe)
            for plr, _acc in [(psky_lr, sky_acc), (ptree_lr, tree_acc)]:
                p = torch.nn.functional.interpolate(
                    plr[i:i+1, None, :, :], size=(th, tw), mode="bilinear", align_corners=False
                )[0, 0]  # (th, tw)
                p = p.float().cpu().numpy()

                _acc[y0:y1, x0:x1] += p * win
            wacc[y0:y1, x0:x1] += win
