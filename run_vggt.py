import sys, glob, os, time, numpy as np, torch
import torch.nn.init as _init

# Workaround: fast stand-in for trunc_normal_ (init values are overwritten by pretrained weights)
def _fast_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(a, b)
_init.trunc_normal_ = _fast_trunc_normal_

sys.path.insert(0, os.path.expanduser("~/amc-test/vggt"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

device = "cuda"
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
t0 = time.time()
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
print(f"model loaded in {time.time()-t0:.0f}s", flush=True)

paths = sorted(glob.glob(os.path.expanduser("~/amc-test/vggt_frames/cam_*.png")))
print("frames:", paths, flush=True)
images = load_and_preprocess_images(paths).to(device)
with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
    preds = model(images)
extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])
np.set_printoptions(precision=3, suppress=True)
E, K = extri[0].float().cpu().numpy(), intri[0].float().cpu().numpy()
for p, e, k in zip(paths, E, K):
    print(os.path.basename(p)); print("extrinsic (3x4):\n", e); print("intrinsic:\n", k)
np.savez(os.path.expanduser("~/amc-test/vggt_out.npz"), extrinsic=E, intrinsic=K,
         world_points=preds["world_points"].float().cpu().numpy(),
         world_points_conf=preds["world_points_conf"].float().cpu().numpy())
print("saved ~/amc-test/vggt_out.npz")
