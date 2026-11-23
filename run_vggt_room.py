import sys, glob, os, numpy as np, torch
import torch.nn.init as _init

def _fast_trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0, generator=None):
    with torch.no_grad():
        return tensor.normal_(mean, std).clamp_(a, b)
_init.trunc_normal_ = _fast_trunc_normal_

sys.path.insert(0, os.path.expanduser("~/amc-test/vggt"))
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

folder = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/amc-test/room")
out = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "~/amc-test/room_out")
os.makedirs(out, exist_ok=True)
paths = sorted(p for p in glob.glob(os.path.join(folder, "*"))
               if p.lower().endswith((".jpg", ".jpeg", ".png")))
print(len(paths), "images", flush=True)

model = VGGT.from_pretrained("facebook/VGGT-1B").to("cuda").eval()
images = load_and_preprocess_images(paths).to("cuda")
with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
    preds = model(images)
extri, intri = pose_encoding_to_extri_intri(preds["pose_enc"], images.shape[-2:])

pts = preds["world_points"][0].float().cpu().numpy()
conf = preds["world_points_conf"][0].float().cpu().numpy()
rgb = (images.permute(0, 2, 3, 1).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
mask = conf >= np.percentile(conf, 50)   # keep the more confident half
P, C = pts[mask], rgb[mask]
if len(P) > 1_500_000:
    idx = np.random.choice(len(P), 1_500_000, replace=False)
    P, C = P[idx], C[idx]
print("points kept:", len(P), flush=True)

with open(os.path.join(out, "room.ply"), "w") as f:
    f.write("ply\nformat ascii 1.0\nelement vertex %d\n" % len(P))
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    np.savetxt(f, np.hstack([P, C]), fmt="%.4f %.4f %.4f %d %d %d")
np.savez(os.path.join(out, "cameras.npz"),
         extrinsic=extri[0].float().cpu().numpy(),
         intrinsic=intri[0].float().cpu().numpy(), paths=np.array(paths))
print("saved to", out)