import json
import string
import sys
from pathlib import Path

import numpy as np

SRC, SEL, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]), Path(sys.argv[3]).resolve()
sel = json.loads(SEL.read_text())
CONTACT = 'solimp="0.998 0.998 0.001" solref="0.001 1" friction="0.95 0.3 0.1"'
LIMITS = dict(place=dict(diam=0.14, height=0.10), clutter=dict(diam=0.18, height=0.22))


def letters(i):
  a = string.ascii_lowercase
  return a[i // 26].upper() + a[i % 26]


def vertices(path):
  v = [list(map(float, line.split()[1:4])) for line in path.read_text().splitlines() if line.startswith("v ")]
  return np.asarray(v)


catalog = []
for kind in ("place", "clutter"):
  for i, name in enumerate(sel[kind]):
    folder = SRC / name
    if not (folder / "model.obj").exists():
      print("missing", name)
      continue
    v = vertices(folder / "model.obj")
    lo, hi = v.min(axis=0), v.max(axis=0)
    ext = hi - lo
    diam = float(max(ext[0], ext[1]))
    scale = min(1.0, LIMITS[kind]["diam"] / diam, LIMITS[kind]["height"] / float(ext[2]))
    lo, hi, ext = lo * scale, hi * scale, ext * scale
    cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
    cls = "Gso" + kind.capitalize() + letters(i)
    key = "gso_" + kind + "_" + letters(i).lower()
    parts = sorted(folder.glob("model_collision_*.obj"), key=lambda p: int(p.stem.split("_")[-1]))
    s = "%.5f %.5f %.5f" % (scale, scale, scale)
    meshes = ['    <mesh name="%s_vis" file="%s" scale="%s"/>' % (key, (folder / "model.obj").as_posix(), s)]
    meshes += ['    <mesh name="%s_c%d" file="%s" scale="%s"/>' % (key, j, p.as_posix(), s) for j, p in enumerate(parts)]
    geoms = ['        <geom type="mesh" mesh="%s_vis" material="%s_mat" contype="0" conaffinity="0" group="1" pos="%.5f %.5f %.5f"/>'
      % (key, key, -cx, -cy, -lo[2])]
    geoms += ['        <geom type="mesh" mesh="%s_c%d" group="0" density="200" %s pos="%.5f %.5f %.5f" rgba="0.8 0.8 0.8 0.3"/>'
      % (key, j, CONTACT, -cx, -cy, -lo[2]) for j in range(len(parts))]
    xml = "\n".join(['<mujoco model="%s">' % key, "  <asset>",
      '    <texture type="2d" name="%s_tex" file="%s"/>' % (key, (folder / "texture.png").as_posix()),
      '    <material name="%s_mat" texture="%s_tex" specular="0.5" shininess="0.5"/>' % (key, key)] + meshes +
      ["  </asset>", "  <worldbody>", "    <body>", '      <body name="object">'] + geoms + ["      </body>",
      '      <site rgba="0 0 0 0" size="0.005" pos="0 0 0" name="bottom_site"/>',
      '      <site rgba="0 0 0 0" size="0.005" pos="0 0 %.5f" name="top_site"/>' % ext[2],
      '      <site rgba="0 0 0 0" size="0.005" pos="%.5f %.5f 0" name="horizontal_radius_site"/>' % (ext[0] / 2, ext[1] / 2),
      "    </body>", "  </worldbody>", "</mujoco>"])
    out = OUT / key
    out.mkdir(parents=True, exist_ok=True)
    (out / (key + ".xml")).write_text(xml)
    catalog.append(dict(name=name, kind=kind, cls=cls, key=key, scale=round(scale, 4), xml=(out / (key + ".xml")).as_posix(),
      size_x=round(float(ext[0]), 4), size_y=round(float(ext[1]), 4), height=round(float(ext[2]), 4), collision_parts=len(parts)))
(OUT / "catalog.json").write_text(json.dumps(catalog, indent=1))
for kind in ("place", "clutter"):
  c = [x for x in catalog if x["kind"] == kind]
  print(kind, len(c), "height %.3f-%.3f" % (min(x["height"] for x in c), max(x["height"] for x in c)),
    "rescaled", sum(1 for x in c if x["scale"] < 1))
