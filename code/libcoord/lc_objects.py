import json
from pathlib import Path

import numpy as np
from robosuite.models.objects import MujocoXMLObject

from libero.libero.envs.base_object import OBJECTS_DICT, register_object

CATALOG = Path(__file__).resolve().parent / "assets" / "libero_gso" / "catalog.json"


def _make(entry):
  def __init__(self, name=entry["key"], obj_name=entry["key"], joints=None):
    joints = [dict(type="free", damping="0.0005")] if joints is None else joints
    MujocoXMLObject.__init__(self, entry["xml"], name=name, joints=joints, obj_type="all", duplicate_collision_geoms=False)
    self.category_name = entry["key"]
    self.rotation = (0.0, 0.0)
    self.rotation_axis = "z"
    self.object_properties = {"vis_site_names": {}}
  return type(entry["cls"], (MujocoXMLObject,), {"__init__": __init__})


def register(path=CATALOG):
  catalog = json.loads(Path(path).read_text())
  for entry in catalog:
    if entry["key"] not in OBJECTS_DICT:
      register_object(_make(entry))
      assert entry["key"] in OBJECTS_DICT, entry["key"]
  return catalog
