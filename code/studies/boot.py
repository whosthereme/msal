import ctypes
import os
import sys

LIB = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "torch", "lib"))
os.add_dll_directory(LIB)
for name in ("msvcp140.dll", "msvcp140_1.dll", "vcruntime140_1.dll", "libiomp5md.dll", "c10.dll"):
  ctypes.WinDLL(os.path.join(LIB, name))
import torch
