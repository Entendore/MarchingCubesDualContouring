"""
Marching Cubes & Dual Contouring — Numba-Accelerated Isosurface Visualization
============================================================================
Install:  pip install pygame PyOpenGL numpy numba
Controls:
  T             - Toggle algorithm (Marching Cubes / Dual Contouring)
  Mouse Drag    - Rotate view          Scroll - Zoom
  Up / Down     - Adjust isovalue      +/-    - Resolution
  1-4           - Scalar field         C      - Color mode
  L             - Toggle lighting      W      - Wireframe
  A             - Animate metaballs    Space  - Auto-rotate
  R             - Rebuild mesh         ESC    - Quit
"""

import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import sys
import time

WIDTH, HEIGHT = 1280, 800

# ── Numba Detection ──────────────────────────────────────────────────────────
try:
    from numba import njit, prange
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False
    print("=" * 60)
    print("  WARNING: Numba not found.")
    print("  Install with:  pip install numba")
    print("  Falling back to pure Python (much slower).")
    print("=" * 60)

# ── MC Lookup Tables ─────────────────────────────────────────────────────────

EDGE_TABLE_PY = [
    0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
    0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
    0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
    0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
    0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
    0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
    0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
    0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
    0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
    0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
    0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
    0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
    0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
    0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
    0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
    0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
    0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
    0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
    0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
    0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
    0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
    0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
    0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
    0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
    0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
    0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
    0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
    0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
    0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
    0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
    0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
    0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
]

TRI_TABLE_PY = [
    [-1],[0,8,3,-1],[0,1,9,-1],[1,8,3,9,8,1,-1],[1,2,10,-1],[0,8,3,1,2,10,-1],
    [9,2,10,0,2,9,-1],[2,8,3,2,10,8,10,9,8,-1],[3,11,2,-1],[0,11,2,8,11,0,-1],
    [1,9,0,2,3,11,-1],[1,11,2,1,9,11,9,8,11,-1],[3,10,1,11,10,3,-1],
    [0,10,1,0,8,10,8,11,10,-1],[3,9,0,3,11,9,11,10,9,-1],[9,8,10,10,8,11,-1],
    [4,7,8,-1],[4,3,0,7,3,4,-1],[0,1,9,8,4,7,-1],[4,1,9,4,7,1,7,3,1,-1],
    [1,2,10,8,4,7,-1],[3,4,7,3,0,4,1,2,10,-1],[9,2,10,9,0,2,8,4,7,-1],
    [2,10,9,2,9,7,2,7,3,7,9,4,-1],[8,4,7,3,11,2,-1],[11,4,7,11,2,4,2,0,4,-1],
    [9,0,1,8,4,7,2,3,11,-1],[4,7,11,9,4,11,9,11,2,9,2,1,-1],
    [3,10,1,3,11,10,7,8,4,-1],[1,11,10,1,4,11,1,0,4,7,11,4,-1],
    [4,7,8,9,0,11,9,11,10,11,0,3,-1],[4,7,11,4,11,9,9,11,10,-1],
    [9,5,4,-1],[9,5,4,0,8,3,-1],[0,5,4,1,5,0,-1],[8,5,4,8,3,5,3,1,5,-1],
    [1,2,10,9,5,4,-1],[3,0,8,1,2,10,4,9,5,-1],[5,2,10,5,4,2,4,0,2,-1],
    [2,10,5,3,2,5,3,5,4,3,4,8,-1],[9,5,4,2,3,11,-1],[0,11,2,0,8,11,4,9,5,-1],
    [0,5,4,0,1,5,2,3,11,-1],[2,1,5,2,5,8,2,8,11,4,8,5,-1],
    [10,3,11,10,1,3,9,5,4,-1],[4,9,5,0,8,1,8,10,1,8,11,10,-1],
    [5,4,0,5,0,11,5,11,10,11,0,3,-1],[5,4,8,5,8,10,10,8,11,-1],
    [9,7,8,5,7,9,-1],[9,3,0,9,5,3,5,7,3,-1],[0,7,8,0,1,7,1,5,7,-1],
    [1,5,3,3,5,7,-1],[9,7,8,9,5,7,10,1,2,-1],[10,1,2,9,5,0,5,3,0,5,7,3,-1],
    [8,0,2,8,2,5,8,5,7,10,5,2,-1],[2,10,5,2,5,3,3,5,7,-1],
    [7,9,5,7,8,9,3,11,2,-1],[9,5,7,9,7,2,9,2,0,2,7,11,-1],
    [2,3,11,0,1,8,1,7,8,1,5,7,-1],[11,2,1,11,1,7,7,1,5,-1],
    [9,5,8,8,5,7,10,1,3,10,3,11,-1],[5,7,0,5,0,9,7,11,0,1,0,10,11,10,0,-1],
    [11,10,0,11,0,3,10,5,0,8,0,7,5,7,0,-1],[11,10,5,7,11,5,-1],
    [10,6,5,-1],[0,8,3,5,10,6,-1],[9,0,1,5,10,6,-1],[1,8,3,1,9,8,5,10,6,-1],
    [1,6,5,2,6,1,-1],[1,6,5,1,2,6,3,0,8,-1],[9,6,5,9,0,6,0,2,6,-1],
    [5,9,8,5,8,2,5,2,6,3,2,8,-1],[2,3,11,10,6,5,-1],[0,8,11,0,11,2,10,6,5,-1],
    [0,1,9,2,3,11,5,10,6,-1],[5,10,6,1,9,2,9,11,2,9,8,11,-1],
    [6,3,11,6,5,3,5,1,3,-1],[0,8,11,0,11,5,0,5,1,5,11,6,-1],
    [3,11,6,0,3,6,0,6,5,0,5,9,-1],[6,5,9,6,9,11,11,9,8,-1],
    [5,10,6,4,7,8,-1],[4,3,0,4,7,3,6,5,10,-1],[1,9,0,5,10,6,8,4,7,-1],
    [10,6,5,1,9,7,1,7,3,7,9,4,-1],[6,1,2,6,5,1,4,7,8,-1],
    [1,2,5,5,2,6,3,0,4,3,4,7,-1],[8,4,7,9,0,5,0,6,5,0,2,6,-1],
    [7,3,9,7,9,4,3,2,9,5,9,6,2,6,9,-1],[3,11,2,7,8,4,10,6,5,-1],
    [5,10,6,4,7,2,4,2,0,2,7,11,-1],[0,1,9,4,7,8,2,3,11,5,10,6,-1],
    [9,2,1,9,11,2,9,4,11,7,11,4,5,10,6,-1],[8,4,7,3,11,5,3,5,1,5,11,6,-1],
    [5,1,11,5,11,6,1,0,11,7,11,4,0,4,11,-1],
    [0,5,9,0,6,5,0,3,6,11,6,3,8,4,7,-1],[6,5,9,6,9,11,4,7,9,7,11,9,-1],
    [10,4,9,6,4,10,-1],[4,10,6,4,9,10,0,8,3,-1],[10,0,1,10,6,0,6,4,0,-1],
    [8,3,1,8,1,6,8,6,4,6,1,10,-1],[1,4,9,1,2,4,2,6,4,-1],
    [3,0,8,1,2,9,2,4,9,2,6,4,-1],[0,2,4,4,2,6,-1],[8,3,2,8,2,4,4,2,6,-1],
    [10,4,9,10,6,4,11,2,3,-1],[0,8,2,2,8,11,4,9,10,4,10,6,-1],
    [3,11,2,0,1,6,0,6,4,6,1,10,-1],[6,4,1,6,1,10,4,8,1,2,1,11,8,11,1,-1],
    [9,6,4,9,3,6,9,1,3,11,6,3,-1],[8,11,1,8,1,0,11,6,1,9,1,4,6,4,1,-1],
    [3,11,6,3,6,0,0,6,4,-1],[6,4,8,11,6,8,-1],[7,10,6,7,8,10,8,9,10,-1],
    [0,7,3,0,10,7,0,9,10,6,7,10,-1],[10,6,7,1,10,7,1,7,8,1,8,0,-1],
    [10,6,7,10,7,1,1,7,3,-1],[1,2,6,1,6,8,1,8,9,8,6,7,-1],
    [2,6,9,2,9,1,6,7,9,0,9,3,7,3,9,-1],[7,8,0,7,0,6,6,0,2,-1],
    [7,3,2,6,7,2,-1],[2,3,11,10,6,8,10,8,9,8,6,7,-1],
    [2,0,7,2,7,11,0,9,7,6,7,10,9,10,7,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1],
    [1,2,10,3,0,8,6,11,7,-1],
    [2,3,11,10,6,8,10,8,9,8,6,7,-1],
    [1,11,2,1,7,11,1,5,0,5,3,0,5,7,3,-1],
    [0,3,8,1,2,10,4,9,5,-1],[4,9,5,1,2,10,3,0,8,-1],
    [5,2,10,5,4,2,4,0,2,-1],[2,10,5,3,2,5,3,5,4,3,4,8,-1],
    [9,5,4,10,1,6,1,7,6,1,3,7,-1],
    [1,6,10,1,7,6,1,0,7,8,7,0,9,5,4,-1],
    [4,0,10,4,10,5,0,3,10,6,10,7,3,7,10,-1],
    [7,6,10,7,10,8,5,4,10,4,8,10,-1],
    [6,9,5,6,11,9,11,8,9,-1],[3,6,11,0,6,3,0,5,6,0,9,5,-1],
    [0,11,8,0,5,11,0,1,5,5,6,11,-1],[6,11,3,6,3,5,5,3,1,-1],
    [1,2,10,9,5,11,9,11,8,11,5,6,-1],
    [0,11,3,0,6,11,0,9,6,5,6,9,1,2,10,-1],
    [11,8,5,11,5,6,8,0,5,10,5,2,0,2,5,-1],
    [6,11,3,6,3,5,2,10,3,10,5,3,-1],
    [5,8,9,5,2,8,5,6,2,3,8,2,-1],[9,5,6,9,6,0,0,6,2,-1],
    [1,5,8,1,8,0,5,6,8,3,8,2,6,2,8,-1],[1,5,6,2,1,6,-1],
    [1,3,6,1,6,10,3,8,6,5,6,9,8,9,6,-1],
    [10,1,0,10,0,6,9,5,0,5,6,0,-1],
    [0,3,8,5,10,6,-1],[10,5,6,-1],
    [11,5,10,7,5,11,-1],[11,5,10,11,7,5,8,3,0,-1],
    [5,11,7,5,10,11,1,9,0,-1],[10,7,5,10,11,7,9,8,1,8,3,1,-1],
    [11,1,2,11,7,1,7,5,1,-1],[0,8,3,1,2,7,1,7,5,7,2,11,-1],
    [9,7,5,9,2,7,9,0,2,2,11,7,-1],[7,5,2,7,2,11,5,9,2,3,2,8,9,8,2,-1],
    [2,5,10,2,3,5,3,7,5,-1],[8,2,0,8,5,2,8,7,5,10,2,5,-1],
    [9,0,1,5,10,3,5,3,7,3,10,2,-1],
    [9,8,2,9,2,1,8,7,2,10,2,5,7,5,2,-1],
    [1,3,5,3,7,5,-1],[0,8,7,0,7,1,1,7,5,-1],
    [9,0,3,9,3,5,5,3,7,-1],[9,8,7,5,9,7,-1],
    [5,8,4,5,10,8,10,11,8,-1],[5,0,4,5,11,0,5,10,11,11,3,0,-1],
    [0,1,9,8,4,10,8,10,11,10,4,5,-1],
    [10,11,4,10,4,5,11,3,4,9,4,1,3,1,4,-1],
    [2,5,1,2,8,5,2,11,8,4,5,8,-1],
    [0,4,11,0,11,3,4,5,11,2,11,1,5,1,11,-1],
    [0,2,5,0,5,9,2,11,5,4,5,8,11,8,5,-1],
    [9,4,5,2,3,11,-1],[2,5,10,3,5,2,3,4,5,3,8,4,-1],
    [5,10,2,5,2,4,4,2,0,-1],
    [3,10,2,3,5,10,3,8,5,4,5,8,0,1,9,-1],
    [5,10,2,5,2,4,1,9,2,9,4,2,-1],
    [8,4,5,8,5,3,3,5,1,-1],[0,4,5,1,0,5,-1],
    [8,4,5,8,5,3,9,0,5,0,3,5,-1],[9,4,5,-1],
    [4,11,7,4,9,11,9,10,11,-1],[0,8,3,4,9,7,9,11,7,9,10,11,-1],
    [1,10,11,1,11,4,1,4,0,7,4,11,-1],
    [3,1,4,3,4,8,1,10,4,7,4,11,10,11,4,-1],
    [4,11,7,9,11,4,9,2,11,9,1,2,-1],
    [9,7,4,9,11,7,9,1,11,2,11,1,0,8,3,-1],
    [11,7,4,11,4,2,2,4,0,-1],[11,7,4,11,4,2,8,3,4,3,2,4,-1],
    [2,9,10,2,7,9,2,3,7,7,4,9,-1],
    [9,10,7,9,7,4,10,2,7,8,7,0,2,0,7,-1],
    [3,7,10,3,10,2,7,4,10,1,10,0,4,0,10,-1],
    [1,10,2,8,7,4,-1],[4,9,1,4,1,7,7,1,3,-1],
    [4,9,1,4,1,7,0,8,1,8,7,1,-1],[4,0,3,7,4,3,-1],[4,8,7,-1],
    [9,10,8,10,11,8,-1],[3,0,9,3,9,11,11,9,10,-1],
    [0,1,10,0,10,8,8,10,11,-1],[3,1,10,11,3,10,-1],
    [1,2,11,1,11,9,9,11,8,-1],[3,0,9,3,9,11,1,2,9,2,11,9,-1],
    [0,2,11,8,0,11,-1],[3,2,11,-1],[2,3,8,2,8,10,10,8,9,-1],
    [9,10,2,0,9,2,-1],[2,3,8,2,8,10,0,1,8,1,10,8,-1],[1,10,2,-1],
    [1,3,8,9,1,8,-1],[9,0,1,-1],[0,3,8,-1],[-1],
]

EDGE_TABLE_NP = np.array(EDGE_TABLE_PY, dtype=np.int32)
TRI_TABLE_FLAT = np.full((256, 16), -1, dtype=np.int32)
for _i, _t in enumerate(TRI_TABLE_PY):
    for _j in range(min(len(_t), 16)):
        TRI_TABLE_FLAT[_i, _j] = _t[_j]
CUBE_VERTS_NP = np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1],
    [0,1,0],[1,1,0],[1,1,1],[0,1,1]], dtype=np.int32)
EDGE_PAIRS_NP = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],
    [0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
CELL_EDGE_OFFSETS = np.array([
    [0,0,0,0],[0,0,1,0],[0,0,0,1],[0,0,1,1],
    [1,0,0,0],[1,1,0,0],[1,0,0,1],[1,1,0,1],
    [2,0,0,0],[2,1,0,0],[2,0,1,0],[2,1,1,0]], dtype=np.int32)

# Python-mode constants
CV_PY = [(0,0,0),(1,0,0),(1,0,1),(0,0,1),(0,1,0),(1,1,0),(1,1,1),(0,1,1)]
EP_PY = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]


# ===================== SCALAR FIELD =====================

def compute_field(ft, sz, t=0.0):
    lin = np.linspace(-1.5, 1.5, sz, dtype=np.float32)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    if ft == 0:
        return np.sqrt(X**2+Y**2+Z**2)-0.8
    elif ft == 1:
        R, r = 0.7, 0.3
        return np.sqrt((np.sqrt(X**2+Z**2)-R)**2+Y**2)-r
    elif ft == 2:
        f = np.zeros_like(X)
        ctrs = [(0.4*math.cos(t*0.7),0.4*math.sin(t*0.5),0.4*math.sin(t*0.3)),
                (-0.3*math.sin(t*0.6),0.3*math.cos(t*0.8),-0.3*math.cos(t*0.4)),
                (0.2*math.cos(t*1.1),-0.4*math.sin(t*0.4),0.2*math.sin(t*0.9))]
        for (cx,cy,cz),m in zip(ctrs,[0.15,0.12,0.10]):
            f += m/((X-cx)**2+(Y-cy)**2+(Z-cz)**2+0.001)
        return f
    elif ft == 3:
        return (np.sin(2*X+t)*np.cos(2*Y+t*0.7)+np.sin(2*Y+t*0.5)*np.cos(2*Z+t*0.3)
                +np.cos(2*Z+t*0.9)*np.sin(2*X-t*0.6))/3.0
    return np.sqrt(X**2+Y**2+Z**2)-0.8


def compute_gradient(f):
    gx=np.zeros_like(f); gy=np.zeros_like(f); gz=np.zeros_like(f)
    gx[1:-1,:,:]=(f[2:,:,:]-f[:-2,:,:])/2.0; gy[:,1:-1,:]=(f[:,2:,:]-f[:,:-2,:])/2.0
    gz[:,:,1:-1]=(f[:,:,2:]-f[:,:,:-2])/2.0
    gx[0,:,:]=f[1,:,:]-f[0,:,:]; gx[-1,:,:]=f[-1,:,:]-f[-2,:,:]
    gy[:,0,:]=f[:,1,:]-f[:,0,:]; gy[:,-1,:]=f[:,-1,:]-f[:,-2,:]
    gz[:,:,0]=f[:,:,1]-f[:,:,0]; gz[:,:,-1]=f[:,:,-1]-f[:,:,-2]
    return gx, gy, gz


# ===================== NUMBA MARCHING CUBES =====================

if USE_NUMBA:
    @njit(parallel=True, cache=True, fastmath=True)
    def _mc_extract(field, iso, et, tt, cv, ep):
        nx,ny,nz=field.shape; ncx,ncy,ncz=nx-1,ny-1,nz-1; h=float(nx-1)*0.5
        cts=np.zeros(ncx,dtype=np.int64)
        for ix in prange(ncx):
            c=0
            for iy in range(ncy):
                for iz in range(ncz):
                    vs=np.array([field[ix,iy,iz],field[ix+1,iy,iz],field[ix+1,iy,iz+1],
                        field[ix,iy,iz+1],field[ix,iy+1,iz],field[ix+1,iy+1,iz],
                        field[ix+1,iy+1,iz+1],field[ix,iy+1,iz+1]],dtype=np.float32)
                    ci=0
                    for i in range(8):
                        if vs[i]>=iso: ci|=(1<<i)
                    if et[ci]==0: continue
                    tr=tt[ci]; ti=0
                    while ti<16 and tr[ti]!=-1: c+=1; ti+=3
            cts[ix]=c
        offs=np.zeros(ncx,dtype=np.int64)
        for i in range(1,ncx): offs[i]=offs[i-1]+cts[i-1]
        total=int(offs[-1]+cts[-1]) if ncx>0 else 0
        vs=np.empty((total*3,3),dtype=np.float32); ns=np.empty((total*3,3),dtype=np.float32)
        for ix in prange(ncx):
            off=offs[ix]; lc=0
            for iy in range(ncy):
                for iz in range(ncz):
                    vals=np.array([field[ix,iy,iz],field[ix+1,iy,iz],field[ix+1,iy,iz+1],
                        field[ix,iy,iz+1],field[ix,iy+1,iz],field[ix+1,iy+1,iz],
                        field[ix+1,iy+1,iz+1],field[ix,iy+1,iz+1]],dtype=np.float32)
                    ci=0
                    for i in range(8):
                        if vals[i]>=iso: ci|=(1<<i)
                    eb=et[ci]
                    if eb==0: continue
                    ev=np.empty((12,3),dtype=np.float32)
                    for e in range(12):
                        if eb&(1<<e):
                            i0=ep[e,0]; i1=ep[e,1]
                            p0x=float(ix)+cv[i0,0]; p0y=float(iy)+cv[i0,1]; p0z=float(iz)+cv[i0,2]
                            p1x=float(ix)+cv[i1,0]; p1y=float(iy)+cv[i1,1]; p1z=float(iz)+cv[i1,2]
                            d=vals[i1]-vals[i0]
                            t=0.5 if abs(d)<1e-12 else (iso-vals[i0])/d
                            ev[e,0]=p0x+t*(p1x-p0x); ev[e,1]=p0y+t*(p1y-p0y); ev[e,2]=p0z+t*(p1z-p0z)
                    tr=tt[ci]; ti=0
                    while ti<16 and tr[ti]!=-1:
                        p0x,p0y,p0z=ev[tr[ti],0],ev[tr[ti],1],ev[tr[ti],2]
                        p1x,p1y,p1z=ev[tr[ti+1],0],ev[tr[ti+1],1],ev[tr[ti+1],2]
                        p2x,p2y,p2z=ev[tr[ti+2],0],ev[tr[ti+2],1],ev[tr[ti+2],2]
                        e1x=p1x-p0x; e1y=p1y-p0y; e1z=p1z-p0z
                        e2x=p2x-p0x; e2y=p2y-p0y; e2z=p2z-p0z
                        fnx=e1y*e2z-e1z*e2y; fny=e1z*e2x-e1x*e2z; fnz=e1x*e2y-e1y*e2x
                        ln=math.sqrt(fnx*fnx+fny*fny+fnz*fnz)
                        if ln>1e-12: fnx/=ln; fny/=ln; fnz/=ln
                        b=(off+lc)*3
                        vs[b,0]=p0x-h; vs[b,1]=p0y-h; vs[b,2]=p0z-h; ns[b,0]=fnx; ns[b,1]=fny; ns[b,2]=fnz
                        vs[b+1,0]=p1x-h; vs[b+1,1]=p1y-h; vs[b+1,2]=p1z-h; ns[b+1]=ns[b]
                        vs[b+2,0]=p2x-h; vs[b+2,1]=p2y-h; vs[b+2,2]=p2z-h; ns[b+2]=ns[b]
                        lc+=1; ti+=3
        return vs,ns,total

    def extract_mc(field, iso):
        v,n,c=_mc_extract(field,float(iso),EDGE_TABLE_NP,TRI_TABLE_FLAT,CUBE_VERTS_NP,EDGE_PAIRS_NP)
        return v.ravel(),n.ravel(),c


# ===================== NUMBA DUAL CONTOURING =====================

if USE_NUMBA:
    @njit(cache=True)
    def _solve_qef(M, rhs):
        reg=1e-6
        a00=M[0,0]+reg; a01=M[0,1]; a02=M[0,2]; a11=M[1,1]+reg; a12=M[1,2]; a22=M[2,2]+reg
        det=a00*(a11*a22-a12*a12)-a01*(a01*a22-a12*a02)+a02*(a01*a12-a11*a02)
        if abs(det)<1e-15: return np.array([0.0,0.0,0.0])
        id=1.0/det
        x0=id*(rhs[0]*(a11*a22-a12*a12)-a01*(rhs[1]*a22-a12*rhs[2])+a02*(rhs[1]*a12-a11*rhs[2]))
        x1=id*(a00*(rhs[1]*a22-a12*rhs[2])-rhs[0]*(a01*a22-a12*a02)+a02*(a01*rhs[2]-rhs[1]*a02))
        x2=id*(a00*(a11*rhs[2]-rhs[1]*a12)-a01*(a01*rhs[2]-rhs[1]*a02)+rhs[0]*(a01*a12-a11*a02))
        return np.array([x0,x1,x2])

    @njit(cache=True, fastmath=True)
    def _dc_extract(field, gx, gy, gz, iso, ceo):
        nx,ny,nz=field.shape; ncx,ncy,ncz=nx-1,ny-1,nz-1
        h=float(nx-1)*0.5; stride=nx*ny*nz; te=3*stride
        ef=np.zeros(te,dtype=np.int32); ep=np.zeros((te,3),dtype=np.float32)
        en=np.zeros((te,3),dtype=np.float32)
        for ix in range(nx-1):
            for iy in range(ny):
                for iz in range(nz):
                    va=field[ix,iy,iz]; vb=field[ix+1,iy,iz]
                    if (va-iso)*(vb-iso)<0:
                        dv=vb-va; t=(iso-va)/dv if abs(dv)>1e-12 else 0.5
                        idx=ix*(ny*nz)+iy*nz+iz; ef[idx]=1
                        ep[idx,0]=ix+t; ep[idx,1]=float(iy); ep[idx,2]=float(iz)
                        en[idx,0]=gx[ix,iy,iz]*(1-t)+gx[ix+1,iy,iz]*t
                        en[idx,1]=gy[ix,iy,iz]*(1-t)+gy[ix+1,iy,iz]*t
                        en[idx,2]=gz[ix,iy,iz]*(1-t)+gz[ix+1,iy,iz]*t
                        ln=math.sqrt(en[idx,0]**2+en[idx,1]**2+en[idx,2]**2)
                        if ln>1e-12: en[idx]/=ln
        o1=stride
        for ix in range(nx):
            for iy in range(ny-1):
                for iz in range(nz):
                    va=field[ix,iy,iz]; vb=field[ix,iy+1,iz]
                    if (va-iso)*(vb-iso)<0:
                        dv=vb-va; t=(iso-va)/dv if abs(dv)>1e-12 else 0.5
                        idx=o1+ix*(ny*nz)+iy*nz+iz; ef[idx]=1
                        ep[idx,0]=float(ix); ep[idx,1]=iy+t; ep[idx,2]=float(iz)
                        en[idx,0]=gx[ix,iy,iz]*(1-t)+gx[ix,iy+1,iz]*t
                        en[idx,1]=gy[ix,iy,iz]*(1-t)+gy[ix,iy+1,iz]*t
                        en[idx,2]=gz[ix,iy,iz]*(1-t)+gz[ix,iy+1,iz]*t
                        ln=math.sqrt(en[idx,0]**2+en[idx,1]**2+en[idx,2]**2)
                        if ln>1e-12: en[idx]/=ln
        o2=2*stride
        for ix in range(nx):
            for iy in range(ny):
                for iz in range(nz-1):
                    va=field[ix,iy,iz]; vb=field[ix,iy,iz+1]
                    if (va-iso)*(vb-iso)<0:
                        dv=vb-va; t=(iso-va)/dv if abs(dv)>1e-12 else 0.5
                        idx=o2+ix*(ny*nz)+iy*nz+iz; ef[idx]=1
                        ep[idx,0]=float(ix); ep[idx,1]=float(iy); ep[idx,2]=iz+t
                        en[idx,0]=gx[ix,iy,iz]*(1-t)+gx[ix,iy,iz+1]*t
                        en[idx,1]=gy[ix,iy,iz]*(1-t)+gy[ix,iy,iz+1]*t
                        en[idx,2]=gz[ix,iy,iz]*(1-t)+gz[ix,iy,iz+1]*t
                        ln=math.sqrt(en[idx,0]**2+en[idx,1]**2+en[idx,2]**2)
                        if ln>1e-12: en[idx]/=ln
        cvp=np.zeros((ncx,ncy,ncz,3),dtype=np.float32)
        cvn=np.zeros((ncx,ncy,ncz,3),dtype=np.float32)
        cvv=np.zeros((ncx,ncy,ncz),dtype=np.int32)
        for cx in range(ncx):
            for cy in range(ncy):
                for cz in range(ncz):
                    nc=0; cp=np.zeros((12,3),dtype=np.float64); cn=np.zeros((12,3),dtype=np.float64)
                    for e in range(12):
                        ax=ceo[e,0]; eix=cx+ceo[e,1]; eiy=cy+ceo[e,2]; eiz=cz+ceo[e,3]
                        if ax==0 and eix>=nx: continue
                        if ax==1 and eiy>=ny: continue
                        if ax==2 and eiz>=nz: continue
                        eidx=ax*stride+eix*(ny*nz)+eiy*nz+eiz
                        if ef[eidx]:
                            cp[nc]=ep[eidx]; cn[nc]=en[eidx]; nc+=1
                    if nc==0: continue
                    cvv[cx,cy,cz]=1
                    if nc==1:
                        cvp[cx,cy,cz]=cp[0]; cvn[cx,cy,cz]=cn[0]
                    else:
                        M=np.zeros((3,3),dtype=np.float64); rhs=np.zeros(3,dtype=np.float64)
                        for k in range(nc):
                            n0,n1,n2=cn[k,0],cn[k,1],cn[k,2]
                            d=n0*cp[k,0]+n1*cp[k,1]+n2*cp[k,2]
                            M[0,0]+=n0*n0; M[0,1]+=n0*n1; M[0,2]+=n0*n2
                            M[1,0]+=n1*n0; M[1,1]+=n1*n1; M[1,2]+=n1*n2
                            M[2,0]+=n2*n0; M[2,1]+=n2*n1; M[2,2]+=n2*n2
                            rhs[0]+=n0*d; rhs[1]+=n1*d; rhs[2]+=n2*d
                        p=_solve_qef(M,rhs)
                        cvp[cx,cy,cz,0]=max(cx,min(cx+1.0,p[0]))
                        cvp[cx,cy,cz,1]=max(cy,min(cy+1.0,p[1]))
                        cvp[cx,cy,cz,2]=max(cz,min(cz+1.0,p[2]))
                        anx=any_=anz=0.0
                        for k in range(nc): anx+=cn[k,0]; any_+=cn[k,1]; anz+=cn[k,2]
                        ln=math.sqrt(anx*anx+any_*any_+anz*anz)
                        if ln>1e-12: anx/=ln; any_/=ln; anz/=ln
                        else: anx=0.0; any_=1.0; anz=0.0
                        cvn[cx,cy,cz,0]=anx; cvn[cx,cy,cz,1]=any_; cvn[cx,cy,cz,2]=anz
        verts=np.empty((te*4,3),dtype=np.float32); norms=np.empty((te*4,3),dtype=np.float32)
        qc=0
        for ei in range(te):
            if ef[ei]==0: continue
            ax=ei//stride; loc=ei%stride
            eix=loc//(ny*nz); rem=loc%(ny*nz); eiy=rem//nz; eiz=rem%nz
            c0x=c0y=c0z=c1x=c1y=c1z=c2x=c2y=c2z=c3x=c3y=c3z=0
            if ax==0: c0x=eix;c0y=eiy;c0z=eiz; c1x=eix;c1y=eiy-1;c1z=eiz; c2x=eix;c2y=eiy-1;c2z=eiz-1; c3x=eix;c3y=eiy;c3z=eiz-1
            elif ax==1: c0x=eix;c0y=eiy;c0z=eiz; c1x=eix;c1y=eiy;c1z=eiz-1; c2x=eix-1;c2y=eiy;c2z=eiz-1; c3x=eix-1;c3y=eiy;c3z=eiz
            else: c0x=eix;c0y=eiy;c0z=eiz; c1x=eix-1;c1y=eiy;c1z=eiz; c2x=eix-1;c2y=eiy-1;c2z=eiz; c3x=eix;c3y=eiy-1;c3z=eiz
            cx_=np.array([c0x,c1x,c2x,c3x],dtype=np.int32)
            cy_=np.array([c0y,c1y,c2y,c3y],dtype=np.int32)
            cz_=np.array([c0z,c1z,c2z,c3z],dtype=np.int32)
            ok=True
            for c in range(4):
                if cx_[c]<0 or cx_[c]>=ncx or cy_[c]<0 or cy_[c]>=ncy or cz_[c]<0 or cz_[c]>=ncz: ok=False; break
                if cvv[cx_[c],cy_[c],cz_[c]]==0: ok=False; break
            if not ok: continue
            pos=np.empty((4,3),dtype=np.float32); nrm=np.empty((4,3),dtype=np.float32)
            for c in range(4): pos[c]=cvp[cx_[c],cy_[c],cz_[c]]; nrm[c]=cvn[cx_[c],cy_[c],cz_[c]]
            e1x=pos[1,0]-pos[0,0]; e1y=pos[1,1]-pos[0,1]; e1z=pos[1,2]-pos[0,2]
            e2x=pos[3,0]-pos[0,0]; e2y=pos[3,1]-pos[0,1]; e2z=pos[3,2]-pos[0,2]
            fnx=e1y*e2z-e1z*e2y; fny=e1z*e2x-e1x*e2z; fnz=e1x*e2y-e1y*e2x
            dot=fnx*en[ei,0]+fny*en[ei,1]+fnz*en[ei,2]
            order=np.array([0,1,2,3],dtype=np.int32)
            if dot<0: order=np.array([0,3,2,1],dtype=np.int32)
            b=qc*4
            for c in range(4):
                o=order[c]
                verts[b+c,0]=pos[o,0]-h; verts[b+c,1]=pos[o,1]-h; verts[b+c,2]=pos[o,2]-h
                norms[b+c]=nrm[o]
            qc+=1
        return verts[:qc*4],norms[:qc*4],qc

    def extract_dc(field, iso):
        gxf,gyf,gzf=compute_gradient(field)
        v,n,c=_dc_extract(field,gxf,gyf,gzf,float(iso),CELL_EDGE_OFFSETS)
        return v.ravel(),n.ravel(),c


# ===================== NUMBA COLORS =====================

if USE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _compute_colors(verts, norms, mode):
        n=len(verts)//3; colors=np.empty((n,3),dtype=np.float32)
        for i in range(n):
            vx=verts[i*3]; vy=verts[i*3+1]; vz=verts[i*3+2]
            nx_=norms[i*3]; ny_=norms[i*3+1]; nz_=norms[i*3+2]
            if mode==0: colors[i,0]=0.3; colors[i,1]=0.6; colors[i,2]=1.0
            elif mode==1: colors[i,0]=0.5+0.5*vx/1.5; colors[i,1]=0.5+0.5*vy/1.5; colors[i,2]=0.5+0.5*vz/1.5
            elif mode==2: colors[i,0]=0.5+0.5*abs(nx_); colors[i,1]=0.5+0.5*abs(ny_); colors[i,2]=0.5+0.5*abs(nz_)
            else:
                t=(vy+1.5)/3.0
                if t<0.0: t=0.0
                if t>1.0: t=1.0
                if t<0.25: s=t/0.25; colors[i,0]=0.0; colors[i,1]=s; colors[i,2]=1.0
                elif t<0.5: s=(t-0.25)/0.25; colors[i,0]=0.0; colors[i,1]=1.0; colors[i,2]=1.0-s
                elif t<0.75: s=(t-0.5)/0.25; colors[i,0]=s; colors[i,1]=1.0; colors[i,2]=0.0
                else: s=(t-0.75)/0.25; colors[i,0]=1.0; colors[i,1]=1.0-s; colors[i,2]=0.0
        return colors.ravel()

    def compute_colors(vf, nf, m): return _compute_colors(vf, nf, m)


# ===================== PYTHON FALLBACKS =====================

if not USE_NUMBA:
    def extract_mc(field, iso):
        nx,ny,nz=field.shape; half=nx/2.0; vl=[]; nl=[]
        for ix in range(nx-1):
            for iy in range(ny-1):
                for iz in range(nz-1):
                    vals=(field[ix,iy,iz],field[ix+1,iy,iz],field[ix+1,iy,iz+1],
                          field[ix,iy,iz+1],field[ix,iy+1,iz],field[ix+1,iy+1,iz],
                          field[ix+1,iy+1,iz+1],field[ix,iy+1,iz+1])
                    ci=sum(((1<<i) for i in range(8) if vals[i]>=iso))
                    eb=EDGE_TABLE_PY[ci]
                    if eb==0: continue
                    ev=[None]*12
                    for e in range(12):
                        if eb&(1<<e):
                            i0,i1=EP_PY[e]
                            v0=(ix+CV_PY[i0][0],iy+CV_PY[i0][1],iz+CV_PY[i0][2])
                            v1=(ix+CV_PY[i1][0],iy+CV_PY[i1][1],iz+CV_PY[i1][2])
                            d=vals[i1]-vals[i0]; t=0.5 if abs(d)<1e-12 else (iso-vals[i0])/d
                            ev[e]=(v0[0]+t*(v1[0]-v0[0]),v0[1]+t*(v1[1]-v0[1]),v0[2]+t*(v1[2]-v0[2]))
                    tri=TRI_TABLE_PY[ci]; ti=0
                    while tri[ti]!=-1:
                        p0,p1,p2=ev[tri[ti]],ev[tri[ti+1]],ev[tri[ti+2]]
                        e1=(p1[0]-p0[0],p1[1]-p0[1],p1[2]-p0[2])
                        e2=(p2[0]-p0[0],p2[1]-p0[1],p2[2]-p0[2])
                        fn=(e1[1]*e2[2]-e1[2]*e2[1],e1[2]*e2[0]-e1[0]*e2[2],e1[0]*e2[1]-e1[1]*e2[0])
                        ln=math.sqrt(fn[0]**2+fn[1]**2+fn[2]**2)
                        if ln>1e-12: fn=(fn[0]/ln,fn[1]/ln,fn[2]/ln)
                        for p in(p0,p1,p2):
                            vl.extend([p[0]-half,p[1]-half,p[2]-half])
                            nl.extend([fn[0],fn[1],fn[2]])
                        ti+=3
        return vl, nl, len(vl)//9

    def _adj4(axis, ix, iy, iz):
        if axis==0: return[(ix,iy,iz),(ix,iy-1,iz),(ix,iy-1,iz-1),(ix,iy,iz-1)]
        if axis==1: return[(ix,iy,iz),(ix,iy,iz-1),(ix-1,iy,iz-1),(ix-1,iy,iz)]
        return[(ix,iy,iz),(ix-1,iy,iz),(ix-1,iy-1,iz),(ix,iy-1,iz)]

    def extract_dc(field, iso):
        nx,ny,nz=field.shape; half=(nx-1)/2.0; eps=1e-12
        gxf,gyf,gzf=compute_gradient(field)
        xings={}
        ranges=[(0,range(nx-1),range(ny),range(nz)),
                (1,range(nx),range(ny-1),range(nz)),
                (2,range(nx),range(ny),range(nz-1))]
        for ax,rx,ry,rz in ranges:
            for ix in rx:
                for iy in ry:
                    for iz in rz:
                        nb=[(ix,iy,iz),(ix+1,iy,iz),(ix,iy+1,iz),(ix,iy,iz+1)]
                        va,vb=field[nb[ax][0],nb[ax][1],nb[ax][2]],field[nb[ax][0]+(1 if ax==0 else 0),nb[ax][1]+(1 if ax==1 else 0),nb[ax][2]+(1 if ax==2 else 0)]
                        if ax==0: va,vb=field[ix,iy,iz],field[ix+1,iy,iz]
                        elif ax==1: va,vb=field[ix,iy,iz],field[ix,iy+1,iz]
                        else: va,vb=field[ix,iy,iz],field[ix,iy,iz+1]
                        if (va-iso)*(vb-iso)<0:
                            dv=vb-va; t=(iso-va)/dv if abs(dv)>eps else 0.5
                            if ax==0: px,py,pz=ix+t,float(iy),float(iz)
                            elif ax==1: px,py,pz=float(ix),iy+t,float(iz)
                            else: px,py,pz=float(ix),float(iy),iz+t
                            if ax==0: ia,ib=(ix,iy,iz),(ix+1,iy,iz)
                            elif ax==1: ia,ib=(ix,iy,iz),(ix,iy+1,iz)
                            else: ia,ib=(ix,iy,iz),(ix,iy,iz+1)
                            nn=(gxf[ia]*(1-t)+gxf[ib]*t,gyf[ia]*(1-t)+gyf[ib]*t,gzf[ia]*(1-t)+gzf[ib]*t)
                            ln=math.sqrt(nn[0]**2+nn[1]**2+nn[2]**2)
                            if ln>eps: nn=(nn[0]/ln,nn[1]/ln,nn[2]/ln)
                            xings[(ax,ix,iy,iz)]=((px,py,pz),nn)
        ch={}
        for key,hd in xings.items():
            for c in _adj4(*key):
                if 0<=c[0]<nx-1 and 0<=c[1]<ny-1 and 0<=c[2]<nz-1:
                    ch.setdefault(c,[]).append(hd)
        cv={}
        for cell,xs in ch.items():
            cx,cy,cz=cell
            if len(xs)==1: cv[cell]=(xs[0][0],xs[0][1])
            else:
                A=np.array([[x[1][0],x[1][1],x[1][2]] for x in xs])
                b=np.array([x[1][0]*x[0][0]+x[1][1]*x[0][1]+x[1][2]*x[0][2] for x in xs])
                try: p=np.linalg.lstsq(A,b,rcond=None)[0]
                except: p=np.mean([x[0] for x in xs],axis=0)
                p=(max(cx,min(cx+1.0,p[0])),max(cy,min(cy+1.0,p[1])),max(cz,min(cz+1.0,p[2])))
                an=[sum(x[1][k] for x in xs)/len(xs) for k in range(3)]
                ln=math.sqrt(sum(x*x for x in an))
                n=tuple(x/ln for x in an) if ln>eps else (0,1,0)
                cv[cell]=(p,n)
        vl=[]; nl=[]
        for key,(cpt,cn) in xings.items():
            cells=_adj4(*key)
            if not all(c in cv for c in cells): continue
            pos=[cv[c][0] for c in cells]; nrm=[cv[c][1] for c in cells]
            e1=(pos[1][0]-pos[0][0],pos[1][1]-pos[0][1],pos[1][2]-pos[0][2])
            e2=(pos[3][0]-pos[0][0],pos[3][1]-pos[0][1],pos[3][2]-pos[0][2])
            fn=(e1[1]*e2[2]-e1[2]*e2[1],e1[2]*e2[0]-e1[0]*e2[2],e1[0]*e2[1]-e1[1]*e2[0])
            dot=fn[0]*cn[0]+fn[1]*cn[1]+fn[2]*cn[2]
            if dot<0: pos=[pos[0],pos[3],pos[2],pos[1]]; nrm=[nrm[0],nrm[3],nrm[2],nrm[1]]
            for p,n in zip(pos,nrm):
                vl.extend([p[0]-half,p[1]-half,p[2]-half]); nl.extend([n[0],n[1],n[2]])
        return vl,nl,len(vl)//12

    def compute_colors(verts, norms, mode):
        n=len(verts)//3; colors=[]
        for i in range(n):
            vx,vy,vz=verts[i*3],verts[i*3+1],verts[i*3+2]
            nx_,ny_,nz_=norms[i*3],norms[i*3+1],norms[i*3+2]
            if mode==0: colors.extend([0.3,0.6,1.0])
            elif mode==1: colors.extend([0.5+0.5*vx/1.5,0.5+0.5*vy/1.5,0.5+0.5*vz/1.5])
            elif mode==2: colors.extend([0.5+0.5*abs(nx_),0.5+0.5*abs(ny_),0.5+0.5*abs(nz_)])
            else:
                t=max(0.0,min(1.0,(vy+1.5)/3.0))
                if t<0.25: s=t/0.25; colors.extend([0,s,1])
                elif t<0.5: s=(t-0.25)/0.25; colors.extend([0,1,1-s])
                elif t<0.75: s=(t-0.5)/0.25; colors.extend([s,1,0])
                else: s=(t-0.75)/0.25; colors.extend([1,1-s,0])
        return colors


# ===================== TEXT RENDERING =====================

def draw_text_2d(x, y, text, font, color=(255, 255, 255)):
    if not text: return
    ts = font.render(text, True, color, (20, 20, 30))
    td = pygame.image.tostring(ts, "RGBA", True)
    w, h = ts.get_size()
    glRasterPos2f(max(1, x), max(1, HEIGHT - y - h))
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, td)


def draw_hud(font, lines):
    glPushAttrib(GL_ALL_ATTRIB_BITS)
    glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); gluOrtho2D(0, WIDTH, 0, HEIGHT)
    glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()
    glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    for i, (text, color) in enumerate(lines):
        draw_text_2d(12, 12 + i * 22, text, font, color)
    glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW)
    glPopAttrib()


# ===================== MAIN =====================

def main():
    pygame.init()
    pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("MC & DC Isosurface" + (" [NUMBA]" if USE_NUMBA else " [Python]"))
    font = pygame.font.SysFont("consolas", 16)

    glClearColor(0.08, 0.08, 0.12, 1.0)
    glEnable(GL_DEPTH_TEST); glEnable(GL_NORMALIZE); glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHT0); glEnable(GL_LIGHT1)
    glLightfv(GL_LIGHT0, GL_POSITION, [2,3,4,0]); glLightfv(GL_LIGHT0, GL_DIFFUSE, [.9,.9,.9,1])
    glLightfv(GL_LIGHT0, GL_AMBIENT, [.15,.15,.2,1]); glLightfv(GL_LIGHT0, GL_SPECULAR, [.6,.6,.6,1])
    glLightfv(GL_LIGHT1, GL_POSITION, [-3,-1,2,0]); glLightfv(GL_LIGHT1, GL_DIFFUSE, [.3,.3,.4,1])
    glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, [.4,.4,.4,1]); glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 30)
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); gluPerspective(45, WIDTH/HEIGHT, 0.1, 50)
    glMatrixMode(GL_MODELVIEW)

    rot_x, rot_y, zoom = 25.0, 40.0, -4.5
    resolution, isovalue, field_type = 30, 0.0, 0
    color_mode, lighting_on, wireframe = 1, True, False
    animate, auto_rotate, algorithm = False, True, 0
    field_names = ["Sphere", "Torus", "Metaballs", "Wave"]
    color_names = ["Solid", "Position", "Normal", "Height"]
    algo_names = ["Marching Cubes", "Dual Contouring"]
    dl_mesh, prim_count, vpp = None, 0, 3

    def rebuild(t=0.0):
        nonlocal dl_mesh, prim_count, vpp
        field = compute_field(field_type, resolution, t)
        if algorithm == 0:
            v, n, c = extract_mc(field, isovalue); vpp = 3
        else:
            v, n, c = extract_dc(field, isovalue); vpp = 4
        prim_count = c
        col = compute_colors(np.asarray(v, dtype=np.float32),
                             np.asarray(n, dtype=np.float32), color_mode)
        if dl_mesh is not None: glDeleteLists(dl_mesh, 1)
        dl_mesh = glGenLists(1); glNewList(dl_mesh, GL_COMPILE)
        glBegin(GL_TRIANGLES if vpp == 3 else GL_QUADS)
        for i in range(prim_count):
            for j in range(vpp):
                idx = (i * vpp + j) * 3
                glColor3f(col[idx], col[idx+1], col[idx+2])
                glNormal3f(n[idx], n[idx+1], n[idx+2])
                glVertex3f(v[idx], v[idx+1], v[idx+2])
        glEnd(); glEndList()

    rebuild()
    clock = pygame.time.Clock()
    dragging = False; last_m = (0, 0); anim_t = 0.0; need_rb = False; bt = 0.0
    running = True
    while running:
        dt = clock.tick(60) / 1000.0; fps = clock.get_fps()
        for ev in pygame.event.get():
            if ev.type == QUIT: running = False
            elif ev.type == KEYDOWN:
                k = ev.key
                if k == K_ESCAPE: running = False
                elif k == K_UP: isovalue += 0.02; need_rb = True
                elif k == K_DOWN: isovalue -= 0.02; need_rb = True
                elif k in (K_EQUALS, K_PLUS, K_KP_PLUS): resolution = min(80, resolution+5); need_rb = True
                elif k in (K_MINUS, K_KP_MINUS): resolution = max(8, resolution-5); need_rb = True
                elif k == K_1: field_type = 0; need_rb = True
                elif k == K_2: field_type = 1; need_rb = True
                elif k == K_3: field_type = 2; need_rb = True
                elif k == K_4: field_type = 3; need_rb = True
                elif k == K_c: color_mode = (color_mode+1) % 4; need_rb = True
                elif k == K_l: lighting_on = not lighting_on
                elif k == K_w: wireframe = not wireframe
                elif k == K_a: animate = not animate
                elif k == K_SPACE: auto_rotate = not auto_rotate
                elif k == K_r: need_rb = True
                elif k == K_t: algorithm = 1 - algorithm; need_rb = True
            elif ev.type == MOUSEBUTTONDOWN:
                if ev.button == 1: dragging = True; last_m = ev.pos
                elif ev.button == 4: zoom = min(-1.5, zoom+0.3)
                elif ev.button == 5: zoom = max(-15.0, zoom-0.3)
            elif ev.type == MOUSEBUTTONUP:
                if ev.button == 1: dragging = False
            elif ev.type == MOUSEMOTION and dragging:
                dx, dy = ev.pos[0]-last_m[0], ev.pos[1]-last_m[1]
                rot_y += dx*0.5; rot_x = max(-90, min(90, rot_x+dy*0.5)); last_m = ev.pos
            elif ev.type == MOUSEWHEEL:
                zoom = max(-15.0, min(-1.5, zoom+ev.y*0.3))

        if animate and field_type == 2: anim_t += dt; need_rb = True
        if need_rb:
            t0 = time.time(); rebuild(anim_t); bt = (time.time()-t0)*1000; need_rb = False
        if auto_rotate: rot_y += 20.0*dt

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity(); glTranslatef(0, 0, zoom)
        glRotatef(rot_x, 1, 0, 0); glRotatef(rot_y, 0, 1, 0)
        (glEnable if lighting_on else glDisable)(GL_LIGHTING)
        if wireframe: glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); glDisable(GL_CULL_FACE)
        else: glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glEnable(GL_CULL_FACE); glCullFace(GL_BACK)
        if dl_mesh is not None: glCallList(dl_mesh)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL); glDisable(GL_CULL_FACE)

        pl = "Tris" if algorithm == 0 else "Quads"
        ac = (100,200,255) if algorithm == 0 else (255,180,100)
        nc = "(NUMBA)" if USE_NUMBA else "(Python)"
        draw_hud(font, [
            (f"Algorithm: {algo_names[algorithm]} {nc}   (T to toggle)", ac),
            (f"FPS: {fps:.0f}   {pl}: {prim_count}   Build: {bt:.1f}ms", (200,220,255)),
            (f"Field: {field_names[field_type]}   Isovalue: {isovalue:.2f}   Resolution: {resolution}", (180,255,180)),
            (f"Color: {color_names[color_mode]}   Light: {'ON' if lighting_on else 'OFF'}   Wire: {'ON' if wireframe else 'OFF'}   Anim: {'ON' if animate else 'OFF'}   Spin: {'ON' if auto_rotate else 'OFF'}", (255,220,180)),
            ("", (255,255,255)),
            ("Drag:Rotate  Scroll:Zoom  Up/Down:Iso  +/-:Res  1-4:Field  C:Color  L:Light  W:Wire  A:Anim  Space:Spin  T:Algo  R:Rebuild", (140,140,160)),
        ])
        pygame.display.flip()
    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()