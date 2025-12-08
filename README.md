# P2M – Realtime Mesh Viewer (Orbbec + PCL + OpenGL)

## Overview
P2M is a standalone realtime RGB‑D mesh reconstruction viewer built with Orbbec SDK, PCL, and OpenGL. It captures frames from an Orbbec depth camera, reconstructs a mesh on the fly, and renders points/mesh via VBO/EBO. Image panels on the right can show the latest RGB and depth previews along with live controls/status text.

## Features (current)
- Realtime RGB‑D capture with frame sync when available; depth‑only fallback.
- Organized fast mesh triangulation with stride and edge/dz rejection.
- Two‑stage hole handling: small neighbor patching (toggle `H`) and plane‑based filling (toggle `F`, default OFF). Plane fill can detect multiple dominant planes.
- OpenGL viewer with toggles for points, mesh modes, auto‑rotate, camera reset, etc.
- Overlay panels (toggle `V`) showing RGB, depth, key hints, FPS/frame time, and the latest status message.
- Save current point cloud and mesh to PLY.

## Project Layout
```
p2m_cpu/
├── src/RealtimeMeshViewer.cpp   # main application
├── include/, lib/               # SDK/third-party headers & libs
├── build/                       # build directory (out-of-source)
│   └── bin/P2M_RealtimeMeshViewer.exe
├── CMakeLists.txt
└── README.md
```

## Building
Prerequisites: Visual Studio toolchain, CMake, Orbbec SDK, and bundled deps in `include/` + `lib/` (already provided here).

Build (Release target):
```bash
mkdir -p build
cd build
cmake ..
cmake --build . --config Release --target P2M_RealtimeMeshViewer
```
The executable is placed at `build/bin/P2M_RealtimeMeshViewer.exe`.

## Running
1) Connect an Orbbec RGB‑D camera and ensure USB drivers are installed.  
2) From `build/bin`, run `P2M_RealtimeMeshViewer.exe`.  
3) The viewer starts capture, reconstructs meshes, and opens an OpenGL window.

## Controls (keyboard/mouse)
- Mouse drag (L): orbit; mouse wheel: move speed.
- `W/S/A/D`: move; arrows up/down: move vertically; `Q/E`: roll.
- `P`: toggle points.  
- `M`: cycle mesh mode (fill / wireframe / off).  
- `V`: toggle RGB/Depth panels and overlay.  
- `F`: toggle plane-based hole fill (default OFF).  
- `H`: toggle small-hole patching.  
- `[` `]`: adjust mesh edge factor.  
- `-` `=`: adjust pixel stride.  
- `,` `.`: adjust hole patch perimeter.  
- `T`: auto-rotate; `R`: reset camera; `Space`: pause/resume capture; `C`: save PLYs; `ESC`: exit.

Status/FPS: Shown in the overlay (right side). Latest key actions also appear there and replace the previous message.

## Saving Data
Press `C` to write `captured_cloud.ply` and `captured_mesh.ply` (with timestamped names) to the working directory.

## Notes
- Plane fill is OFF by default; enable with `F` when you want dominant-plane guided hole filling.
- Panels are overlays; when `V` is off they are not drawn and the main view uses the full window.
- Performance printouts go to the console; per-frame overlay shows FPS and last frame time.

## License / Credits
Uses Orbbec SDK, PCL, OpenGL/GLEW/FreeGLUT, Boost, and OpenCV (DLLs provided alongside the build). License follows the respective third-party licenses and Orbbec SDK terms.
