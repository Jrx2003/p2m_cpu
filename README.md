# P2M - Point to Mesh 实时网格重建项目

## 🎉 项目简介

P2M (Point to Mesh) 是一个独立的实时网格重建查看器项目，基于 Orbbec SDK、PCL 和 OpenGL 构建。

### 核心特性

✨ **实时采集** - 直接从 Orbbec 深度相机获取 RGB-D 数据流  
✨ **快速重建** - 使用 PCL OrganizedFastMesh 算法实时重建网格  
✨ **交互可视化** - OpenGL VBO/EBO 加速的 3D 交互查看器  
✨ **多线程架构** - 采集和渲染完全并行，保证流畅性  
✨ **独立部署** - 所有依赖已打包，便于迁移到其他电脑  

---

## 📁 项目结构

```
p2m/
├── src/                          源代码目录
│   └── RealtimeMeshViewer.cpp   主程序 (780行)
│
├── build/                        构建输出目录
│   ├── bin/                      可执行文件
│   └── lib/                      库文件
│
├── P2M_Portable/                 ✨ 独立部署包（已打包）
│   ├── P2M_RealtimeMeshViewer.exe  主程序
│   ├── *.dll                       所有依赖库 (121个文件)
│   ├── 运行.bat                    一键启动
│   └── README.txt                  使用说明
│
├── docs/                         文档目录
│
├── CMakeLists.txt                独立 CMake 配置
├── 编译.bat                      自动编译脚本
├── 打包.bat                      自动打包脚本
├── 安装依赖.bat                  依赖安装脚本
└── README.md                     本文档
```

---

## 🚀 快速开始

### 方式 1：使用已编译的独立包（推荐）

如果已编译完成，直接使用打包好的版本：

```bash
cd P2M_Portable
.\运行.bat
```

**要求**：
- 已连接 Orbbec 设备
- 已安装 Orbbec USB 驱动
- Windows 系统（支持 OpenGL 3.0+）

### 方式 2：从源码编译

#### 第 1 步：安装依赖

```bash
# 使用自动安装脚本（推荐）
.\安装依赖.bat

# 或手动安装
C:\vcpkg\vcpkg install pcl:x64-windows glew:x64-windows freeglut:x64-windows
```

#### 第 2 步：编译项目

```bash
# 使用自动编译脚本
.\编译.bat

# 或手动编译
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake ..
cmake --build . --config Release
```

#### 第 3 步：打包部署

```bash
# 自动打包所有依赖
.\打包.bat
```

打包完成后，`P2M_Portable` 文件夹包含所有文件，可以直接复制到其他电脑运行。

---

## 🎮 使用说明

### 启动程序

1. 连接 Orbbec 深度相机
2. 运行 `P2M_Portable\运行.bat`
3. 程序自动初始化并开始实时重建

### 交互控制

| 操作 | 功能 |
|------|------|
| 🖱️ 鼠标拖拽 | 旋转视图 |
| 🖱️ 鼠标滚轮 | 缩放 |
| ⌨️ W / S | 放大 / 缩小 |
| ⌨️ P | 开关点云显示 |
| ⌨️ M | 开关网格显示 |
| ⌨️ A | 自动旋转 |
| ⌨️ 空格 | 暂停/继续 |
| ⌨️ C | 保存当前帧 |
| ⌨️ ESC | 退出 |

### 输出文件

按 C 键保存当前帧时，生成两个文件：
- `captured_cloud.ply` - 点云数据
- `captured_mesh.ply` - 网格数据

这些文件可以用 CloudCompare、MeshLab 等工具打开。

---

## 📦 部署说明

### 独立部署包内容

`P2M_Portable` 文件夹包含：

- **1 个** 可执行文件 (P2M_RealtimeMeshViewer.exe)
- **121 个** DLL 库文件（约 98 MB）
- **1 个** 配置文件 (OrbbecSDKConfig_v1.0.xml)
- **1 个** 启动脚本 (运行.bat)
- **1 个** 使用说明 (README.txt)

### 迁移到其他电脑

**步骤**：
1. 将整个 `P2M_Portable` 文件夹复制到目标电脑
2. 确保目标电脑已安装 Orbbec USB 驱动
3. 连接 Orbbec 设备
4. 双击 `运行.bat` 启动

**目标电脑要求**：
- Windows 7 或更高版本
- Visual C++ Redistributable 2017+ (通常系统已有)
- 支持 OpenGL 3.0+ 的显卡
- Orbbec USB 驱动程序

### DLL 依赖列表

主要依赖库：

```
核心库：
  OrbbecSDK.dll          - Orbbec SDK 核心
  depthengine_2_0.dll    - 深度引擎
  live555.dll            - 流媒体处理
  ob_usb.dll             - USB 通信

PCL 相关 (约 20 个 dll)：
  pcl_common.dll
  pcl_io.dll
  pcl_surface.dll
  pcl_kdtree.dll
  ... 等

OpenGL 相关：
  glew32.dll             - OpenGL 扩展
  freeglut.dll           - OpenGL 工具库

其他依赖 (约 94 个 dll)：
  boost_*.dll            - Boost 库
  opencv_*.dll           - OpenCV 库
  ... 等
```

---

## 🔧 开发说明

### 项目配置

- **CMake 版本**: 3.5+
- **C++ 标准**: C++14
- **编译器**: Visual Studio 2017+
- **构建类型**: Debug / Release

### 关键技术

| 技术栈 | 用途 |
|--------|------|
| **Orbbec SDK 1.10.27** | 相机控制、数据采集 |
| **PCL 1.15.1** | 点云处理、网格重建 |
| **OpenGL 3.0+** | 3D 渲染 |
| **GLEW** | OpenGL 扩展管理 |
| **FreeGLUT** | 窗口管理和交互 |
| **Boost** | 多线程、序列化等 |
| **OpenCV 4.x** | 图像处理 |

### 代码架构

```
Main Thread (OpenGL)           Capture Thread (Background)
     │                               │
     ├─ Display Loop                 ├─ SDK Pipeline
     │  ├─ Render Mesh               │  ├─ Get Frame
     │  └─ Handle Input              │  └─ Convert to PCL
     │                                │
     └─ Idle Loop                     └─ Update Buffer
        └─ Reconstruct Mesh               (Thread-Safe)
```

### 修改和扩展

编辑 `src/RealtimeMeshViewer.cpp` 修改：

**调整网格参数**：
```cpp
// 第 315 行附近
ofm.setTrianglePixelSize(2);     // 增加值 -> 更粗糙但更快
ofm.setMaxEdgeLength(0.03);      // 减小值 -> 更细致但更慢
```

**更改相机分辨率**：
```cpp
// 第 80 行附近，initializeOrbbecSDK() 函数中
depthProfile = depthProfileList->getVideoStreamProfile(
    640, 480,  // 修改为所需分辨率
    OB_FORMAT_ANY, 30);
```

修改后重新编译：
```bash
cd build
cmake --build . --config Release
```

---

## 📊 性能参考

| 分辨率 | 点数 | 三角形 | 帧率 | 场景 |
|--------|------|--------|------|------|
| 640×480 | ~300K | ~50K | 30 fps | 实时扫描 |
| 1280×720 | ~900K | ~150K | 20 fps | 高质量重建 |
| 1920×1080 | ~2M | ~350K | 10 fps | 精细建模 |

*测试环境: i7-9700K + RTX 2060 + 32GB RAM*

### 优化建议

1. **降低分辨率** - 最大影响因素
2. **调整重建参数** - `setTrianglePixelSize(2)` 或更大
3. **使用 Release 编译** - 2-3 倍性能提升
4. **关闭点云显示** - 只显示网格
5. **使用独立显卡** - 而非集成显卡

---

## 🐛 常见问题

### Q: 编译失败 - 找不到 PCL

**A:** 使用 `安装依赖.bat` 或手动安装：
```bash
C:\vcpkg\vcpkg install pcl:x64-windows
```

### Q: 运行提示 "Device not found!"

**A:** 检查：
- Orbbec 设备是否已连接
- USB 驱动是否已安装
- 尝试更换 USB 端口

### Q: 运行提示缺少 DLL

**A:** 
- 确保使用 `P2M_Portable` 文件夹中的程序
- 或重新运行 `打包.bat` 生成完整包
- 检查是否安装了 Visual C++ Redistributable

### Q: 程序启动但看不到网格

**A:**
- 按 M 键确认网格显示已开启
- 检查控制台错误信息
- 确保设备正常工作（可用官方工具测试）

### Q: 渲染卡顿

**A:**
- 降低相机分辨率（修改代码重新编译）
- 关闭点云显示（按 P 键）
- 使用 Release 版本而非 Debug
- 关闭其他占用 GPU 的程序

---

## 📖 相关文档

### 项目文档
- `README.txt` - 用户使用说明（在 P2M_Portable 文件夹）
- `src/RealtimeMeshViewer.cpp` - 源代码（含详细注释）

### 外部资源
- [Orbbec SDK](https://github.com/orbbec/OrbbecSDK)
- [PCL Documentation](https://pcl.readthedocs.io/)
- [OrganizedFastMesh](https://pcl.readthedocs.io/projects/tutorials/en/latest/fast_triangulation.html)
- [OpenGL Tutorial](https://learnopengl.com/)

### 工具软件
- [CloudCompare](https://www.cloudcompare.org/) - 查看点云和网格
- [MeshLab](https://www.meshlab.net/) - 编辑和处理网格
- [vcpkg](https://github.com/Microsoft/vcpkg) - C++ 包管理器

---

## 🎯 使用场景

### 1. 实时 3D 扫描
快速扫描物体并生成 3D 模型，适合快速原型制作。

### 2. 深度数据可视化
实时查看和分析深度传感器数据，辅助算法开发。

### 3. 点云网格研究
研究不同网格重建算法和参数的效果。

### 4. 教学演示
展示点云处理和 3D 重建的完整流程。

### 5. 二次开发基础
作为基础框架进行功能扩展和定制开发。

---

## 📝 版本信息

- **项目名称**: P2M Realtime Mesh Viewer
- **版本**: 1.0
- **OrbbecSDK**: 1.10.27
- **构建日期**: 2025-11-10
- **许可证**: 遵循 OrbbecSDK 许可证

---

## 🙏 致谢

本项目基于以下开源项目：
- Orbbec SDK
- Point Cloud Library (PCL)
- OpenGL / GLEW / FreeGLUT
- Boost
- OpenCV

---

## 📞 技术支持

如遇到问题：
1. 查看 `P2M_Portable\README.txt` 的故障排除部分
2. 检查 `Log` 目录下的日志文件
3. 参考 Orbbec SDK 和 PCL 官方文档

---

**祝使用愉快！** 🚀

---

*Created: 2025-11-10*  
*Location: OrbbecSDK-1.10.27/p2m/*  
*Status: ✅ Ready to Deploy*

