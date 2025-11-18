// RealtimeMeshViewer.cpp
// Real-time mesh reconstruction and visualization using Orbbec SDK + PCL + OpenGL

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>

// Orbbec SDK
#include "libobsensor/ObSensor.hpp"

// PCL
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/io/ply_io.h>

// OpenGL
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

// Forward declarations for globals used before definition
extern float g_edgeFactor;
extern int   g_pixStride;
extern float g_dzMax;

//==================== Real-time Mesh Reconstruction Class ====================

class RealtimeMeshReconstruction {
private:
    // Point cloud data
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PolygonMesh mesh;
    int width{}, height{};
    float maxDistance;
    
    // Thread safety (use recursive_mutex to allow nested locking)
    std::recursive_mutex cloudMutex;
    std::atomic<bool> hasNewData{false};
    std::atomic<bool> shouldStop{false};
    
    // Consistency buffers - keep mesh and cloud in sync
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_mesh; // cloud snapshot used to build 'mesh'
    uint64_t cloudStamp = 0;   // increments every time we update cloud from sensor
    uint64_t meshStamp  = 0;   // the cloudStamp used to produce current 'mesh'
    
    // Orbbec SDK
    std::unique_ptr<ob::Pipeline> pipeline;
    std::unique_ptr<ob::PointCloudFilter> pointCloudFilter;
    std::thread captureThread;
    bool hasColorSensor;

public:
    RealtimeMeshReconstruction() {
        cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        cloud_for_mesh.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        maxDistance = 0.05f;  // 50mm default
        width = 0;
        height = 0;
        hasColorSensor = false;
    }

    ~RealtimeMeshReconstruction() {
        stopCapture();
    }

    bool initializeOrbbecSDK() {
        std::cout << "Initializing Orbbec SDK..." << std::endl;
        
        try {
            // Create pipeline
            pipeline = std::make_unique<ob::Pipeline>();
            
            // Configure streams
            auto config = std::make_shared<ob::Config>();
            
            // Try to enable color stream with specific resolution (1280x720 @ 15fps)
            std::shared_ptr<ob::VideoStreamProfile> colorProfile = nullptr;
            try {
                auto colorProfiles = pipeline->getStreamProfileList(OB_SENSOR_COLOR);
                if(colorProfiles && colorProfiles->count() > 0) {
                    // Try to find 1280x720 profile - prefer 15fps
                    std::shared_ptr<ob::StreamProfile> targetProfile = nullptr;
                    try {
                        targetProfile = colorProfiles->getVideoStreamProfile(1280, 720, OB_FORMAT_ANY, 15);
                        std::cout << "Found 1280x720@15fps color profile" << std::endl;
                    }
                    catch(...) {
                        try {
                            targetProfile = colorProfiles->getVideoStreamProfile(1280, 720, OB_FORMAT_ANY, 30);
                            std::cout << "Found 1280x720@30fps color profile (fallback)" << std::endl;
                        }
                        catch(...) {
                            targetProfile = colorProfiles->getProfile(OB_PROFILE_DEFAULT);
                            std::cout << "Using default color profile" << std::endl;
                        }
                    }
                    
                    colorProfile = targetProfile->as<ob::VideoStreamProfile>();
                    std::cout << "Color stream: " << colorProfile->width() << "x" << colorProfile->height() 
                              << " @" << colorProfile->fps() << "fps" << std::endl;
                    config->enableStream(colorProfile);
                    hasColorSensor = true;
                }
            }
            catch(ob::Error &e) {
                std::cerr << "Color sensor not available: " << e.getMessage() << std::endl;
                hasColorSensor = false;
            }
            
            // Configure depth stream with alignment
            std::shared_ptr<ob::StreamProfileList> depthProfileList;
            OBAlignMode alignMode = ALIGN_DISABLE;
            
            if(colorProfile) {
                // Try hardware alignment first
                depthProfileList = pipeline->getD2CDepthProfileList(colorProfile, ALIGN_D2C_HW_MODE);
                if(depthProfileList->count() > 0) {
                    alignMode = ALIGN_D2C_HW_MODE;
                    std::cout << "Using hardware D2C alignment" << std::endl;
                }
                else {
                    // Try software alignment
                    depthProfileList = pipeline->getD2CDepthProfileList(colorProfile, ALIGN_D2C_SW_MODE);
                    if(depthProfileList->count() > 0) {
                        alignMode = ALIGN_D2C_SW_MODE;
                        std::cout << "Using software D2C alignment" << std::endl;
                    }
                }
                
                try {
                    pipeline->enableFrameSync();
                    std::cout << "Frame synchronization enabled" << std::endl;
                }
                catch(ob::Error &e) {
                    std::cerr << "Frame sync not supported: " << e.getMessage() << std::endl;
                }
            }
            else {
                depthProfileList = pipeline->getStreamProfileList(OB_SENSOR_DEPTH);
            }
            
            if(depthProfileList->count() > 0) {
                std::shared_ptr<ob::StreamProfile> depthProfile;
                try {
                    if(colorProfile) {
                        depthProfile = depthProfileList->getVideoStreamProfile(
                            OB_WIDTH_ANY, OB_HEIGHT_ANY, OB_FORMAT_ANY, colorProfile->fps());
                    }
                }
                catch(...) {
                    depthProfile = nullptr;
                }
                
                if(!depthProfile) {
                    depthProfile = depthProfileList->getProfile(OB_PROFILE_DEFAULT);
                }
                
                config->enableStream(depthProfile);
                
                // Get resolution - use color resolution if available for RGB-D point cloud
                if(colorProfile) {
                    width = colorProfile->width();
                    height = colorProfile->height();
                    std::cout << "Point cloud resolution (from color): " << width << "x" << height << std::endl;
                }
                else {
                    auto videoProfile = depthProfile->as<ob::VideoStreamProfile>();
                    width = videoProfile->width();
                    height = videoProfile->height();
                    std::cout << "Point cloud resolution (from depth): " << width << "x" << height << std::endl;
                }
            }
            
            config->setAlignMode(alignMode);
            
            // Start pipeline
            pipeline->start(config);
            
            // Create point cloud filter
            pointCloudFilter = std::make_unique<ob::PointCloudFilter>();
            auto cameraParam = pipeline->getCameraParam();
            pointCloudFilter->setCameraParam(cameraParam);
            
            // Use RGB point cloud if color sensor is available, otherwise use depth only
            if(hasColorSensor) {
                pointCloudFilter->setCreatePointFormat(OB_FORMAT_RGB_POINT);
                std::cout << "Using RGB point cloud mode" << std::endl;
            }
            else {
                pointCloudFilter->setCreatePointFormat(OB_FORMAT_POINT);
                std::cout << "Using depth-only point cloud mode (no color)" << std::endl;
            }
            
            std::cout << "Orbbec SDK initialized successfully!" << std::endl;
            return true;
        }
        catch(ob::Error &e) {
            std::cerr << "Failed to initialize Orbbec SDK: " << e.getMessage() << std::endl;
            return false;
        }
    }

    void startCapture() {
        std::cout << "Starting capture thread..." << std::endl;
        shouldStop = false;
        captureThread = std::thread(&RealtimeMeshReconstruction::captureLoop, this);
    }

    void stopCapture() {
        if(captureThread.joinable()) {
            std::cout << "Stopping capture thread..." << std::endl;
            shouldStop = true;
            captureThread.join();
        }
        if(pipeline) {
            pipeline->stop();
        }
    }

    void captureLoop() {
        std::cout << "Capture loop started" << std::endl;
        std::cout << "Mode: " << (hasColorSensor ? "RGB-D" : "Depth-only") << std::endl;
        
        int consecutiveErrors = 0;
        const int maxConsecutiveErrors = 10;
        int frameCount = 0;
        int skippedFrames = 0;
        
        // Give sensors time to warm up
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        while(!shouldStop) {
            try {
                // Increase timeout to 1000ms for first few frames
                int timeout = (frameCount < 5) ? 1000 : 500;
                auto frameset = pipeline->waitForFrames(timeout);
                
                if(!frameset) {
                    if(frameCount < 5) {
                        std::cout << "No frameset received (attempt " << frameCount << ")" << std::endl;
                    }
                    continue;
                }
                
                if(!frameset->depthFrame()) {
                    skippedFrames++;
                    continue;
                }
                
                // Check if we need color frame but don't have it
                if(hasColorSensor && !frameset->colorFrame()) {
                    skippedFrames++;
                    if(skippedFrames % 30 == 1) {  // Print every 30 skipped frames
                        std::cout << "Waiting for synchronized color frame... (skipped: " << skippedFrames << ")" << std::endl;
                    }
                    continue;
                }
                
                // Get depth value scale
                auto depthValueScale = frameset->depthFrame()->getValueScale();
                pointCloudFilter->setPositionDataScaled(depthValueScale);
                
                // Generate point cloud
                auto frame = pointCloudFilter->process(frameset);
                if(!frame) {
                    std::cerr << "Point cloud filter returned null frame!" << std::endl;
                    continue;
                }
                
                // Convert to PCL point cloud
                updatePointCloudFromFrame(frame);
                
                // Reset error counter on success
                consecutiveErrors = 0;
                frameCount++;
                
                if(frameCount == 1) {
                    std::cout << "✓ First point cloud generated successfully!" << std::endl;
                    std::cout << "  Skipped " << skippedFrames << " incomplete frames" << std::endl;
                }
                else if(frameCount % 100 == 0) {
                    std::cout << "Processed " << frameCount << " frames (skipped: " << skippedFrames << ")" << std::endl;
                }
                
            }
            catch(ob::Error &e) {
                consecutiveErrors++;
                if(consecutiveErrors < maxConsecutiveErrors) {
                    std::cerr << "Capture error (" << consecutiveErrors << "): " << e.getMessage() << std::endl;
                }
                else if(consecutiveErrors == maxConsecutiveErrors) {
                    std::cerr << "Too many consecutive errors, suppressing further messages..." << std::endl;
                }
                // Continue trying
            }
            catch(std::exception &e) {
                consecutiveErrors++;
                if(consecutiveErrors < maxConsecutiveErrors) {
                    std::cerr << "Capture error (" << consecutiveErrors << "): " << e.what() << std::endl;
                }
                // Continue trying
            }
        }
        
        std::cout << "Capture loop ended. Total frames: " << frameCount << std::endl;
    }

    void updatePointCloudFromFrame(std::shared_ptr<ob::Frame> frame) {
        static bool firstUpdate = true;
        
        // Create new point cloud
        auto newCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>);
        
        newCloud->width = width;
        newCloud->height = height;
        newCloud->is_dense = false;
        newCloud->points.resize(width * height);
        // Initialize all points to NaN to avoid (0,0,0) defaults being used
        for(auto &p : newCloud->points) {
            p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
            p.r = p.g = p.b = 0;
        }
        
        static const auto min_distance = 1e-6;
        
        if(firstUpdate) {
            std::cout << "Processing " << (hasColorSensor ? "RGB-D" : "depth-only") 
                      << " point cloud (" << width << "x" << height << ")" << std::endl;
        }
        
        if(hasColorSensor) {
            // Process RGB point cloud
            int pointsSize = frame->dataSize() / sizeof(OBColorPoint);
            OBColorPoint *points = (OBColorPoint *)frame->data();
            if(pointsSize != width * height) {
                std::cout << "[Warn] RGB pointsSize=" << pointsSize
                          << " != width*height=" << (width * height) << std::endl;
            }
            
            for(int i = 0; i < pointsSize && i < (width * height); i++) {
                pcl::PointXYZRGB &p = newCloud->points[i];
                
                // Convert from mm to meters
                p.x = points[i].x / 1000.0f;
                p.y = points[i].y / 1000.0f;
                p.z = points[i].z / 1000.0f;
                p.r = points[i].r;
                p.g = points[i].g;
                p.b = points[i].b;
                
                // Mark invalid points as NaN
                if(fabs(points[i].x) < min_distance && 
                   fabs(points[i].y) < min_distance && 
                   fabs(points[i].z) < min_distance) {
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        else {
            // Process depth-only point cloud
            int pointsSize = frame->dataSize() / sizeof(OBPoint);
            OBPoint *points = (OBPoint *)frame->data();
            if(pointsSize != width * height) {
                std::cout << "[Warn] depth pointsSize=" << pointsSize
                          << " != width*height=" << (width * height) << std::endl;
            }
            
            for(int i = 0; i < pointsSize && i < (width * height); i++) {
                pcl::PointXYZRGB &p = newCloud->points[i];
                
                // Convert from mm to meters
                p.x = points[i].x / 1000.0f;
                p.y = points[i].y / 1000.0f;
                p.z = points[i].z / 1000.0f;
                
                // Use grayscale for depth-only mode (map depth to color)
                float depth = p.z * 1000.0f;  // Back to mm for visualization
                uint8_t gray = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, 255.0f * (1.0f - depth / 3000.0f))));
                p.r = gray;
                p.g = gray;
                p.b = gray;
                
                // Mark invalid points as NaN
                if(fabs(points[i].x) < min_distance && 
                   fabs(points[i].y) < min_distance && 
                   fabs(points[i].z) < min_distance) {
                    p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        
        // Count valid points
        int validPoints = 0;
        for(const auto& p : newCloud->points) {
            if(!std::isnan(p.x)) validPoints++;
        }
        
        // Thread-safe update
        {
            std::lock_guard<std::recursive_mutex> lock(cloudMutex);
            cloud = newCloud;
            ++cloudStamp;  // Increment frame stamp
            hasNewData = true;
        }
        
        if(firstUpdate) {
            std::cout << "✓ Point cloud updated! Valid points: " << validPoints 
                      << " / " << (width * height) << std::endl;
            firstUpdate = false;
        }
    }

    bool hasNewDataAvailable() {
        return hasNewData.exchange(false);
    }

    float estimateNeighborSpacing() {
        if(width <= 1 || height <= 1) return 0.0f;

        std::vector<float> dists;
        dists.reserve(10000);

        int stepRow = std::max(1, height / 200);
        int stepCol = std::max(1, width  / 200);

        auto dist = [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) -> float {
            if(!pcl::isFinite(a) || !pcl::isFinite(b)) return -1.0f;
            float dx = a.x - b.x;
            float dy = a.y - b.y;
            float dz = a.z - b.z;
            return std::sqrt(dx*dx + dy*dy + dz*dz);
        };

        std::lock_guard<std::recursive_mutex> lock(cloudMutex);
        
        for(int r = 0; r < height - 1; r += stepRow) {
            for(int c = 0; c < width - 1; c += stepCol) {
                int idx  = r * width + c;
                int idxR = idx + 1;
                int idxD = idx + width;

                if(idxR < static_cast<int>(cloud->points.size())) {
                    float d = dist(cloud->points[idx], cloud->points[idxR]);
                    if(d > 0.0f && d < 0.2f) dists.push_back(d);
                }
                if(idxD < static_cast<int>(cloud->points.size())) {
                    float d = dist(cloud->points[idx], cloud->points[idxD]);
                    if(d > 0.0f && d < 0.2f) dists.push_back(d);
                }
                if(dists.size() >= 10000) break;
            }
            if(dists.size() >= 10000) break;
        }

        if(dists.empty()) return 0.0f;

        std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
        return dists[dists.size() / 2];
    }

    void reconstructMesh() {
        // Take snapshot of cloud + corresponding frame stamp
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr used(new pcl::PointCloud<pcl::PointXYZRGB>);
        uint64_t stamp_at_start = 0;
        {
            std::lock_guard<std::recursive_mutex> lock(cloudMutex);
            *used = *cloud;  // Deep copy
            stamp_at_start = cloudStamp;
        }

        static bool firstReconstruct = true;
        static int reconstructCount = 0;

        const int W = used->width, H = used->height;
        if(!used || W == 0 || H == 0 || static_cast<size_t>(W * H) != used->points.size()) {
            std::cerr << "[Mesh] Cloud not organized, w=" << W
                      << " h=" << H << " n=" << used->points.size() << std::endl;
            return;
        }

        // Count valid points
        int validPoints = 0;
        for(const auto& p : used->points) {
            if(pcl::isFinite(p)) validPoints++;
        }
        
        if(firstReconstruct) {
            std::cout << "[Mesh] Cloud dimensions: " << W << "x" << H << std::endl;
            std::cout << "[Mesh] Valid points: " << validPoints << " / " << used->points.size() 
                      << " (" << (100.0f * validPoints / used->points.size()) << "%)" << std::endl;
        }

        // Estimate spacing from the snapshot (compute locally to avoid lock)
        float spacing = 0.0f;
        if(W > 1 && H > 1) {
            std::vector<float> dists;
            dists.reserve(10000);
            int stepRow = std::max(1, H / 200);
            int stepCol = std::max(1, W / 200);
            auto dist = [](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b) -> float {
                if(!pcl::isFinite(a) || !pcl::isFinite(b)) return -1.0f;
                float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
                return std::sqrt(dx*dx + dy*dy + dz*dz);
            };
            for(int r = 0; r < H - 1; r += stepRow) {
                for(int c = 0; c < W - 1; c += stepCol) {
                    int idx = r * W + c;
                    int idxR = idx + 1;
                    int idxD = idx + W;
                    if(idxR < static_cast<int>(used->points.size())) {
                        float d = dist(used->points[idx], used->points[idxR]);
                        if(d > 0.0f && d < 0.2f) dists.push_back(d);
                    }
                    if(idxD < static_cast<int>(used->points.size())) {
                        float d = dist(used->points[idx], used->points[idxD]);
                        if(d > 0.0f && d < 0.2f) dists.push_back(d);
                    }
                    if(dists.size() >= 10000) break;
                }
                if(dists.size() >= 10000) break;
            }
            if(!dists.empty()) {
                std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
                spacing = dists[dists.size() / 2];
            }
        }
        if(spacing <= 0.0f) spacing = 0.01f;

        // Manual grid triangulation with stride
        const int s = std::max(1, g_pixStride);

        // For neighbors after stride, actual adjacent distance is approximately spacing * s
        float strideSpacing = spacing * static_cast<float>(s);

        // Maximum allowed edge length: scaled by g_edgeFactor for stride
        maxDistance = strideSpacing * g_edgeFactor;
        const float maxDist2 = maxDistance * maxDistance;

        // Helpers
        auto sane = [&](int i)->bool {
            const auto &p = used->points[i];
            if(!pcl::isFinite(p)) return false;
            if(std::fabs(p.x) > 10.f || std::fabs(p.y) > 10.f || p.z < 0.05f || p.z > 10.f) return false;
            return true;
        };
        auto edge_ok = [&](int a, int b)->bool {
            const auto &pa = used->points[a], &pb = used->points[b];
            float dx = pa.x - pb.x, dy = pa.y - pb.y, dz = pa.z - pb.z;
            if(dx*dx + dy*dy + dz*dz > maxDist2) return false;
            if(std::fabs(dz) > g_dzMax) return false;  // This continues to guard against "spikes"
            return true;
        };
        std::vector<pcl::Vertices> tris;
        tris.reserve(((W - s) * (H - s)) * 2 / std::max(1, s * s) + 1024);

        for(int r = 0; r <= H - 1 - s; r += s) {
            for(int c = 0; c <= W - 1 - s; c += s) {
                int i00 = r * W + c;
                int i10 = i00 + s;
                int i01 = i00 + s * W;
                int i11 = i01 + s;

                bool ok00 = sane(i00), ok10 = sane(i10), ok01 = sane(i01), ok11 = sane(i11);
                if(!(ok00 | ok10 | ok01 | ok11)) continue;

                if(ok00 && ok10 && ok11 && edge_ok(i00,i10) && edge_ok(i10,i11) && edge_ok(i11,i00)) {
                    pcl::Vertices v;
                    v.vertices.resize(3);
                    v.vertices[0] = static_cast<uint32_t>(i00);
                    v.vertices[1] = static_cast<uint32_t>(i10);
                    v.vertices[2] = static_cast<uint32_t>(i11);
                    tris.push_back(std::move(v));
                }
                if(ok00 && ok11 && ok01 && edge_ok(i00,i11) && edge_ok(i11,i01) && edge_ok(i01,i00)) {
                    pcl::Vertices v;
                    v.vertices.resize(3);
                    v.vertices[0] = static_cast<uint32_t>(i00);
                    v.vertices[1] = static_cast<uint32_t>(i11);
                    v.vertices[2] = static_cast<uint32_t>(i01);
                    tris.push_back(std::move(v));
                }
            }
        }

        // Monitoring: track maximum kept edge length
        float max_kept = 0.f;
        const auto &pts = used->points;
        auto L = [&](int a, int b)->float{
            const auto &pa = pts[a], &pb = pts[b];
            float dx=pa.x-pb.x, dy=pa.y-pb.y, dz=pa.z-pb.z;
            return std::sqrt(dx*dx+dy*dy+dz*dz);
        };
        for(const auto &t : tris){
            int i0=t.vertices[0], i1=t.vertices[1], i2=t.vertices[2];
            max_kept = std::max({max_kept, L(i0,i1), L(i1,i2), L(i2,i0)});
        }

        // Publish mesh + cloud_for_mesh atomically (keep them in sync)
        {
            std::lock_guard<std::recursive_mutex> lock(cloudMutex);
            mesh.polygons.swap(tris);
            *cloud_for_mesh = *used;  // Save the cloud snapshot used to build this mesh
            meshStamp = stamp_at_start;
        }

        reconstructCount++;
        std::cout << "[Mesh] #" << reconstructCount
                  << " stride=" << s
                  << " spacing=" << spacing
                  << " maxEdge=" << maxDistance
                  << " tris=" << mesh.polygons.size()
                  << " maxEdgeKept=" << max_kept << std::endl;

        firstReconstruct = false;
    }

    void saveMesh(const std::string& filename) {
        std::lock_guard<std::recursive_mutex> lock(cloudMutex);
        std::cout << "Saving mesh to: " << filename << std::endl;
        pcl::io::savePLYFile(filename, mesh);
        std::cout << "Mesh saved with " << mesh.polygons.size() << " triangles" << std::endl;
    }

    void savePointCloud(const std::string& filename) {
        std::lock_guard<std::recursive_mutex> lock(cloudMutex);
        std::cout << "Saving point cloud to: " << filename << std::endl;
        pcl::io::savePLYFile(filename, *cloud);
        std::cout << "Point cloud saved with " << cloud->points.size() << " points" << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloudCopy() {
        std::lock_guard<std::recursive_mutex> lock(cloudMutex);
        return pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>(*cloud));
    }

    void getMeshSnapshot(pcl::PointCloud<pcl::PointXYZRGB>::Ptr &outCloud,
                         pcl::PolygonMesh &outMesh,
                         uint64_t &outStamp) {
        std::lock_guard<std::recursive_mutex> lock(cloudMutex);
        outCloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(
            new pcl::PointCloud<pcl::PointXYZRGB>(*cloud_for_mesh)); // Deep copy
        outMesh = mesh;     // Copy
        outStamp = meshStamp;
    }

    const pcl::PolygonMesh& getMesh() const { return mesh; }
    int getWidth() const { return width; }
    int getHeight() const { return height; }
};

//==================== OpenGL Viewer with VBO ====================

// Mesh display modes
enum MeshDisplayMode {
    MESH_OFF = 0,
    MESH_FILL = 1,
    MESH_WIREFRAME = 2
};

RealtimeMeshReconstruction* g_app = nullptr;

int   g_winW = 1280;
int   g_winH = 720;
float g_angleX = 20.0f;    // Original initial view angle
float g_angleY = -30.0f;   // Original initial view angle
float g_distance = 2.5f;  // Original distance
int   g_lastX = 0;
int   g_lastY = 0;
bool  g_leftDown = false;
bool  g_drawPoints = true;   // Start with points visible for debugging
MeshDisplayMode g_meshMode = MESH_FILL;  // Start with filled mesh
bool  g_autoRotate = false;
bool  g_paused = false;

float g_centerX = 0.0f, g_centerY = 0.0f, g_centerZ = 0.5f;  // Original center
float g_sceneScale = 1.0f;

// VBO / EBO handles
GLuint g_vboPos = 0;
GLuint g_vboCol = 0;
GLuint g_eboMesh = 0;
GLsizei g_pointCount = 0;
GLsizei g_meshIndexCount = 0;

// Mesh edge length factor (controls max triangle edge length = spacing * factor)
float g_edgeFactor = 30.0f;

// Grid triangulation controls
int   g_pixStride = 4;      // pixel stride for manual grid triangulation
float g_dzMax     = 0.08f;  // allowed depth jump between adjacent vertices (meters)

// Performance tracking
int g_frameCount = 0;
int g_lastTime = 0;
float g_fps = 0.0f;

void cleanupGLBuffers() {
    if(g_vboPos) { glDeleteBuffers(1, &g_vboPos); g_vboPos = 0; }
    if(g_vboCol) { glDeleteBuffers(1, &g_vboCol); g_vboCol = 0; }
    if(g_eboMesh) { glDeleteBuffers(1, &g_eboMesh); g_eboMesh = 0; }
    g_pointCount = 0;
    g_meshIndexCount = 0;
}

void updateGLBuffers() {
    if(!g_app) return;

    cleanupGLBuffers();

    // Get consistent snapshot (cloud and mesh from the same frame)
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud;
    pcl::PolygonMesh mesh;
    uint64_t stamp = 0;
    g_app->getMeshSnapshot(cloud, mesh, stamp);

    g_pointCount = static_cast<GLsizei>(cloud->points.size());
    
    static bool firstUpdate = true;
    if(firstUpdate) {
        std::cout << "[OpenGL] First buffer update: " << g_pointCount << " points" << std::endl;
        firstUpdate = false;
    }

    // Vertex position VBO
    glGenBuffers(1, &g_vboPos);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboPos);
    glBufferData(GL_ARRAY_BUFFER, g_pointCount * 3 * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
    {
        float* pos = static_cast<float*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if(pos) {
            for(size_t i = 0; i < cloud->points.size(); ++i) {
                const auto& p = cloud->points[i];
                if(pcl::isFinite(p)) {
                    pos[3*i + 0] = p.x;
                    pos[3*i + 1] = p.y;
                    pos[3*i + 2] = p.z;
                } else {
                    // Push invalid points far away to avoid pulling triangles to origin
                    pos[3*i + 0] = 1e6f;
                    pos[3*i + 1] = 1e6f;
                    pos[3*i + 2] = 1e6f;
                }
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Color VBO
    glGenBuffers(1, &g_vboCol);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboCol);
    glBufferData(GL_ARRAY_BUFFER, g_pointCount * 3 * sizeof(GLubyte), nullptr, GL_DYNAMIC_DRAW);
    {
        GLubyte* col = static_cast<GLubyte*>(glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY));
        if(col) {
            for(size_t i = 0; i < cloud->points.size(); ++i) {
                const auto& p = cloud->points[i];
                col[3*i + 0] = p.r;
                col[3*i + 1] = p.g;
                col[3*i + 2] = p.b;
            }
            glUnmapBuffer(GL_ARRAY_BUFFER);
        }
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Mesh indices EBO with invalid-triangle filtering + safety check
    std::vector<GLuint> indices;
    indices.reserve(mesh.polygons.size() * 3);
    const auto &pts = cloud->points;
    
    // Estimate maxDistance for safety check (consistent with reconstruction)
    float estimatedSpacing = 0.01f;  // Default spacing estimate
    if(cloud->width > 1 && cloud->height > 1) {
        // Quick estimate: sample a few neighbor distances
        int sampleCount = 0;
        float sumDist = 0.0f;
        for(int r = 0; r < std::min(100, static_cast<int>(cloud->height) - 1) && sampleCount < 1000; r += 10) {
            for(int c = 0; c < std::min(100, static_cast<int>(cloud->width) - 1) && sampleCount < 1000; c += 10) {
                int idx = r * cloud->width + c;
                int idxR = idx + 1;
                if(idxR < static_cast<int>(pts.size()) && pcl::isFinite(pts[idx]) && pcl::isFinite(pts[idxR])) {
                    float dx = pts[idx].x - pts[idxR].x;
                    float dy = pts[idx].y - pts[idxR].y;
                    float dz = pts[idx].z - pts[idxR].z;
                    float d = std::sqrt(dx*dx + dy*dy + dz*dz);
                    if(d > 0.0f && d < 0.2f) {
                        sumDist += d;
                        sampleCount++;
                    }
                }
            }
        }
        if(sampleCount > 0) {
            estimatedSpacing = sumDist / sampleCount;
        }
    }
    float maxDist_render = estimatedSpacing * g_edgeFactor;
    const float maxDist2_render = maxDist_render * maxDist_render;
    
    auto len2 = [&](GLuint a, GLuint b)->float {
        const auto &pa = pts[a], &pb = pts[b];
        float dx=pa.x-pb.x, dy=pa.y-pb.y, dz=pa.z-pb.z;
        return dx*dx+dy*dy+dz*dz;
    };
    
    for(const auto& poly : mesh.polygons) {
        if(poly.vertices.size() != 3) continue;
        GLuint i0 = poly.vertices[0];
        GLuint i1 = poly.vertices[1];
        GLuint i2 = poly.vertices[2];
        if(i0 >= pts.size() || i1 >= pts.size() || i2 >= pts.size()) continue;
        if(!pcl::isFinite(pts[i0]) || !pcl::isFinite(pts[i1]) || !pcl::isFinite(pts[i2])) continue;
        
        // Safety check: filter triangles with edges that are too long or have excessive z-jump
        float l01 = len2(i0, i1), l12 = len2(i1, i2), l20 = len2(i2, i0);
        if(l01 > maxDist2_render || l12 > maxDist2_render || l20 > maxDist2_render) continue;
        if(std::fabs(pts[i0].z - pts[i1].z) > g_dzMax ||
           std::fabs(pts[i1].z - pts[i2].z) > g_dzMax ||
           std::fabs(pts[i2].z - pts[i0].z) > g_dzMax) continue;
        
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }
    g_meshIndexCount = static_cast<GLsizei>(indices.size());
    
    static bool firstMeshUpdate = true;
    if(firstMeshUpdate && g_meshIndexCount > 0) {
        std::cout << "[OpenGL] ✓ Mesh buffers created! Triangles: " << (g_meshIndexCount/3) << std::endl;
        firstMeshUpdate = false;
    }
    
    if(g_meshIndexCount > 0) {
        glGenBuffers(1, &g_eboMesh);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_eboMesh);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(GLuint),
                     indices.data(), GL_DYNAMIC_DRAW);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
}

void idleFunc() {
    static int updateCount = 0;
    static int lastTime = 0;
    static int frameIndex = 0;

    // Limit to ~10 FPS (100 ms per frame)
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    int elapsed = currentTime - lastTime;
    if(elapsed < 100) {
        return;
    }

    int frameStart = glutGet(GLUT_ELAPSED_TIME);

    int meshTime = 0;
    int bufferTime = 0;
    int totalTime = 0;

    if(!g_paused && g_app && g_app->hasNewDataAvailable()) {
        int t0 = glutGet(GLUT_ELAPSED_TIME);
        g_app->reconstructMesh();
        int t1 = glutGet(GLUT_ELAPSED_TIME);
        updateGLBuffers();
        int t2 = glutGet(GLUT_ELAPSED_TIME);

        meshTime = t1 - t0;
        bufferTime = t2 - t1;
        totalTime = t2 - frameStart;

        updateCount++;
        frameIndex++;
        if(updateCount == 1) {
            std::cout << "[OpenGL] First data received and buffers updated!" << std::endl;
        }
        // Print per-frame processing time
        std::cout << "[Perf] frame " << frameIndex
                  << " | mesh: " << meshTime << " ms"
                  << " | buffer: " << bufferTime << " ms"
                  << " | total: " << totalTime << " ms"
                  << std::endl;
    }

    lastTime = currentTime;
    
    if(g_autoRotate) {
        g_angleY += 0.5f;
    }
    
    glutPostRedisplay();
}

void displayFunc() {
    static int frameNum = 0;
    frameNum++;
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (g_winW > 0 && g_winH > 0) ? static_cast<double>(g_winW)/g_winH : 16.0/9.0;
    gluPerspective(45.0, aspect, 0.01, 100.0);  // 45° FOV for better view

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0f, 0.0f, -g_distance);
    glRotatef(g_angleX, 1.0f, 0.0f, 0.0f);
    glRotatef(g_angleY, 0.0f, 1.0f, 0.0f);
    glScalef(g_sceneScale, g_sceneScale, g_sceneScale);
    glTranslatef(-g_centerX, -g_centerY, -g_centerZ);

    glEnable(GL_DEPTH_TEST);
    
    if(frameNum == 100 || frameNum == 500) {
        std::cout << "[OpenGL] Frame " << frameNum << ": pointCount=" << g_pointCount 
                  << ", meshIndices=" << g_meshIndexCount << std::endl;
    }

    if(g_app && g_pointCount > 0) {
        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, g_vboPos);
        glVertexPointer(3, GL_FLOAT, 0, reinterpret_cast<void*>(0));

        glBindBuffer(GL_ARRAY_BUFFER, g_vboCol);
        glColorPointer(3, GL_UNSIGNED_BYTE, 0, reinterpret_cast<void*>(0));

        // Draw mesh with different modes
        if(g_meshMode != MESH_OFF && g_meshIndexCount > 0) {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_eboMesh);
            
            if(g_meshMode == MESH_FILL) {
                // Filled mesh
                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
                glDrawElements(GL_TRIANGLES, g_meshIndexCount, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
            }
            else if(g_meshMode == MESH_WIREFRAME) {
                // Wireframe mesh (网格)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
                glLineWidth(1.0f);
                glDrawElements(GL_TRIANGLES, g_meshIndexCount, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
            }
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        }

        // Draw points
        if(g_drawPoints) {
            glPointSize(2.0f);
            glDrawArrays(GL_POINTS, 0, g_pointCount);
        }

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glDisableClientState(GL_VERTEX_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
    } else {
        // Draw reference axes when no point cloud data (so user knows OpenGL is working)
        glLineWidth(3.0f);
        glBegin(GL_LINES);
        // X axis - Red
        glColor3f(1.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.5f, 0.0f, 0.0f);
        // Y axis - Green
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.5f, 0.0f);
        // Z axis - Blue
        glColor3f(0.0f, 0.0f, 1.0f);
        glVertex3f(0.0f, 0.0f, 0.0f);
        glVertex3f(0.0f, 0.0f, 0.5f);
        glEnd();
    }

    // FPS display
    g_frameCount++;
    int currentTime = glutGet(GLUT_ELAPSED_TIME);
    if(currentTime - g_lastTime > 1000) {
        g_fps = g_frameCount * 1000.0f / (currentTime - g_lastTime);
        g_frameCount = 0;
        g_lastTime = currentTime;
    }

    glutSwapBuffers();
}

void reshapeFunc(int w, int h) {
    g_winW = std::max(1, w);
    g_winH = std::max(1, h);
    glViewport(0, 0, g_winW, g_winH);
}

void mouseFunc(int button, int state, int x, int y) {
    if(button == GLUT_LEFT_BUTTON) {
        if(state == GLUT_DOWN) {
            g_leftDown = true;
            g_lastX = x;
            g_lastY = y;
        } else {
            g_leftDown = false;
        }
    }
    // Mouse wheel
    if(button == 3 && state == GLUT_DOWN) {
        g_distance = std::max(0.1f, g_distance - 0.1f);
        glutPostRedisplay();
    }
    if(button == 4 && state == GLUT_DOWN) {
        g_distance += 0.1f;
        glutPostRedisplay();
    }
}

void motionFunc(int x, int y) {
    if(!g_leftDown) return;
    int dx = x - g_lastX;
    int dy = y - g_lastY;
    g_angleY += dx * 0.5f;
    g_angleX += dy * 0.5f;
    g_lastX = x;
    g_lastY = y;
    glutPostRedisplay();
}

void keyboardFunc(unsigned char key, int, int) {
    switch(key) {
        case 27: // ESC
            if(g_app) {
                g_app->stopCapture();
            }
            cleanupGLBuffers();
            std::exit(0);
            break;
        case 'p':
        case 'P':
            g_drawPoints = !g_drawPoints;
            std::cout << "Points: " << (g_drawPoints ? "ON" : "OFF") << std::endl;
            break;
        case 'm':
        case 'M':
            // Cycle through mesh modes: FILL -> WIREFRAME -> OFF -> FILL
            g_meshMode = static_cast<MeshDisplayMode>((g_meshMode + 1) % 3);
            if(g_meshMode == MESH_OFF) {
                std::cout << "Mesh: OFF" << std::endl;
            }
            else if(g_meshMode == MESH_FILL) {
                std::cout << "Mesh: FILL (triangles: " << g_meshIndexCount/3 << ")" << std::endl;
            }
            else if(g_meshMode == MESH_WIREFRAME) {
                std::cout << "Mesh: WIREFRAME (triangles: " << g_meshIndexCount/3 << ")" << std::endl;
            }
            break;
        case ']':
            g_edgeFactor += 0.5f;
            std::cout << "edge x" << g_edgeFactor << std::endl;
            break;
        case '[':
            g_edgeFactor = std::max(0.5f, g_edgeFactor - 0.5f);
            std::cout << "edge x" << g_edgeFactor << std::endl;
            break;
        case '-':
            g_pixStride = std::max(1, g_pixStride - 1);
            std::cout << "pixStride=" << g_pixStride << std::endl;
            break;
        case '=':
            g_pixStride = std::min(16, g_pixStride + 1);
            std::cout << "pixStride=" << g_pixStride << std::endl;
            break;
        case 'w':
        case 'W':
            g_distance = std::max(0.1f, g_distance - 0.1f);
            break;
        case 's':
        case 'S':
            g_distance += 0.1f;
            break;
        case 'a':
        case 'A':
            g_autoRotate = !g_autoRotate;
            std::cout << "Auto-rotate: " << (g_autoRotate ? "ON" : "OFF") << std::endl;
            break;
        case ' ':
            g_paused = !g_paused;
            std::cout << "Paused: " << (g_paused ? "YES" : "NO") << std::endl;
            break;
        case 'c':
        case 'C':
            if(g_app) {
                g_app->savePointCloud("captured_cloud.ply");
                g_app->saveMesh("captured_mesh.ply");
                std::cout << "Saved current frame!" << std::endl;
            }
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

void printControls() {
    std::cout << "\n==================== CONTROLS ====================\n";
    std::cout << "  Mouse drag      : Rotate view\n";
    std::cout << "  Mouse wheel     : Zoom in/out\n";
    std::cout << "  W / S           : Zoom in/out\n";
    std::cout << "  P               : Toggle points display\n";
    std::cout << "  M               : Cycle mesh mode (FILL/WIREFRAME/OFF)\n";
    std::cout << "  [ / ]           : Adjust mesh edge factor\n";
    std::cout << "  - / =           : Adjust mesh pixel stride\n";
    std::cout << "  A               : Toggle auto-rotate\n";
    std::cout << "  SPACE           : Pause/Resume\n";
    std::cout << "  C               : Capture (save current frame)\n";
    std::cout << "  ESC             : Exit\n";
    std::cout << "==================================================\n\n";
}

//==================== Main ====================

int main(int argc, char** argv) {
    std::cout << "\n========================================\n";
    std::cout << "  Realtime Mesh Viewer with Orbbec SDK\n";
    std::cout << "========================================\n\n";

    RealtimeMeshReconstruction app;
    g_app = &app;

    // Initialize Orbbec SDK
    if(!app.initializeOrbbecSDK()) {
        std::cerr << "Failed to initialize Orbbec SDK!" << std::endl;
        return -1;
    }

    // Start capture thread
    app.startCapture();

    // Wait a moment for first frame
    std::cout << "Waiting for first frame..." << std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Initialize OpenGL/GLUT
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(g_winW, g_winH);
    glutCreateWindow("Realtime Mesh Viewer - Orbbec SDK");

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glEnable(GL_DEPTH_TEST);

    // Set up callbacks
    glutDisplayFunc(displayFunc);
    glutReshapeFunc(reshapeFunc);
    glutMouseFunc(mouseFunc);
    glutMotionFunc(motionFunc);
    glutKeyboardFunc(keyboardFunc);
    glutIdleFunc(idleFunc);

    printControls();

    // Enter main loop
    glutMainLoop();

    return 0;
}