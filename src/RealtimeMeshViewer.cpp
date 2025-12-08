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
#include <limits>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <string>
#include <sstream>
#include <iomanip>

// Orbbec SDK
#include "libobsensor/ObSensor.hpp"

// PCL
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/surface/organized_fast_mesh.h>
#include <pcl/io/ply_io.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>

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
extern float g_holePerimeter;
extern std::atomic<bool> g_paused;
extern bool  g_planeFillEnabled;

// Forward declaration for mesh hole patching helper (fast neighbor interpolation)
void patchSmallMeshHoles(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                         int stride,
                         float maxNeighborDist);

// Forward declaration for large-plane-based hole filling
void fillLargeHolesWithPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                             int stride,
                             float spacing);

// Global toggle: whether hole patching (small + plane) is enabled
bool g_holePatchEnabled = true;

std::string currentTimestampString() {
    auto now = std::chrono::system_clock::now();
    std::time_t nowTime = std::chrono::system_clock::to_time_t(now);
    std::tm tmBuf{};
#ifdef _WIN32
    localtime_s(&tmBuf, &nowTime);
#else
    localtime_r(&nowTime, &tmBuf);
#endif
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", &tmBuf);
    return std::string(buffer);
}

std::string appendTimestampToFilename(const std::string& filename) {
    std::string stamp = currentTimestampString();
    auto posSlash = filename.find_last_of("/\\");
    auto posDot = filename.find_last_of('.');
    if(posDot == std::string::npos || (posSlash != std::string::npos && posDot < posSlash)) {
        return filename + "_" + stamp;
    }
    return filename.substr(0, posDot) + "_" + stamp + filename.substr(posDot);
}

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
    std::atomic<bool> cloudUpdated{false};
    std::atomic<bool> meshReady{false};
    std::thread meshThread;
    
    // Consistency buffers - keep mesh and cloud in sync
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_for_mesh; // cloud snapshot used to build 'mesh'
    uint64_t cloudStamp = 0;   // increments every time we update cloud from sensor
    uint64_t meshStamp  = 0;   // the cloudStamp used to produce current 'mesh'
    
    // Orbbec SDK
    std::unique_ptr<ob::Pipeline> pipeline;
    std::unique_ptr<ob::PointCloudFilter> pointCloudFilter;
    std::thread captureThread;
    bool hasColorSensor;

    // Latest image buffers for UI display
    std::vector<uint8_t> latestRgbImage;
    std::vector<uint8_t> latestDepthImage;
    std::mutex imageMutex;
    std::atomic<bool> imagesReady{false};
    int imageW{0};
    int imageH{0};

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
        meshThread = std::thread(&RealtimeMeshReconstruction::meshLoop, this);
    }

    void stopCapture() {
        if(captureThread.joinable()) {
            std::cout << "Stopping capture thread..." << std::endl;
            shouldStop = true;
            captureThread.join();
        }
        if(meshThread.joinable()) {
            meshThread.join();
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
                    std::cout << "[OK] First point cloud generated successfully!" << std::endl;
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
                p.x = -points[i].x / 1000.0f;
                p.y = -points[i].y / 1000.0f;
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
            // Mark any trailing points (if pointsSize < width*height) as invalid
            for(int i = pointsSize; i < width * height; ++i) {
                pcl::PointXYZRGB &p = newCloud->points[i];
                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                p.r = p.g = p.b = 0;
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
                p.x = -points[i].x / 1000.0f;
                p.y = -points[i].y / 1000.0f;
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
            for(int i = pointsSize; i < width * height; ++i) {
                pcl::PointXYZRGB &p = newCloud->points[i];
                p.x = p.y = p.z = std::numeric_limits<float>::quiet_NaN();
                p.r = p.g = p.b = 0;
            }
        }
        
        // Count valid points
        int validPoints = 0;
        for(const auto& p : newCloud->points) {
            if(!std::isnan(p.x)) validPoints++;
        }

        // Prepare 2D RGB/depth previews for UI panels
        std::vector<uint8_t> rgbImage;
        std::vector<uint8_t> depthImage;
        if(width > 0 && height > 0) {
            const size_t imgSize = static_cast<size_t>(width) * height * 3;
            rgbImage.resize(imgSize);
            depthImage.resize(imgSize);
            for(size_t i = 0; i < newCloud->points.size(); ++i) {
                const auto& p = newCloud->points[i];
                const size_t idx = i * 3;
                rgbImage[idx + 0] = p.r;
                rgbImage[idx + 1] = p.g;
                rgbImage[idx + 2] = p.b;

                uint8_t depthGray = 0;
                if(pcl::isFinite(p)) {
                    const float depthMm = p.z * 1000.0f;
                    const float clamped = std::min(std::max(depthMm, 0.0f), 4000.0f);
                    depthGray = static_cast<uint8_t>(255.0f * (1.0f - clamped / 4000.0f));
                }
                depthImage[idx + 0] = depthGray;
                depthImage[idx + 1] = depthGray;
                depthImage[idx + 2] = depthGray;
            }
        }
        
        // Thread-safe update
        {
            std::lock_guard<std::recursive_mutex> lock(cloudMutex);
            cloud = newCloud;
            ++cloudStamp;  // Increment frame stamp
            hasNewData = true;
            cloudUpdated = true;
        }
        if(!rgbImage.empty() && !depthImage.empty()) {
            std::lock_guard<std::mutex> imgLock(imageMutex);
            imageW = width;
            imageH = height;
            latestRgbImage.swap(rgbImage);
            latestDepthImage.swap(depthImage);
            imagesReady = true;
        }
        
        if(firstUpdate) {
            std::cout << "[OK] Point cloud updated! Valid points: " << validPoints 
                      << " / " << (width * height) << std::endl;
            firstUpdate = false;
        }
    }

    void meshLoop() {
        while(!shouldStop) {
            if(cloudUpdated.exchange(false)) {
                reconstructMesh();
                meshReady = true;
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }

    bool hasNewDataAvailable() {
        return hasNewData.exchange(false);
    }

    bool hasNewMeshAvailable() {
        return meshReady.exchange(false);
    }

    bool getLatestImages(std::vector<uint8_t>& rgbOut,
                         std::vector<uint8_t>& depthOut,
                         int &outW,
                         int &outH,
                         bool &hasColorOut) {
        if(!imagesReady.exchange(false)) {
            return false;
        }
        std::lock_guard<std::mutex> lock(imageMutex);
        if(imageW == 0 || imageH == 0) {
            return false;
        }
        rgbOut = latestRgbImage;
        depthOut = latestDepthImage;
        outW = imageW;
        outH = imageH;
        hasColorOut = hasColorSensor;
        return true;
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

        if(firstReconstruct) {
            int validPoints = 0;
            for(const auto& p : used->points) {
                if(pcl::isFinite(p)) validPoints++;
            }
            std::cout << "[Mesh] Cloud dimensions: " << W << "x" << H << std::endl;
            std::cout << "[Mesh] Valid points: " << validPoints << " / " << used->points.size() 
                      << " (" << (100.0f * validPoints / used->points.size()) << "%)" << std::endl;
        }

        // Estimate spacing from the snapshot (compute locally to avoid lock) with caching
        static float cachedSpacing = 0.0f;
        static int spacingRefreshCounter = 0;
        auto computeSpacing = [&]() -> float {
            if(W <= 1 || H <= 1) return 0.0f;
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
            if(dists.empty()) return 0.0f;
            std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
            return dists[dists.size() / 2];
        };

        if(spacingRefreshCounter == 0 || spacingRefreshCounter % 60 == 0 || cachedSpacing <= 0.0f) {
            cachedSpacing = computeSpacing();
        }
        spacingRefreshCounter++;

        float spacing = cachedSpacing;
        if(spacing <= 0.0f) spacing = 0.01f;

        // Manual grid triangulation with stride
        const int s = std::max(1, g_pixStride);

        // Hole patching (small neighbor + plane) guarded by toggle
        if(g_holePatchEnabled) {
            patchSmallMeshHoles(used, s, g_holePerimeter);
        }
        if(g_planeFillEnabled) {
            fillLargeHolesWithPlane(used, s, spacing);
        }

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
            pcl::PCLPointCloud2 cloud2;
            pcl::toPCLPointCloud2(*used, cloud2);
            mesh.cloud = std::move(cloud2);
            *cloud_for_mesh = *used;  // Save the cloud snapshot used to build this mesh
            meshStamp = stamp_at_start;
        }

        reconstructCount++;
        if(!g_paused) {
            std::cout << "[Mesh] #" << reconstructCount
                      << " stride=" << s
                      << " spacing=" << spacing
                      << " maxEdge=" << maxDistance
                      << " tris=" << mesh.polygons.size()
                      << " maxEdgeKept=" << max_kept << std::endl;
        }

        firstReconstruct = false;
    }

    void saveMesh(const std::string& filename) {
        std::lock_guard<std::recursive_mutex> lock(cloudMutex);
        if(mesh.cloud.data.empty()) {
            std::cerr << "[SaveMesh] Mesh has no vertex cloud; aborting write." << std::endl;
            return;
        }
        const std::string stampedName = appendTimestampToFilename(filename);
        std::cout << "Saving mesh to: " << stampedName << std::endl;
        pcl::io::savePLYFile(stampedName, mesh);
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

//==================== Mesh hole patching (small holes only) ====================

void patchSmallMeshHoles(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                         int stride,
                         float maxNeighborDist)
{
    if(!cloud || stride <= 0) return;

    const int W = static_cast<int>(cloud->width);
    const int H = static_cast<int>(cloud->height);
    if(W <= 0 || H <= 0 || cloud->points.size() != static_cast<size_t>(W * H)) return;

    auto &pts = cloud->points;
    const int step    = stride;      // horizontal / vertical stride step
    const int rowStep = stride * W;  // stride rows to index step
    const float maxNeighborDist2 = (maxNeighborDist > 0.0f)
        ? (maxNeighborDist * maxNeighborDist)
        : std::numeric_limits<float>::max();
    int patched = 0;

    auto dist2 = [&](const pcl::PointXYZRGB& a, const pcl::PointXYZRGB& b)->float {
        float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
        return dx*dx + dy*dy + dz*dz;
    };

    // Only scan stride-aligned coordinates (same as triangulation samples)
    for(int r = 0; r < H; r += step) {
        for(int c = 0; c < W; c += step) {
            int idx = r * W + c;
            if(pcl::isFinite(pts[idx])) continue;

            bool hasLeft  = (c - step) >= 0 && pcl::isFinite(pts[idx - step]);
            bool hasRight = (c + step) < W  && pcl::isFinite(pts[idx + step]);
            bool hasUp    = (r - step) >= 0 && (idx - rowStep) >= 0 &&
                            pcl::isFinite(pts[idx - rowStep]);
            bool hasDown  = (r + step) < H  && (idx + rowStep) < static_cast<int>(pts.size()) &&
                            pcl::isFinite(pts[idx + rowStep]);

            if(!(hasLeft || hasRight || hasUp || hasDown)) continue;

            pcl::PointXYZRGB p{};
            int count = 0;

            // 4-neighbor average (preferred)
            if(hasLeft && hasRight && hasUp && hasDown) {
                if(dist2(pts[idx - step],    pts[idx + step])    < maxNeighborDist2 &&
                   dist2(pts[idx - rowStep], pts[idx + rowStep]) < maxNeighborDist2) {
                    p.x = (pts[idx - step].x    + pts[idx + step].x
                         + pts[idx - rowStep].x + pts[idx + rowStep].x) * 0.25f;
                    p.y = (pts[idx - step].y    + pts[idx + step].y
                         + pts[idx - rowStep].y + pts[idx + rowStep].y) * 0.25f;
                    p.z = (pts[idx - step].z    + pts[idx + step].z
                         + pts[idx - rowStep].z + pts[idx + rowStep].z) * 0.25f;
                    p.r = static_cast<uint8_t>((pts[idx - step].r    + pts[idx + step].r
                                              + pts[idx - rowStep].r + pts[idx + rowStep].r) >> 2);
                    p.g = static_cast<uint8_t>((pts[idx - step].g    + pts[idx + step].g
                                              + pts[idx - rowStep].g + pts[idx + rowStep].g) >> 2);
                    p.b = static_cast<uint8_t>((pts[idx - step].b    + pts[idx + step].b
                                              + pts[idx - rowStep].b + pts[idx + rowStep].b) >> 2);
                    count = 4;
                }
            }

            // Horizontal interpolation
            if(count == 0 && hasLeft && hasRight) {
                if(dist2(pts[idx - step], pts[idx + step]) < maxNeighborDist2) {
                    p.x = (pts[idx - step].x + pts[idx + step].x) * 0.5f;
                    p.y = (pts[idx - step].y + pts[idx + step].y) * 0.5f;
                    p.z = (pts[idx - step].z + pts[idx + step].z) * 0.5f;
                    p.r = static_cast<uint8_t>((pts[idx - step].r + pts[idx + step].r) >> 1);
                    p.g = static_cast<uint8_t>((pts[idx - step].g + pts[idx + step].g) >> 1);
                    p.b = static_cast<uint8_t>((pts[idx - step].b + pts[idx + step].b) >> 1);
                    count = 2;
                }
            }

            // Vertical interpolation
            if(count == 0 && hasUp && hasDown) {
                if(dist2(pts[idx - rowStep], pts[idx + rowStep]) < maxNeighborDist2) {
                    p.x = (pts[idx - rowStep].x + pts[idx + rowStep].x) * 0.5f;
                    p.y = (pts[idx - rowStep].y + pts[idx + rowStep].y) * 0.5f;
                    p.z = (pts[idx - rowStep].z + pts[idx + rowStep].z) * 0.5f;
                    p.r = static_cast<uint8_t>((pts[idx - rowStep].r + pts[idx + rowStep].r) >> 1);
                    p.g = static_cast<uint8_t>((pts[idx - rowStep].g + pts[idx + rowStep].g) >> 1);
                    p.b = static_cast<uint8_t>((pts[idx - rowStep].b + pts[idx + rowStep].b) >> 1);
                    count = 2;
                }
            }

            if(count > 0) {
                pts[idx] = p;
                patched++;
            }
        }
    }
}

void fillLargeHolesWithPlane(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
                             int stride,
                             float spacing)
{
    if(!cloud || stride <= 0 || !g_planeFillEnabled) return;

    const int W = static_cast<int>(cloud->width);
    const int H = static_cast<int>(cloud->height);
    if(W <= 0 || H <= 0 || cloud->points.size() != static_cast<size_t>(W * H)) return;

    auto &pts = cloud->points;

    // ---- 1) Sample finite points and run a lightweight RANSAC plane fit ----
    std::vector<int> sampleIdx;
    sampleIdx.reserve(4000);

    const int sampleStep = stride * 4;  // coarser sampling to reduce cost
    for(int r = 0; r < H; r += sampleStep) {
        for(int c = 0; c < W; c += sampleStep) {
            int idx = r * W + c;
            if(pcl::isFinite(pts[idx])) {
                sampleIdx.push_back(idx);
            }
        }
    }
    if(sampleIdx.size() < 50) {
        return; // not enough samples for plane fitting
    }

    auto randIdx = [&](int n) -> int {
        return std::rand() % n;
    };

    struct Plane {
        float a{}, b{}, c{}, d{};
        int inliers{};
    };

    const int   maxIter     = 80;
    const float distThresh  = std::max(0.01f, spacing * 2.0f); // inlier distance threshold
    const float distThresh2 = distThresh * distThresh;
    const int   maxPlanes   = 3;

    auto findPlane = [&](const std::vector<int>& pool,
                         Plane& outPlane,
                         std::vector<int>& outInliers)->bool {
        if(pool.size() < 30) return false;
        int bestInlierCount = 0;
        float bestA = 0.f, bestB = 0.f, bestC = 1.f, bestD = 0.f;
        for(int iter = 0; iter < maxIter; ++iter) {
            int i1 = randIdx(static_cast<int>(pool.size()));
            int i2 = randIdx(static_cast<int>(pool.size()));
            int i3 = randIdx(static_cast<int>(pool.size()));
            if(i1 == i2 || i1 == i3 || i2 == i3) continue;
            const auto &p1 = pts[pool[i1]];
            const auto &p2 = pts[pool[i2]];
            const auto &p3 = pts[pool[i3]];
            if(!pcl::isFinite(p1) || !pcl::isFinite(p2) || !pcl::isFinite(p3)) continue;
            float ux = p2.x - p1.x, uy = p2.y - p1.y, uz = p2.z - p1.z;
            float vx = p3.x - p1.x, vy = p3.y - p1.y, vz = p3.z - p1.z;
            float nx = uy * vz - uz * vy;
            float ny = uz * vx - ux * vz;
            float nz = ux * vy - uy * vx;
            float norm2 = nx*nx + ny*ny + nz*nz;
            if(norm2 < 1e-6f) continue;
            float invNorm = 1.0f / std::sqrt(norm2);
            nx *= invNorm; ny *= invNorm; nz *= invNorm;
            float d = -(nx * p1.x + ny * p1.y + nz * p1.z);
            int inlierCount = 0;
            for(int idx : pool) {
                const auto &q = pts[idx];
                float dist = nx * q.x + ny * q.y + nz * q.z + d;
                if(dist * dist < distThresh2) {
                    ++inlierCount;
                }
            }
            if(inlierCount > bestInlierCount) {
                bestInlierCount = inlierCount;
                bestA = nx; bestB = ny; bestC = nz; bestD = d;
            }
        }
        if(bestInlierCount < static_cast<int>(pool.size() * 0.3f)) {
            return false;
        }
        outInliers.clear();
        outInliers.reserve(pool.size());
        for(int idx : pool) {
            const auto &q = pts[idx];
            float dist = bestA * q.x + bestB * q.y + bestC * q.z + bestD;
            if(dist * dist < distThresh2) {
                outInliers.push_back(idx);
            }
        }
        outPlane = Plane{bestA, bestB, bestC, bestD, static_cast<int>(outInliers.size())};
        return true;
    };

    std::vector<int> pool = sampleIdx;
    std::vector<Plane> planes;
    planes.reserve(maxPlanes);
    for(int k = 0; k < maxPlanes && !pool.empty(); ++k) {
        Plane pl;
        std::vector<int> inliers;
        if(!findPlane(pool, pl, inliers)) break;
        planes.push_back(pl);
        // Remove inliers from pool for next plane
        std::sort(inliers.begin(), inliers.end());
        std::vector<int> newPool;
        newPool.reserve(pool.size());
        for(int idx : pool) {
            if(!std::binary_search(inliers.begin(), inliers.end(), idx)) {
                newPool.push_back(idx);
            }
        }
        pool.swap(newPool);
    }

    if(planes.empty()) {
        return;
    } else {
        std::cout << "[PlaneFill] Found " << planes.size() << " plane(s)" << std::endl;
    }

    // ---- 2) Patch stride-grid holes that align with any detected plane ----

    const int step    = stride;
    const int rowStep = stride * W;
    int patched = 0;

    for(int r = 0; r < H; r += step) {
        for(int c = 0; c < W; c += step) {
            int idx = r * W + c;
            if(pcl::isFinite(pts[idx])) continue;  // only patch missing points

            // Collect nearby plane inliers to estimate (x,y)
            std::vector<const pcl::PointXYZRGB*> neighbors;
            neighbors.reserve(8);

            auto tryAddNeighbor = [&](int rr, int cc) {
                if(rr < 0 || rr >= H || cc < 0 || cc >= W) return;
                int id = rr * W + cc;
                const auto &q = pts[id];
                if(!pcl::isFinite(q)) return;
                for(const auto& pl : planes) {
                    float dist = pl.a * q.x + pl.b * q.y + pl.c * q.z + pl.d;
                    if(dist * dist < distThresh2 * 4.0f) { // allow slightly looser band
                        neighbors.push_back(&q);
                        break;
                    }
                }
            };

            // Search plane neighbors in a 3x3 stride block
            for(int dr = -step*2; dr <= step*2; dr += step) {
                for(int dc = -step*2; dc <= step*2; dc += step) {
                    if(dr == 0 && dc == 0) continue;
                    tryAddNeighbor(r + dr, c + dc);
                }
            }
            if(neighbors.size() < 3) continue; // need at least 3 neighbors

            // Estimate (x,y) via neighbor mean; solve z from best matching plane
            float meanX = 0.f, meanY = 0.f, meanZ = 0.f;
            uint32_t meanR = 0, meanG = 0, meanB = 0;
            for(auto q : neighbors) {
                meanX += q->x;
                meanY += q->y;
                meanZ += q->z;
                meanR += q->r;
                meanG += q->g;
                meanB += q->b;
            }
            float invN = 1.0f / neighbors.size();
            meanX *= invN;
            meanY *= invN;
            meanZ *= invN;
            meanR = static_cast<uint32_t>(meanR * invN);
            meanG = static_cast<uint32_t>(meanG * invN);
            meanB = static_cast<uint32_t>(meanB * invN);

            // Pick plane that best explains the neighbors
            int bestPlaneIdx = -1;
            int bestPlaneSupport = 0;
            for(size_t pi = 0; pi < planes.size(); ++pi) {
                int support = 0;
                const auto& pl = planes[pi];
                for(auto q : neighbors) {
                    float dist = pl.a * q->x + pl.b * q->y + pl.c * q->z + pl.d;
                    if(dist * dist < distThresh2 * 4.0f) {
                        ++support;
                    }
                }
                if(support > bestPlaneSupport) {
                    bestPlaneSupport = support;
                    bestPlaneIdx = static_cast<int>(pi);
                }
            }
            if(bestPlaneIdx < 0) continue;
            const auto& plane = planes[bestPlaneIdx];

            pcl::PointXYZRGB p{};
            p.x = meanX;
            p.y = meanY;

            // If normal z is tiny (near-vertical plane), fall back to meanZ
            if(std::fabs(plane.c) > 1e-3f) {
                p.z = -(plane.a * p.x + plane.b * p.y + plane.d) / plane.c;
            } else {
                p.z = meanZ;
            }

            // Basic z sanity
            if(p.z < 0.05f || p.z > 20.0f) {
                continue;
            }

            p.r = static_cast<uint8_t>(meanR);
            p.g = static_cast<uint8_t>(meanG);
            p.b = static_cast<uint8_t>(meanB);

            pts[idx] = p;
            ++patched;
        }
    }

    if(patched > 0) {
        std::cout << "[PlaneFill] Patched " << patched
                  << " stride-grid points on dominant plane" << std::endl;
    }
}

//==================== OpenGL Viewer with VBO ====================

// Mesh display modes
enum MeshDisplayMode {
    MESH_OFF = 0,
    MESH_FILL = 1,
    MESH_WIREFRAME = 2
};

RealtimeMeshReconstruction* g_app = nullptr;

constexpr float kDefaultAngleX   = 20.0f;
constexpr float kDefaultAngleY   = -30.0f;
constexpr float kDefaultCenterX  = 0.0f;
constexpr float kDefaultCenterY  = 0.0f;
constexpr float kDefaultCenterZ  = 0.5f;
constexpr float kDefaultDistance = 2.5f;

int   g_winW = 1280;
int   g_winH = 720;
float g_angleX = kDefaultAngleX;    // Pitch
float g_angleY = kDefaultAngleY;    // Yaw
float g_roll   = 0.0f;              // Roll
float g_camX = 0.0f, g_camY = 0.5f, g_camZ = -2.5f;  // Drone camera position
float g_moveSpeed = 0.05f;                           // Movement speed (m per keypress)
int   g_lastX = 0;
int   g_lastY = 0;
bool  g_leftDown = false;
bool  g_drawPoints = true;   // Start with points visible for debugging
MeshDisplayMode g_meshMode = MESH_FILL;  // Start with filled mesh
bool  g_autoRotate = false;
std::atomic<bool> g_paused{false};

float g_sceneScale = 1.0f;

// VBO / EBO handles
GLuint g_vboPos = 0;
GLuint g_vboCol = 0;
GLuint g_eboMesh = 0;
GLsizei g_pointCount = 0;
GLsizei g_meshIndexCount = 0;

// Mesh edge length factor (controls max triangle edge length = spacing * factor)
float g_edgeFactor = 40.0f;

// Grid triangulation controls
int   g_pixStride = 4;      // pixel stride for manual grid triangulation
float g_dzMax     = 0.08f;  // allowed depth jump between adjacent vertices (meters)

// Mesh hole patching threshold (perimeter in meters)
float g_holePerimeter = 0.6f;
bool  g_planeFillEnabled = false;

// Image panel controls/textures
bool   g_showImagePanel = true;
GLuint g_texRgb   = 0;
GLuint g_texDepth = 0;
int    g_imageW   = 0;
int    g_imageH   = 0;

constexpr float kPanelPadding = 10.0f;

// Performance tracking
int g_frameCount = 0;
int g_lastTime = 0;
float g_fps = 0.0f;
float g_lastFrameMs = 0.0f;
std::string g_lastStatus = "Ready";

void cleanupGLBuffers() {
    if(g_vboPos) { glDeleteBuffers(1, &g_vboPos); g_vboPos = 0; }
    if(g_vboCol) { glDeleteBuffers(1, &g_vboCol); g_vboCol = 0; }
    if(g_eboMesh) { glDeleteBuffers(1, &g_eboMesh); g_eboMesh = 0; }
    if(g_texRgb) { glDeleteTextures(1, &g_texRgb); g_texRgb = 0; }
    if(g_texDepth) { glDeleteTextures(1, &g_texDepth); g_texDepth = 0; }
    g_pointCount = 0;
    g_meshIndexCount = 0;
}

void updateGLBuffers() {
    if(!g_app) return;

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

    // Vertex position VBO (reuse buffer, reallocate only if needed)
    if(!g_vboPos) glGenBuffers(1, &g_vboPos);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboPos);
    static GLsizeiptr vboPosCapacity = 0;
    GLsizeiptr neededPos = static_cast<GLsizeiptr>(g_pointCount) * 3 * sizeof(float);
    if(neededPos > vboPosCapacity) {
        glBufferData(GL_ARRAY_BUFFER, neededPos, nullptr, GL_DYNAMIC_DRAW);
        vboPosCapacity = neededPos;
    }
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

    // Color VBO (reuse)
    if(!g_vboCol) glGenBuffers(1, &g_vboCol);
    glBindBuffer(GL_ARRAY_BUFFER, g_vboCol);
    static GLsizeiptr vboColCapacity = 0;
    GLsizeiptr neededCol = static_cast<GLsizeiptr>(g_pointCount) * 3 * sizeof(GLubyte);
    if(neededCol > vboColCapacity) {
        glBufferData(GL_ARRAY_BUFFER, neededCol, nullptr, GL_DYNAMIC_DRAW);
        vboColCapacity = neededCol;
    }
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

    // Mesh indices EBO with lightweight validation
    std::vector<GLuint> indices;
    indices.reserve(mesh.polygons.size() * 3);
    const auto &pts = cloud->points;

    for(const auto& poly : mesh.polygons) {
        if(poly.vertices.size() != 3) continue;
        GLuint i0 = poly.vertices[0];
        GLuint i1 = poly.vertices[1];
        GLuint i2 = poly.vertices[2];
        if(i0 >= pts.size() || i1 >= pts.size() || i2 >= pts.size()) continue;
        if(!pcl::isFinite(pts[i0]) || !pcl::isFinite(pts[i1]) || !pcl::isFinite(pts[i2])) continue;

        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
    }
    g_meshIndexCount = static_cast<GLsizei>(indices.size());
    
    static bool firstMeshUpdate = true;
    if(firstMeshUpdate && g_meshIndexCount > 0) {
        std::cout << "[OpenGL] Mesh buffers created. Triangles: " << (g_meshIndexCount/3) << std::endl;
        firstMeshUpdate = false;
    }
    
    if(g_meshIndexCount > 0) {
        if(!g_eboMesh) glGenBuffers(1, &g_eboMesh);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_eboMesh);
        static GLsizeiptr eboCapacity = 0;
        GLsizeiptr neededEbo = static_cast<GLsizeiptr>(indices.size()) * sizeof(GLuint);
        if(neededEbo > eboCapacity) {
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, neededEbo, nullptr, GL_DYNAMIC_DRAW);
            eboCapacity = neededEbo;
        }
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, neededEbo, indices.data());
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }
}

void ensureTexture2D(GLuint &tex) {
    if(!tex) {
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    } else {
        glBindTexture(GL_TEXTURE_2D, tex);
    }
}

void updateImageTextures() {
    if(!g_app) return;

    std::vector<uint8_t> rgb;
    std::vector<uint8_t> depth;
    int w = 0, h = 0;
    bool hasColor = false;
    if(!g_app->getLatestImages(rgb, depth, w, h, hasColor)) {
        return;
    }
    (void)hasColor;

    g_imageW = w;
    g_imageH = h;
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if(!rgb.empty()) {
        ensureTexture2D(g_texRgb);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, rgb.data());
    }
    if(!depth.empty()) {
        ensureTexture2D(g_texDepth);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, depth.data());
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void drawImagePanel(GLuint tex, float x, float y, float w, float h) {
    glBindTexture(GL_TEXTURE_2D, tex);
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 1.0f); glVertex2f(x,         y - h);
    glTexCoord2f(1.0f, 1.0f); glVertex2f(x + w,     y - h);
    glTexCoord2f(1.0f, 0.0f); glVertex2f(x + w,     y);
    glTexCoord2f(0.0f, 0.0f); glVertex2f(x,         y);
    glEnd();
}

void drawBitmapLine(float x, float y, const char* text) {
    glRasterPos2f(x, y);
    while(*text) {
        glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *text);
        ++text;
    }
}

void setStatus(const std::string& msg) {
    g_lastStatus = msg;
}

void drawImagePanels() {
    if(!g_showImagePanel) return;
    if((g_texRgb == 0 && g_texDepth == 0) || g_imageW == 0 || g_imageH == 0) return;

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0, g_winW, 0, g_winH, -1, 1);

    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
    glDisable(GL_CULL_FACE);
    glEnable(GL_TEXTURE_2D);
    glColor3f(1.0f, 1.0f, 1.0f);

    const float panelWidth = std::max(120.0f, g_winW * 0.28f);
    const float aspect = static_cast<float>(g_imageH) / static_cast<float>(g_imageW);
    const float panelHeight = panelWidth * aspect;

    float x = g_winW - panelWidth - kPanelPadding;
    float y = g_winH - kPanelPadding;

    if(g_texRgb) {
        drawImagePanel(g_texRgb, x, y, panelWidth, panelHeight);
        y -= (panelHeight + kPanelPadding);
    }
    if(g_texDepth) {
        drawImagePanel(g_texDepth, x, y, panelWidth, panelHeight);
        y -= (panelHeight + kPanelPadding);
    }

    // Key hint block under panels (English)
    glDisable(GL_TEXTURE_2D);
    glColor3f(1.0f, 1.0f, 1.0f);
    const char* lines[] = {
        "Controls:",
        "W/S: Move forward/backward",
        "A/D: Strafe left/right",
        "Q/E: Roll left/right",
        "Arrows: Move up/down",
        "P: Toggle points",
        "M: Cycle mesh mode",
        "V: Toggle RGB/Depth panels",
        "F: Toggle plane fill",
        "T: Auto rotate, R: Reset view",
        "H: Toggle hole patch",
        "Space: Pause/Resume",
        "C: Capture ply",
        "ESC: Exit"
    };
    float textX = x;
    float textY = y;
    const float lineH = 14.0f;
    for(const char* line : lines) {
        drawBitmapLine(textX, textY, line);
        textY -= lineH;
    }

    // Metrics and last status
    std::ostringstream ossFps;
    ossFps << "FPS: " << std::fixed << std::setprecision(1) << g_fps;
    std::ostringstream ossLat;
    ossLat << "Frame time: " << static_cast<int>(g_lastFrameMs) << " ms";
    std::string statusLine = "Status: " + g_lastStatus;

    drawBitmapLine(textX, textY, ossFps.str().c_str()); textY -= lineH;
    drawBitmapLine(textX, textY, ossLat.str().c_str()); textY -= lineH;
    drawBitmapLine(textX, textY, statusLine.c_str());   textY -= lineH;

    glDepthMask(GL_TRUE);
    glEnable(GL_DEPTH_TEST);

    glPopMatrix();
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
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

    int totalTime = 0;

    if(!g_paused && g_app && g_app->hasNewMeshAvailable()) {
        int t0 = glutGet(GLUT_ELAPSED_TIME);
        updateGLBuffers();
        if(g_showImagePanel) {
            updateImageTextures();
        }
        int t1 = glutGet(GLUT_ELAPSED_TIME);

        totalTime = t1 - frameStart;
        g_lastFrameMs = static_cast<float>(totalTime);

        updateCount++;
        frameIndex++;
        if(updateCount == 1) {
            std::cout << "[OpenGL] First data received and buffers updated!" << std::endl;
        }
        std::cout << "[Perf] frame " << frameIndex
                  << " | total: " << totalTime << " ms" << std::endl;
    }

    lastTime = currentTime;
    
    if(g_autoRotate) {
        g_angleY += 0.5f;
    }
    
    glutPostRedisplay();
}

void resetCameraPose() {
    g_angleX = kDefaultAngleX;
    g_angleY = kDefaultAngleY;
    g_roll   = 0.0f;

    const float deg2rad = 3.14159265f / 180.0f;
    float yaw   = g_angleY * deg2rad;
    float pitch = g_angleX * deg2rad;
    float cosPitch = std::cos(pitch);
    float sinPitch = std::sin(pitch);
    float cosYaw   = std::cos(yaw);
    float sinYaw   = std::sin(yaw);

    float dirX = cosPitch * sinYaw;
    float dirY = sinPitch;
    float dirZ = cosPitch * cosYaw;

    g_camX = kDefaultCenterX - dirX * kDefaultDistance;
    g_camY = kDefaultCenterY - dirY * kDefaultDistance;
    g_camZ = kDefaultCenterZ - dirZ * kDefaultDistance;
}

void getCameraBasis(
    float &dirX,  float &dirY,  float &dirZ,
    float &upX,   float &upY,   float &upZ,
    float &rightX,float &rightY,float &rightZ)
{
    const float deg2rad = 3.14159265f / 180.0f;
    float yaw   = g_angleY * deg2rad;
    float pitch = g_angleX * deg2rad;
    float roll  = g_roll   * deg2rad;

    float cosPitch = std::cos(pitch);
    float sinPitch = std::sin(pitch);
    float cosYaw   = std::cos(yaw);
    float sinYaw   = std::sin(yaw);

    dirX = cosPitch * sinYaw;
    dirY = sinPitch;
    dirZ = cosPitch * cosYaw;

    float worldUpX = 0.0f, worldUpY = 1.0f, worldUpZ = 0.0f;
    float dotUp = dirX * worldUpX + dirY * worldUpY + dirZ * worldUpZ;
    if(std::fabs(dotUp) > 0.95f) {
        worldUpX = 0.0f;
        worldUpY = 0.0f;
        worldUpZ = 1.0f;
    }

    rightX = dirY * worldUpZ - dirZ * worldUpY;
    rightY = dirZ * worldUpX - dirX * worldUpZ;
    rightZ = dirX * worldUpY - dirY * worldUpX;
    float rightLen = std::sqrt(rightX*rightX + rightY*rightY + rightZ*rightZ);
    if(rightLen > 1e-6f) {
        rightX /= rightLen;
        rightY /= rightLen;
        rightZ /= rightLen;
    }

    upX = rightY * dirZ - rightZ * dirY;
    upY = rightZ * dirX - rightX * dirZ;
    upZ = rightX * dirY - rightY * dirX;
    float upLen = std::sqrt(upX*upX + upY*upY + upZ*upZ);
    if(upLen > 1e-6f) {
        upX /= upLen;
        upY /= upLen;
        upZ /= upLen;
    }

    float cu = std::cos(roll);
    float su = std::sin(roll);

    float upX2    = upX    * cu + rightX * su;
    float upY2    = upY    * cu + rightY * su;
    float upZ2    = upZ    * cu + rightZ * su;

    float rightX2 = rightX * cu - upX    * su;
    float rightY2 = rightY * cu - upY    * su;
    float rightZ2 = rightZ * cu - upZ    * su;

    upX = upX2;    upY = upY2;    upZ = upZ2;
    rightX = rightX2; rightY = rightY2; rightZ = rightZ2;
}

void displayFunc() {
    static int frameNum = 0;
    frameNum++;

    // Full-window viewport; side panels are drawn as overlays without shrinking the view
    glViewport(0, 0, g_winW, g_winH);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    double aspect = (g_winW > 0 && g_winH > 0) ? static_cast<double>(g_winW)/g_winH : 16.0/9.0;
    gluPerspective(45.0, aspect, 0.01, 100.0);  // 45 deg FOV

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    float dirX, dirY, dirZ;
    float upX, upY, upZ;
    float rightX, rightY, rightZ;
    getCameraBasis(dirX, dirY, dirZ, upX, upY, upZ, rightX, rightY, rightZ);

    gluLookAt(
        g_camX,             g_camY,             g_camZ,
        g_camX + dirX,      g_camY + dirY,      g_camZ + dirZ,
        upX, upY, upZ);

    glScalef(g_sceneScale, g_sceneScale, g_sceneScale);

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
                // Wireframe mesh
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

    // Make sure overlay quads are filled even if mesh mode set polygon mode to lines
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // Overlay RGB/Depth panels on the right
    bool panelsVisible = g_showImagePanel && g_imageW > 0 && g_imageH > 0 && (g_texRgb || g_texDepth);
    if(panelsVisible) {
        drawImagePanels();
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
    // Mouse wheel adjusts movement speed
    if(button == 3 && state == GLUT_DOWN) {
        g_moveSpeed = std::min(1.0f, g_moveSpeed * 1.2f);
        std::cout << "moveSpeed = " << g_moveSpeed << std::endl;
        glutPostRedisplay();
    }
    if(button == 4 && state == GLUT_DOWN) {
        g_moveSpeed = std::max(0.005f, g_moveSpeed / 1.2f);
        std::cout << "moveSpeed = " << g_moveSpeed << std::endl;
        glutPostRedisplay();
    }
}

void motionFunc(int x, int y) {
    int dx = x - g_lastX;
    int dy = y - g_lastY;
    g_lastX = x;
    g_lastY = y;

    if(!g_leftDown) return;

    g_angleY += dx * 0.5f;
    g_angleX += dy * 0.5f;

    if(g_angleY > 360.0f || g_angleY < -360.0f) {
        g_angleY = std::fmod(g_angleY, 360.0f);
    }
    if(g_angleX > 360.0f || g_angleX < -360.0f) {
        g_angleX = std::fmod(g_angleX, 360.0f);
    }

    glutPostRedisplay();
}

void keyboardFunc(unsigned char key, int, int) {
    const float deg2rad = 3.14159265f / 180.0f;

    switch(key) {
        case 27: // ESC
            if(g_app) {
                g_app->stopCapture();
            }
            cleanupGLBuffers();
            std::exit(0);
            break;
        case 'w':
        case 'W': {
            float yaw = g_angleY * deg2rad;
            g_camX += std::sin(yaw) * g_moveSpeed;
            g_camZ += std::cos(yaw) * g_moveSpeed;
            break;
        }
        case 's':
        case 'S': {
            float yaw = g_angleY * deg2rad;
            g_camX -= std::sin(yaw) * g_moveSpeed;
            g_camZ -= std::cos(yaw) * g_moveSpeed;
            break;
        }
        case 'd':
        case 'D': {
            float yaw = g_angleY * deg2rad;
            g_camX -= std::cos(yaw) * g_moveSpeed;
            g_camZ += std::sin(yaw) * g_moveSpeed;
            break;
        }
        case 'a':
        case 'A': {
            float yaw = g_angleY * deg2rad;
            g_camX += std::cos(yaw) * g_moveSpeed;
            g_camZ -= std::sin(yaw) * g_moveSpeed;
            break;
        }
        case 'q':
        case 'Q':
            g_roll = std::max(-45.0f, g_roll - 2.0f);
            break;
        case 'e':
        case 'E':
            g_roll = std::min(45.0f, g_roll + 2.0f);
            break;
        case 'p':
        case 'P':
            g_drawPoints = !g_drawPoints;
            std::cout << "Points: " << (g_drawPoints ? "ON" : "OFF") << std::endl;
            setStatus(std::string("Points: ") + (g_drawPoints ? "ON" : "OFF"));
            break;
        case 'v':
        case 'V':
            g_showImagePanel = !g_showImagePanel;
            std::cout << "RGB/Depth panels: " << (g_showImagePanel ? "ON" : "OFF") << std::endl;
            setStatus(std::string("Panels: ") + (g_showImagePanel ? "ON" : "OFF"));
            break;
        case 'm':
        case 'M':
            g_meshMode = static_cast<MeshDisplayMode>((g_meshMode + 1) % 3);
            if(g_meshMode == MESH_OFF) {
                std::cout << "Mesh: OFF" << std::endl;
                setStatus("Mesh: OFF");
            }
            else if(g_meshMode == MESH_FILL) {
                std::cout << "Mesh: FILL (triangles: " << g_meshIndexCount/3 << ")" << std::endl;
                setStatus("Mesh: FILL");
            }
            else if(g_meshMode == MESH_WIREFRAME) {
                std::cout << "Mesh: WIREFRAME (triangles: " << g_meshIndexCount/3 << ")" << std::endl;
                setStatus("Mesh: WIREFRAME");
            }
            break;
        case ']':
            g_edgeFactor += 0.5f;
            std::cout << "edge x" << g_edgeFactor << std::endl;
            setStatus("Edge factor x" + std::to_string(g_edgeFactor));
            break;
        case '[':
            g_edgeFactor = std::max(0.5f, g_edgeFactor - 0.5f);
            std::cout << "edge x" << g_edgeFactor << std::endl;
            setStatus("Edge factor x" + std::to_string(g_edgeFactor));
            break;
        case '-':
            g_pixStride = std::max(1, g_pixStride - 1);
            std::cout << "pixStride=" << g_pixStride << std::endl;
            setStatus("Pix stride=" + std::to_string(g_pixStride));
            break;
        case '=':
            g_pixStride = std::min(16, g_pixStride + 1);
            std::cout << "pixStride=" << g_pixStride << std::endl;
            setStatus("Pix stride=" + std::to_string(g_pixStride));
            break;
        case '.':
        case '>':
            g_holePerimeter = std::min(5.0f, g_holePerimeter + 0.1f);
            std::cout << "holePerimeter=" << g_holePerimeter << " m" << std::endl;
            setStatus("Hole perimeter=" + std::to_string(g_holePerimeter));
            break;
        case ',':
        case '<':
            g_holePerimeter = std::max(0.1f, g_holePerimeter - 0.1f);
            std::cout << "holePerimeter=" << g_holePerimeter << " m" << std::endl;
            setStatus("Hole perimeter=" + std::to_string(g_holePerimeter));
            break;
        case 'r':
        case 'R':
            resetCameraPose();
            std::cout << "[Camera] Reset to default view" << std::endl;
            setStatus("Camera reset");
            break;
        case 't':
        case 'T':
            g_autoRotate = !g_autoRotate;
            std::cout << "Auto-rotate: " << (g_autoRotate ? "ON" : "OFF") << std::endl;
            setStatus(std::string("Auto-rotate: ") + (g_autoRotate ? "ON" : "OFF"));
            break;
        case ' ':
            g_paused = !g_paused;
            std::cout << "Paused: " << (g_paused ? "YES" : "NO") << std::endl;
            setStatus(std::string("Paused: ") + (g_paused ? "YES" : "NO"));
            break;
        case 'c':
        case 'C':
            if(g_app) {
                g_app->savePointCloud("captured_cloud.ply");
                g_app->saveMesh("captured_mesh.ply");
                std::cout << "Saved current frame!" << std::endl;
                setStatus("Captured current frame");
            }
            break;
        case 'h':
        case 'H':
            g_holePatchEnabled = !g_holePatchEnabled;
            std::cout << "Hole patching (small + plane): "
                      << (g_holePatchEnabled ? "ON" : "OFF") << std::endl;
            setStatus(std::string("Hole patching: ") + (g_holePatchEnabled ? "ON" : "OFF"));
            break;
        case 'f':
        case 'F':
            g_planeFillEnabled = !g_planeFillEnabled;
            std::cout << "Plane-based hole fill: "
                      << (g_planeFillEnabled ? "ON" : "OFF") << std::endl;
            setStatus(std::string("Plane fill: ") + (g_planeFillEnabled ? "ON" : "OFF"));
            break;
        default:
            break;
    }
    glutPostRedisplay();
}

void specialFunc(int key, int, int) {
    switch(key) {
        case GLUT_KEY_UP:
            g_camY += g_moveSpeed;
            break;
        case GLUT_KEY_DOWN:
            g_camY -= g_moveSpeed;
            break;
        default:
            return;
    }
    glutPostRedisplay();
}

void printControls() {
    std::cout << "\n==================== CONTROLS ====================\n";
    std::cout << "  Mouse drag (L)  : Look around (yaw/pitch)\n";
    std::cout << "  Mouse wheel     : Change move speed\n";
    std::cout << "  W / S           : Move forward / backward\n";
    std::cout << "  A / D           : Strafe left / right\n";
    std::cout << "  Q / E           : Roll left / right\n";
    std::cout << "  Arrow Up/Down   : Move up / down\n";
    std::cout << "  P               : Toggle points display\n";
    std::cout << "  V               : Toggle RGB/Depth side panels\n";
    std::cout << "  F               : Toggle plane-based hole fill\n";
    std::cout << "  M               : Cycle mesh mode (FILL/WIREFRAME/OFF)\n";
    std::cout << "  [ / ]           : Adjust mesh edge factor\n";
    std::cout << "  - / =           : Adjust mesh pixel stride\n";
    std::cout << "  , / .           : Adjust hole patch perimeter\n";
    std::cout << "  H               : Toggle hole patching (small + plane)\n";
    std::cout << "  R               : Reset camera to default view\n";
    std::cout << "  T               : Toggle auto-rotate\n";
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

    resetCameraPose();

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
    glutSpecialFunc(specialFunc);
    glutIdleFunc(idleFunc);

    printControls();

    // Enter main loop
    glutMainLoop();

    return 0;
}
