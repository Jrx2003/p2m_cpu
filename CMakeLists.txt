cmake_minimum_required(VERSION 3.5)
project(P2M_RealtimeMeshViewer)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# For multi-config generators (Visual Studio)
foreach(OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/bin)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} ${CMAKE_BINARY_DIR}/lib)
endforeach()

# Find OrbbecSDK (relative to project root)
set(OrbbecSDK_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(OrbbecSDK_INCLUDE_DIRS "${OrbbecSDK_DIR}/include")

if(WIN32)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(OrbbecSDK_LIB_DIR "${OrbbecSDK_DIR}/lib/win_x64")
    else()
        set(OrbbecSDK_LIB_DIR "${OrbbecSDK_DIR}/lib/win_x86")
    endif()
elseif(UNIX AND NOT APPLE)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(OrbbecSDK_LIB_DIR "${OrbbecSDK_DIR}/lib/linux_x64")
    else()
        set(OrbbecSDK_LIB_DIR "${OrbbecSDK_DIR}/lib/arm64")
    endif()
endif()

set(OrbbecSDK_LIBS "${OrbbecSDK_LIB_DIR}/OrbbecSDK.lib")

message(STATUS "OrbbecSDK Include: ${OrbbecSDK_INCLUDE_DIRS}")
message(STATUS "OrbbecSDK Library: ${OrbbecSDK_LIBS}")

# Find required packages
find_package(PCL 1.8 REQUIRED COMPONENTS common io surface)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

message(STATUS "PCL version: ${PCL_VERSION}")
message(STATUS "PCL include dirs: ${PCL_INCLUDE_DIRS}")
message(STATUS "PCL libraries: ${PCL_LIBRARIES}")

# Add executable
add_executable(${PROJECT_NAME}
    src/RealtimeMeshViewer.cpp
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${OrbbecSDK_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIRS}
    ${GLEW_INCLUDE_DIRS}
    ${GLUT_INCLUDE_DIRS}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OrbbecSDK_LIBS}
    ${PCL_LIBRARIES}
    ${OPENGL_LIBRARIES}
    ${GLEW_LIBRARIES}
    ${GLUT_LIBRARIES}
)

# Add PCL definitions
target_compile_definitions(${PROJECT_NAME} PRIVATE
    ${PCL_DEFINITIONS}
)

# Link directories
link_directories(${PCL_LIBRARY_DIRS})

# Copy OrbbecSDK DLLs to output directory after build
if(WIN32)
    # Use generator expression to get the correct output directory for all configurations
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OrbbecSDK_LIB_DIR}/OrbbecSDK.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/OrbbecSDK.dll"
        COMMENT "Copying OrbbecSDK.dll to output directory"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OrbbecSDK_LIB_DIR}/ob_usb.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/ob_usb.dll"
        COMMENT "Copying ob_usb.dll to output directory"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OrbbecSDK_LIB_DIR}/depthengine_2_0.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/depthengine_2_0.dll"
        COMMENT "Copying depthengine_2_0.dll to output directory"
    )
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OrbbecSDK_LIB_DIR}/live555.dll"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/live555.dll"
        COMMENT "Copying live555.dll to output directory"
    )
endif()

# Installation rules
install(TARGETS ${PROJECT_NAME}
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install OrbbecSDK DLLs
if(WIN32)
    install(DIRECTORY ${OrbbecSDK_LIB_DIR}/
        DESTINATION bin
        FILES_MATCHING PATTERN "*.dll"
    )
endif()

# Print build configuration
message(STATUS "==================================")
message(STATUS "  P2M Project Configuration")
message(STATUS "==================================")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output Directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "==================================")

