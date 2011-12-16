cmake_minimum_required(VERSION 2.6)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(Eigen3 REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

set(BASE_SOURCE_FILES
    bbox/bb_cluster_confidence.cpp
    bbox/bb_distance.cpp
    bbox/bb_points.cpp
    bbox/bb_predict.cpp
    bbox/bb_scan.cpp
    img/ImageSource.cpp
    img/img_blur.cpp
    img/img_patch.cpp
    mex/bb_overlap.cpp
    mex/distance.cpp
    mex/fern.cpp
    mex/lk.cpp
    mex/warp.cpp
    tld/tld.cpp
    utils/mat2img.cpp
    utils/median.cpp
)

set(TLDC_SOURCE_FILES
    run_TLD.cpp
    tld/tldExample.cpp
)

set(PUBLIC_HEADER_FILES
    include/tld/tld.h
    include/tld/structs.h
    include/tld/tldconst.h
    include/tld/ImageSource.h
)

add_executable(tldc ${TLDC_SOURCE_FILES})

add_library(libtldc SHARED ${BASE_SOURCE_FILES})
set_target_properties(libtldc PROPERTIES OUTPUT_NAME "tldc")

target_link_libraries(tldc
    cv
    cxcore
    highgui
    libtldc
)

INSTALL(TARGETS tldc libtldc
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

INSTALL(FILES
    ${PUBLIC_HEADER_FILES}
    DESTINATION include/tld
)


add_custom_target(tags
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMAND ctags --fields=+S ${BASE_SOURCE_FILES} ${TLDC_SOURCE_FILES}
)
