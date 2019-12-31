cmake_minimum_required( VERSION 2.8 )
project( imageBasics )

# 添加c++ 11标准支持
set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 寻找OpenCV库
find_package( OpenCV 4.0 REQUIRED )
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )
# 添加Eigen头文件
include_directories( "/usr/include/eigen3" )


set(CMAKE_CXX_STANDARD 11)
set(BUILD_DIR .)  #设置编译目录,也就是Makefile文件所在目录
message(${BUILD_DIR}) #打印目录路径
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
add_custom_target(demo COMMAND make -C ${BUILD_DIR}) #最关键的就是这句, 设置外部编译文件而不是使用CMakeLists.txt


add_executable(
		main_prj main.cpp
		entrance.cpp
		./stitch_connector.h
		./src/aruchid_pipeline.cpp ./include/aruchid_pipeline.h
		./src/aruchid_featuresfind.cpp ./include/aruchid_featuresfind.hpp
		./src/aruchid_get_homo.cpp  ./include/aruchid_get_homo.h
		./src/aruchid_printpic.cpp  ./include/aruchid_printpic.h
#        src/backup.cpp
)
target_link_libraries( main_prj ${OpenCV_LIBS} )

#
#add_executable(
#		rot_prj rotation_overlap_test.cpp
#		entrance.cpp
#		./stitch_connector.h
#		./src/aruchid_pipeline.cpp ./include/aruchid_pipeline.h
#		./src/aruchid_featuresfind.cpp ./include/aruchid_featuresfind.h
#		./src/aruchid_get_homo.cpp  ./include/aruchid_get_homo.h
#		./src/aruchid_printpic.cpp  ./include/aruchid_printpic.h
#		./src/aruchid_rotation.cpp  ./include/aruchid_rotation.h
#		./src/overlap.cpp ./include/overlap.hpp
#        src/backup_.cpp)
#
#target_link_libraries( rot_prj ${OpenCV_LIBS} )

#add_executable(cam takephoto.cpp
##		./src/backup.cpp
#		./src/aruchid_featuresfind.cpp ./include/aruchid_featuresfind.h
#		./src/overlap.cpp ./include/overlap.hpp src/overlap_optflow.cpp)
#target_link_libraries( cam ${OpenCV_LIBS} )
#
#add_executable(optl src_out/try_optflow.cpp
#		#		./src/backup.cpp
#		./src/aruchid_featuresfind.cpp ./include/aruchid_featuresfind.h
#		./src/overlap.cpp ./include/overlap.hpp src_out/split.cpp)
#target_link_libraries( optl ${OpenCV_LIBS} )
###
#add_executable(split src_out/split.cpp
#		#		./src/backup.cpp
#)
#target_link_libraries( split ${OpenCV_LIBS} )
#


add_executable(overlapoft src/overlap_optflow.cpp
		#		./src/backup.cpp
#		entrance.cpp
#		./stitch_connector.h
#		./src/aruchid_pipeline.cpp ./include/aruchid_pipeline.h
##		./src/aruchid_featuresfind.cpp ./include/aruchid_featuresfind.hpp
#		./src/aruchid_get_homo.cpp  ./include/aruchid_get_homo.h
#		./src/aruchid_printpic.cpp  ./include/aruchid_printpic.h
		./src_out/optical.hpp ./src_out/optical.cpp
		)
target_link_libraries( overlapoft ${OpenCV_LIBS} )

add_executable(
		retrieve
		src/retrieve.cpp
	./include/aruchid_featuresfind.hpp
	./src/aruchid_featuresfind.cpp)
target_link_libraries( retrieve ${OpenCV_LIBS} )



add_executable(
		mobile_app
		./app_test.cpp
		./capture_stitching_mobile.cpp
		./capture_stitching_mobile.hpp
)
target_link_libraries( mobile_app ${OpenCV_LIBS} )

##

#add_executable(optl2 opt_2.cpp
#		#		./src/backup.cpp
#		./src/aruchid_featuresfind.cpp ./include/aruchid_featuresfind.h
#		src/opt_overlap.cpp src/opt_overlap.h)
#target_link_libraries( optl2 ${OpenCV_LIBS} )
#
#
#add_executable(mouse mousetrack.cpp)
#target_link_libraries( mouse ${OpenCV_LIBS} )