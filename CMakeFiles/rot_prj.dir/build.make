# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.11

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aruchid/IDT_stitching

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aruchid/IDT_stitching

# Include any dependencies generated for this target.
include CMakeFiles/rot_prj.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/rot_prj.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/rot_prj.dir/flags.make

CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.o: rotation_overlap_test.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.o -c /home/aruchid/IDT_stitching/rotation_overlap_test.cpp

CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/rotation_overlap_test.cpp > CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.i

CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/rotation_overlap_test.cpp -o CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.s

CMakeFiles/rot_prj.dir/entrance.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/entrance.cpp.o: entrance.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/rot_prj.dir/entrance.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/entrance.cpp.o -c /home/aruchid/IDT_stitching/entrance.cpp

CMakeFiles/rot_prj.dir/entrance.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/entrance.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/entrance.cpp > CMakeFiles/rot_prj.dir/entrance.cpp.i

CMakeFiles/rot_prj.dir/entrance.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/entrance.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/entrance.cpp -o CMakeFiles/rot_prj.dir/entrance.cpp.s

CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.o: src/aruchid_pipeline.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.o -c /home/aruchid/IDT_stitching/src/aruchid_pipeline.cpp

CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/aruchid_pipeline.cpp > CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.i

CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/aruchid_pipeline.cpp -o CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.s

CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.o: src/aruchid_featuresfind.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.o -c /home/aruchid/IDT_stitching/src/aruchid_featuresfind.cpp

CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/aruchid_featuresfind.cpp > CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.i

CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/aruchid_featuresfind.cpp -o CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.s

CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.o: src/aruchid_get_homo.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.o -c /home/aruchid/IDT_stitching/src/aruchid_get_homo.cpp

CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/aruchid_get_homo.cpp > CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.i

CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/aruchid_get_homo.cpp -o CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.s

CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.o: src/aruchid_printpic.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.o -c /home/aruchid/IDT_stitching/src/aruchid_printpic.cpp

CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/aruchid_printpic.cpp > CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.i

CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/aruchid_printpic.cpp -o CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.s

CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.o: src/aruchid_rotation.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.o -c /home/aruchid/IDT_stitching/src/aruchid_rotation.cpp

CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/aruchid_rotation.cpp > CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.i

CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/aruchid_rotation.cpp -o CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.s

CMakeFiles/rot_prj.dir/src/overlap.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/overlap.cpp.o: src/overlap.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/rot_prj.dir/src/overlap.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/overlap.cpp.o -c /home/aruchid/IDT_stitching/src/overlap.cpp

CMakeFiles/rot_prj.dir/src/overlap.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/overlap.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/overlap.cpp > CMakeFiles/rot_prj.dir/src/overlap.cpp.i

CMakeFiles/rot_prj.dir/src/overlap.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/overlap.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/overlap.cpp -o CMakeFiles/rot_prj.dir/src/overlap.cpp.s

CMakeFiles/rot_prj.dir/src/backup_.cpp.o: CMakeFiles/rot_prj.dir/flags.make
CMakeFiles/rot_prj.dir/src/backup_.cpp.o: src/backup_.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/rot_prj.dir/src/backup_.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/rot_prj.dir/src/backup_.cpp.o -c /home/aruchid/IDT_stitching/src/backup_.cpp

CMakeFiles/rot_prj.dir/src/backup_.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/rot_prj.dir/src/backup_.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src/backup_.cpp > CMakeFiles/rot_prj.dir/src/backup_.cpp.i

CMakeFiles/rot_prj.dir/src/backup_.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/rot_prj.dir/src/backup_.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src/backup_.cpp -o CMakeFiles/rot_prj.dir/src/backup_.cpp.s

# Object files for target rot_prj
rot_prj_OBJECTS = \
"CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.o" \
"CMakeFiles/rot_prj.dir/entrance.cpp.o" \
"CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.o" \
"CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.o" \
"CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.o" \
"CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.o" \
"CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.o" \
"CMakeFiles/rot_prj.dir/src/overlap.cpp.o" \
"CMakeFiles/rot_prj.dir/src/backup_.cpp.o"

# External object files for target rot_prj
rot_prj_EXTERNAL_OBJECTS =

rot_prj: CMakeFiles/rot_prj.dir/rotation_overlap_test.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/entrance.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/aruchid_pipeline.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/aruchid_featuresfind.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/aruchid_get_homo.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/aruchid_printpic.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/aruchid_rotation.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/overlap.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/src/backup_.cpp.o
rot_prj: CMakeFiles/rot_prj.dir/build.make
rot_prj: /usr/local/opencv4/lib/libopencv_stitching.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_gapi.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_bgsegm.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_freetype.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_stereo.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_sfm.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_xphoto.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_hfs.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_rgbd.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_videostab.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_ccalib.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_bioinspired.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_aruco.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_saliency.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_structured_light.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_face.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_xobjdetect.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_dpm.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_dnn_objdetect.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_hdf.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_line_descriptor.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_superres.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_xfeatures2d.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_fuzzy.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_reg.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_img_hash.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_tracking.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_surface_matching.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_shape.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_viz.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_phase_unwrapping.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_photo.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_objdetect.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_optflow.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_ximgproc.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_video.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_calib3d.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_datasets.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_plot.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_text.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_ml.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_features2d.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_highgui.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_videoio.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_imgcodecs.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_dnn.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_imgproc.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_flann.so.4.0.0
rot_prj: /usr/local/opencv4/lib/libopencv_core.so.4.0.0
rot_prj: CMakeFiles/rot_prj.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Linking CXX executable rot_prj"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/rot_prj.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/rot_prj.dir/build: rot_prj

.PHONY : CMakeFiles/rot_prj.dir/build

CMakeFiles/rot_prj.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/rot_prj.dir/cmake_clean.cmake
.PHONY : CMakeFiles/rot_prj.dir/clean

CMakeFiles/rot_prj.dir/depend:
	cd /home/aruchid/IDT_stitching && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching/CMakeFiles/rot_prj.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/rot_prj.dir/depend

