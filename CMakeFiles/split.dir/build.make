# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /home/aruchid/aruchid/tools/clion-2019.2.5/bin/cmake/linux/bin/cmake

# The command to remove a file.
RM = /home/aruchid/aruchid/tools/clion-2019.2.5/bin/cmake/linux/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/aruchid/IDT_stitching

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/aruchid/IDT_stitching

# Include any dependencies generated for this target.
include CMakeFiles/split.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/split.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/split.dir/flags.make

CMakeFiles/split.dir/src_out/split.cpp.o: CMakeFiles/split.dir/flags.make
CMakeFiles/split.dir/src_out/split.cpp.o: src_out/split.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/split.dir/src_out/split.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/split.dir/src_out/split.cpp.o -c /home/aruchid/IDT_stitching/src_out/split.cpp

CMakeFiles/split.dir/src_out/split.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/split.dir/src_out/split.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/aruchid/IDT_stitching/src_out/split.cpp > CMakeFiles/split.dir/src_out/split.cpp.i

CMakeFiles/split.dir/src_out/split.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/split.dir/src_out/split.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/aruchid/IDT_stitching/src_out/split.cpp -o CMakeFiles/split.dir/src_out/split.cpp.s

# Object files for target split
split_OBJECTS = \
"CMakeFiles/split.dir/src_out/split.cpp.o"

# External object files for target split
split_EXTERNAL_OBJECTS =

split: CMakeFiles/split.dir/src_out/split.cpp.o
split: CMakeFiles/split.dir/build.make
split: /usr/local/opencv4/lib/libopencv_stitching.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_gapi.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_bgsegm.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_freetype.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_stereo.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_sfm.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_xphoto.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_hfs.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_rgbd.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_videostab.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_ccalib.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_bioinspired.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_aruco.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_saliency.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_structured_light.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_face.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_xobjdetect.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_dpm.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_dnn_objdetect.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_hdf.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_line_descriptor.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_superres.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_xfeatures2d.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_fuzzy.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_reg.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_img_hash.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_tracking.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_surface_matching.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_shape.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_viz.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_phase_unwrapping.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_photo.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_objdetect.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_optflow.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_ximgproc.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_video.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_calib3d.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_datasets.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_plot.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_text.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_ml.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_features2d.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_highgui.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_videoio.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_imgcodecs.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_dnn.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_imgproc.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_flann.so.4.0.0
split: /usr/local/opencv4/lib/libopencv_core.so.4.0.0
split: CMakeFiles/split.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/aruchid/IDT_stitching/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable split"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/split.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/split.dir/build: split

.PHONY : CMakeFiles/split.dir/build

CMakeFiles/split.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/split.dir/cmake_clean.cmake
.PHONY : CMakeFiles/split.dir/clean

CMakeFiles/split.dir/depend:
	cd /home/aruchid/IDT_stitching && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching /home/aruchid/IDT_stitching/CMakeFiles/split.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/split.dir/depend
