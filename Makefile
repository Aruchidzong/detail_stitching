# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/home/aruchid/aruchid/tools/clion-2019.2.5/bin/cmake/linux/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/home/aruchid/aruchid/tools/clion-2019.2.5/bin/cmake/linux/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/aruchid/IDT_stitching/CMakeFiles /home/aruchid/IDT_stitching/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/aruchid/IDT_stitching/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named main_prj

# Build rule for target.
main_prj: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 main_prj
.PHONY : main_prj

# fast build rule for target.
main_prj/fast:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/build
.PHONY : main_prj/fast

#=============================================================================
# Target rules for targets named overlapoft

# Build rule for target.
overlapoft: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 overlapoft
.PHONY : overlapoft

# fast build rule for target.
overlapoft/fast:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/build
.PHONY : overlapoft/fast

#=============================================================================
# Target rules for targets named retrieve

# Build rule for target.
retrieve: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 retrieve
.PHONY : retrieve

# fast build rule for target.
retrieve/fast:
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/build
.PHONY : retrieve/fast

#=============================================================================
# Target rules for targets named demo

# Build rule for target.
demo: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 demo
.PHONY : demo

# fast build rule for target.
demo/fast:
	$(MAKE) -f CMakeFiles/demo.dir/build.make CMakeFiles/demo.dir/build
.PHONY : demo/fast

#=============================================================================
# Target rules for targets named mobile_app

# Build rule for target.
mobile_app: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 mobile_app
.PHONY : mobile_app

# fast build rule for target.
mobile_app/fast:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/build
.PHONY : mobile_app/fast

app_test.o: app_test.cpp.o

.PHONY : app_test.o

# target to build an object file
app_test.cpp.o:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/app_test.cpp.o
.PHONY : app_test.cpp.o

app_test.i: app_test.cpp.i

.PHONY : app_test.i

# target to preprocess a source file
app_test.cpp.i:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/app_test.cpp.i
.PHONY : app_test.cpp.i

app_test.s: app_test.cpp.s

.PHONY : app_test.s

# target to generate assembly for a file
app_test.cpp.s:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/app_test.cpp.s
.PHONY : app_test.cpp.s

capture_stitching_mobile.o: capture_stitching_mobile.cpp.o

.PHONY : capture_stitching_mobile.o

# target to build an object file
capture_stitching_mobile.cpp.o:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/capture_stitching_mobile.cpp.o
.PHONY : capture_stitching_mobile.cpp.o

capture_stitching_mobile.i: capture_stitching_mobile.cpp.i

.PHONY : capture_stitching_mobile.i

# target to preprocess a source file
capture_stitching_mobile.cpp.i:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/capture_stitching_mobile.cpp.i
.PHONY : capture_stitching_mobile.cpp.i

capture_stitching_mobile.s: capture_stitching_mobile.cpp.s

.PHONY : capture_stitching_mobile.s

# target to generate assembly for a file
capture_stitching_mobile.cpp.s:
	$(MAKE) -f CMakeFiles/mobile_app.dir/build.make CMakeFiles/mobile_app.dir/capture_stitching_mobile.cpp.s
.PHONY : capture_stitching_mobile.cpp.s

entrance.o: entrance.cpp.o

.PHONY : entrance.o

# target to build an object file
entrance.cpp.o:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/entrance.cpp.o
.PHONY : entrance.cpp.o

entrance.i: entrance.cpp.i

.PHONY : entrance.i

# target to preprocess a source file
entrance.cpp.i:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/entrance.cpp.i
.PHONY : entrance.cpp.i

entrance.s: entrance.cpp.s

.PHONY : entrance.s

# target to generate assembly for a file
entrance.cpp.s:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/entrance.cpp.s
.PHONY : entrance.cpp.s

main.o: main.cpp.o

.PHONY : main.o

# target to build an object file
main.cpp.o:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/main.cpp.o
.PHONY : main.cpp.o

main.i: main.cpp.i

.PHONY : main.i

# target to preprocess a source file
main.cpp.i:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/main.cpp.i
.PHONY : main.cpp.i

main.s: main.cpp.s

.PHONY : main.s

# target to generate assembly for a file
main.cpp.s:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/main.cpp.s
.PHONY : main.cpp.s

src/aruchid_featuresfind.o: src/aruchid_featuresfind.cpp.o

.PHONY : src/aruchid_featuresfind.o

# target to build an object file
src/aruchid_featuresfind.cpp.o:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_featuresfind.cpp.o
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/src/aruchid_featuresfind.cpp.o
.PHONY : src/aruchid_featuresfind.cpp.o

src/aruchid_featuresfind.i: src/aruchid_featuresfind.cpp.i

.PHONY : src/aruchid_featuresfind.i

# target to preprocess a source file
src/aruchid_featuresfind.cpp.i:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_featuresfind.cpp.i
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/src/aruchid_featuresfind.cpp.i
.PHONY : src/aruchid_featuresfind.cpp.i

src/aruchid_featuresfind.s: src/aruchid_featuresfind.cpp.s

.PHONY : src/aruchid_featuresfind.s

# target to generate assembly for a file
src/aruchid_featuresfind.cpp.s:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_featuresfind.cpp.s
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/src/aruchid_featuresfind.cpp.s
.PHONY : src/aruchid_featuresfind.cpp.s

src/aruchid_get_homo.o: src/aruchid_get_homo.cpp.o

.PHONY : src/aruchid_get_homo.o

# target to build an object file
src/aruchid_get_homo.cpp.o:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_get_homo.cpp.o
.PHONY : src/aruchid_get_homo.cpp.o

src/aruchid_get_homo.i: src/aruchid_get_homo.cpp.i

.PHONY : src/aruchid_get_homo.i

# target to preprocess a source file
src/aruchid_get_homo.cpp.i:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_get_homo.cpp.i
.PHONY : src/aruchid_get_homo.cpp.i

src/aruchid_get_homo.s: src/aruchid_get_homo.cpp.s

.PHONY : src/aruchid_get_homo.s

# target to generate assembly for a file
src/aruchid_get_homo.cpp.s:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_get_homo.cpp.s
.PHONY : src/aruchid_get_homo.cpp.s

src/aruchid_pipeline.o: src/aruchid_pipeline.cpp.o

.PHONY : src/aruchid_pipeline.o

# target to build an object file
src/aruchid_pipeline.cpp.o:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_pipeline.cpp.o
.PHONY : src/aruchid_pipeline.cpp.o

src/aruchid_pipeline.i: src/aruchid_pipeline.cpp.i

.PHONY : src/aruchid_pipeline.i

# target to preprocess a source file
src/aruchid_pipeline.cpp.i:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_pipeline.cpp.i
.PHONY : src/aruchid_pipeline.cpp.i

src/aruchid_pipeline.s: src/aruchid_pipeline.cpp.s

.PHONY : src/aruchid_pipeline.s

# target to generate assembly for a file
src/aruchid_pipeline.cpp.s:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_pipeline.cpp.s
.PHONY : src/aruchid_pipeline.cpp.s

src/aruchid_printpic.o: src/aruchid_printpic.cpp.o

.PHONY : src/aruchid_printpic.o

# target to build an object file
src/aruchid_printpic.cpp.o:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_printpic.cpp.o
.PHONY : src/aruchid_printpic.cpp.o

src/aruchid_printpic.i: src/aruchid_printpic.cpp.i

.PHONY : src/aruchid_printpic.i

# target to preprocess a source file
src/aruchid_printpic.cpp.i:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_printpic.cpp.i
.PHONY : src/aruchid_printpic.cpp.i

src/aruchid_printpic.s: src/aruchid_printpic.cpp.s

.PHONY : src/aruchid_printpic.s

# target to generate assembly for a file
src/aruchid_printpic.cpp.s:
	$(MAKE) -f CMakeFiles/main_prj.dir/build.make CMakeFiles/main_prj.dir/src/aruchid_printpic.cpp.s
.PHONY : src/aruchid_printpic.cpp.s

src/overlap_optflow.o: src/overlap_optflow.cpp.o

.PHONY : src/overlap_optflow.o

# target to build an object file
src/overlap_optflow.cpp.o:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/src/overlap_optflow.cpp.o
.PHONY : src/overlap_optflow.cpp.o

src/overlap_optflow.i: src/overlap_optflow.cpp.i

.PHONY : src/overlap_optflow.i

# target to preprocess a source file
src/overlap_optflow.cpp.i:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/src/overlap_optflow.cpp.i
.PHONY : src/overlap_optflow.cpp.i

src/overlap_optflow.s: src/overlap_optflow.cpp.s

.PHONY : src/overlap_optflow.s

# target to generate assembly for a file
src/overlap_optflow.cpp.s:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/src/overlap_optflow.cpp.s
.PHONY : src/overlap_optflow.cpp.s

src/retrieve.o: src/retrieve.cpp.o

.PHONY : src/retrieve.o

# target to build an object file
src/retrieve.cpp.o:
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/src/retrieve.cpp.o
.PHONY : src/retrieve.cpp.o

src/retrieve.i: src/retrieve.cpp.i

.PHONY : src/retrieve.i

# target to preprocess a source file
src/retrieve.cpp.i:
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/src/retrieve.cpp.i
.PHONY : src/retrieve.cpp.i

src/retrieve.s: src/retrieve.cpp.s

.PHONY : src/retrieve.s

# target to generate assembly for a file
src/retrieve.cpp.s:
	$(MAKE) -f CMakeFiles/retrieve.dir/build.make CMakeFiles/retrieve.dir/src/retrieve.cpp.s
.PHONY : src/retrieve.cpp.s

src_out/optical.o: src_out/optical.cpp.o

.PHONY : src_out/optical.o

# target to build an object file
src_out/optical.cpp.o:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/src_out/optical.cpp.o
.PHONY : src_out/optical.cpp.o

src_out/optical.i: src_out/optical.cpp.i

.PHONY : src_out/optical.i

# target to preprocess a source file
src_out/optical.cpp.i:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/src_out/optical.cpp.i
.PHONY : src_out/optical.cpp.i

src_out/optical.s: src_out/optical.cpp.s

.PHONY : src_out/optical.s

# target to generate assembly for a file
src_out/optical.cpp.s:
	$(MAKE) -f CMakeFiles/overlapoft.dir/build.make CMakeFiles/overlapoft.dir/src_out/optical.cpp.s
.PHONY : src_out/optical.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... main_prj"
	@echo "... overlapoft"
	@echo "... retrieve"
	@echo "... demo"
	@echo "... mobile_app"
	@echo "... edit_cache"
	@echo "... app_test.o"
	@echo "... app_test.i"
	@echo "... app_test.s"
	@echo "... capture_stitching_mobile.o"
	@echo "... capture_stitching_mobile.i"
	@echo "... capture_stitching_mobile.s"
	@echo "... entrance.o"
	@echo "... entrance.i"
	@echo "... entrance.s"
	@echo "... main.o"
	@echo "... main.i"
	@echo "... main.s"
	@echo "... src/aruchid_featuresfind.o"
	@echo "... src/aruchid_featuresfind.i"
	@echo "... src/aruchid_featuresfind.s"
	@echo "... src/aruchid_get_homo.o"
	@echo "... src/aruchid_get_homo.i"
	@echo "... src/aruchid_get_homo.s"
	@echo "... src/aruchid_pipeline.o"
	@echo "... src/aruchid_pipeline.i"
	@echo "... src/aruchid_pipeline.s"
	@echo "... src/aruchid_printpic.o"
	@echo "... src/aruchid_printpic.i"
	@echo "... src/aruchid_printpic.s"
	@echo "... src/overlap_optflow.o"
	@echo "... src/overlap_optflow.i"
	@echo "... src/overlap_optflow.s"
	@echo "... src/retrieve.o"
	@echo "... src/retrieve.i"
	@echo "... src/retrieve.s"
	@echo "... src_out/optical.o"
	@echo "... src_out/optical.i"
	@echo "... src_out/optical.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system
