# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils

# Include any dependencies generated for this target.
include CMakeFiles/point_utils.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/point_utils.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/point_utils.dir/flags.make

CMakeFiles/point_utils.dir/point_utils.cpp.o: CMakeFiles/point_utils.dir/flags.make
CMakeFiles/point_utils.dir/point_utils.cpp.o: point_utils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/point_utils.dir/point_utils.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/point_utils.dir/point_utils.cpp.o -c /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils/point_utils.cpp

CMakeFiles/point_utils.dir/point_utils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/point_utils.dir/point_utils.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils/point_utils.cpp > CMakeFiles/point_utils.dir/point_utils.cpp.i

CMakeFiles/point_utils.dir/point_utils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/point_utils.dir/point_utils.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils/point_utils.cpp -o CMakeFiles/point_utils.dir/point_utils.cpp.s

CMakeFiles/point_utils.dir/point_utils.cpp.o.requires:

.PHONY : CMakeFiles/point_utils.dir/point_utils.cpp.o.requires

CMakeFiles/point_utils.dir/point_utils.cpp.o.provides: CMakeFiles/point_utils.dir/point_utils.cpp.o.requires
	$(MAKE) -f CMakeFiles/point_utils.dir/build.make CMakeFiles/point_utils.dir/point_utils.cpp.o.provides.build
.PHONY : CMakeFiles/point_utils.dir/point_utils.cpp.o.provides

CMakeFiles/point_utils.dir/point_utils.cpp.o.provides.build: CMakeFiles/point_utils.dir/point_utils.cpp.o


# Object files for target point_utils
point_utils_OBJECTS = \
"CMakeFiles/point_utils.dir/point_utils.cpp.o"

# External object files for target point_utils
point_utils_EXTERNAL_OBJECTS =

libpoint_utils.so: CMakeFiles/point_utils.dir/point_utils.cpp.o
libpoint_utils.so: CMakeFiles/point_utils.dir/build.make
libpoint_utils.so: /home/ign.fr/drobert/anaconda3/envs/spt/lib/libboost_numpy38.so.1.73.0
libpoint_utils.so: /home/ign.fr/drobert/anaconda3/envs/spt/lib/libpython3.8.so
libpoint_utils.so: /home/ign.fr/drobert/anaconda3/envs/spt/lib/libboost_python38.so.1.73.0
libpoint_utils.so: CMakeFiles/point_utils.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libpoint_utils.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/point_utils.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/point_utils.dir/build: libpoint_utils.so

.PHONY : CMakeFiles/point_utils.dir/build

CMakeFiles/point_utils.dir/requires: CMakeFiles/point_utils.dir/point_utils.cpp.o.requires

.PHONY : CMakeFiles/point_utils.dir/requires

CMakeFiles/point_utils.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/point_utils.dir/cmake_clean.cmake
.PHONY : CMakeFiles/point_utils.dir/clean

CMakeFiles/point_utils.dir/depend:
	cd /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils /home/ign.fr/drobert-admin/projects/superpoint_transformer/superpoint_transformer/partition/utils/CMakeFiles/point_utils.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/point_utils.dir/depend

