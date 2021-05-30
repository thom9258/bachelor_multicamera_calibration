# (((FORCE_NO_BACKUP)))
COMPILER		= g++ -m64 -g -Wall -std=c++11
TARGET			= a.out
SOURCES			= main.cpp 

# compiler flags:
#  -m64 		forces use of 64 bit processing
#  -g    		adds debugging information to the executable file
#  -Wall 		turns on most, but not all, compiler warnings
#  -std=c++11 	compiler version

# OpenCV from source, kudos to:
# https://docs.opencv.org/master/d7/d9f/tutorial_linux_install.html 
OpenCV_include_PATHS 	= -I /usr/local/include/opencv4
OpenCV_lib_PATHS 		= -L /usr/local/lib
OpenCV_LIBS  			= -lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core  -lpthread   
OpenCV 					= $(OpenCV_lib_PATHS) $(OpenCV_LIBS) $(OpenCV_include_PATHS)

all: $(TARGET)

# To add external libraries, add call to end of sources
# the build target executable:
$(TARGET): $(TARGET)
	 $(COMPILER) -o $(TARGET) $(SOURCES) $(OpenCV)

clean:
	$(RM) $(TARGET)
