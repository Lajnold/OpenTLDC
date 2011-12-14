CC := g++
INCLUDES := -I/usr/include/eigen3/
LIBS := -lcv -lhighgui -lcxcore
OPTS := -J3 -O2 -g -pipe -Wall -c -fmessage-length=0 -MMD -MP
CFLAGS :=
USRLIB := -L/usr/lib

OBJS := \
./run_TLD.o \
./bbox/bb_cluster_confidence.o \
./bbox/bb_distance.o \
./bbox/bb_points.o \
./bbox/bb_predict.o \
./bbox/bb_scan.o \
./img/ImageSource.o \
./img/img_blur.o \
./img/img_patch.o \
./mex/bb_overlap.o \
./mex/distance.o \
./mex/fern.o \
./mex/lk.o \
./mex/warp.o \
./tld/tldDetection.o \
./tld/tldDisplay.o \
./tld/tldExample.o \
./tld/tldGenerateFeatures.o \
./tld/tldGenerateNegativeData.o \
./tld/tldGeneratePositiveData.o \
./tld/tldGetPattern.o \
./tld/tldInit.o \
./tld/tldLearning.o \
./tld/tldNN.o \
./tld/tldPatch2Pattern.o \
./tld/tldProcessFrame.o \
./tld/tldSplitNegativeData.o \
./tld/tldTracking.o \
./tld/tldTrainNN.o \
./utils/mat2img.o \
./utils/median.o 

.PHONY: all clean

tldc: $(OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(CC) $(USRLIB) -o"tldc" $(OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

clean:
	-$(RM) $(OBJS) *.d **/*.d tldc
	-@echo ' '


%.o: %.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CC) $(CFLAGS) $(INCLUDES) $(OPTS) -MF"$*.d" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

-include $(OBJS:.o=.d)
