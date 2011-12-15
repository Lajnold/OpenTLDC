CC := g++
INCLUDES := -I/usr/include/eigen3/
LIBS := -lcv -lhighgui -lcxcore
OPTS := -O2 -g -pipe -Wall -c -fmessage-length=0 -MMD -MP
CFLAGS :=
USRLIB := -L/usr/lib
TAGS = ctags --fields=+S

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
./tld/tld.o \
./tld/tldExample.o \
./utils/mat2img.o \
./utils/median.o 

SRCS = $(OBJS:.o=.cpp)
HDRS = tld/*.h bbox/*.h img/*.h mex/*.h utils/*.h

.PHONY: all clean

tldc: $(OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(CC) $(USRLIB) -o"tldc" $(OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

tags: $(SRCS) $(HDRS)
	$(TAGS) $(SRCS) $(HDRS)

clean:
	-$(RM) $(OBJS) *.d **/*.d tldc
	-$(RM) tags
	-@echo ' '


%.o: %.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CC) $(CFLAGS) $(INCLUDES) $(OPTS) -MF"$*.d" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

-include $(OBJS:.o=.d)
