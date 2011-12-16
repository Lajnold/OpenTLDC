CC := g++
INCLUDES := -I/usr/include/eigen3/ -I include/
OPTS := -O2 -g -pipe -Wall -c -MMD -MP
CFLAGS :=
LDFLAGS := -L/usr/lib
LIBS := -lcv -lhighgui -lcxcore
TAGS = ctags --fields=+S
PREFIX = /usr/local
CP = /bin/cp

ifeq ($(MAKECMDGOALS), libtldc)
	OPTS := $(OPTS) -fPIC
endif

BASE_OBJS := \
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
./utils/mat2img.o \
./utils/median.o 

TLDC_OBJS := \
./run_TLD.o \
./tld/tldExample.o

ALL_OBJS := $(BASE_OBJS) $(TLDC_OBJS)

SRCS = $(ALL_OBJS:.o=.cpp)
HDRS = tld/*.h bbox/*.h img/*.h mex/*.h utils/*.h

.PHONY: all clean

tldc: $(BASE_OBJS) $(TLDC_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	$(CC) $(LDFLAGS) -o"tldc" $(BASE_OBJS) $(TLDC_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

libtldc: libtldc.so
libtldc.so: $(BASE_OBJS)
	@echo 'Building library: $@'
	$(CC) $(LDFLAGS) -shared -Wl,-soname,libtldc.so -o libtldc.so $(BASE_OBJS) $(LIBS)

tags: $(SRCS) $(HDRS)
	$(TAGS) $(SRCS) $(HDRS)

install: libtldc.so
	$(RM) $(PREFIX)/lib/libtldc.so && cp libtldc.so $(PREFIX)/lib/
	@mkdir -p $(PREFIX)/include/tld
	$(CP) include/tld/*.h $(PREFIX)/include/tld/

clean:
	-$(RM) $(ALL_OBJS) *.d **/*.d tldc
	-$(RM) tags
	-@echo ' '

%.o: %.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CC) $(CFLAGS) $(INCLUDES) $(OPTS) -MF"$*.d" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

-include $(ALL_OBJS:.o=.d)
