# This makefile is intended for the GNU C compiler.
# Your code must compile (with GCC) with the given CFLAGS.
# You may experiment with the OPT variable to invoke additional compiler options
HOST = $(shell hostname)
BANG = $(shell hostname | grep ccom-bang | wc -c)
BANG-COMPUTE = $(shell hostname | grep compute | wc -c)
STAMPEDE = $(shell hostname | grep stampede | wc -c)
AMAZON = $(shell hostname | grep 'ip-' | wc -c)

ifneq ($(STAMPEDE), 0)
multi := 1
NO_BLAS = 0
# PUB = /home1/00660/tg458182/cse262-wi15
include $(PUB)/Arch/arch.intel-mkl
else
ifneq ($(BANG), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
else
ifneq ($(BANG-COMPUTE), 0)
atlas := 1
multi := 0
NO_BLAS = 1
include $(PUB)/Arch/arch.gnu_c99.generic
else
ifneq ($(AMAZON), 0)
atlas := 1
multi := 0
NO_BLAS = 1
amazon := 1
include $(PUB)/Arch/arch.gnu_c99.generic
endif
endif
endif
endif

WARNINGS += -Wall -pedantic

# If you want to copy data blocks to contiguous storage
# This applies to the hand code version
ifeq ($(copy), 1)
    C++FLAGS += -DCOPY
    CFLAGS += -DCOPY
endif


# If you want to use restrict pointers, make restrict=1
# This applies to the hand code version
ifeq ($(restrict), 1)
    C++FLAGS += -D__RESTRICT
    CFLAGS += -D__RESTRICT
# ifneq ($(CARVER), 0)
#    C++FLAGS += -restrict
#     CFLAGS += -restrict
# endif
endif

ifeq ($(NO_BLAS), 1)
    C++FLAGS += -DNO_BLAS
    CFLAGS += -DNO_BLAS
endif

MY_OPT = -O4 -mfpmath=sse -mno-align-double -march=core2

ifeq ($(amazon), 1)
    MY_OPT += -mavx
endif

OPTIMIZATION = $(MY_OPT)

targets = benchmark-naive benchmark-blocked benchmark-blas
objects = benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o
UTIL   = wall_time.o cmdLine.o


ifeq ($(amazon), 1)
    targets += benchmark-avx
    objects += avx-dgemm-blocked.o
endif

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : benchmark.o dgemm-naive.o  $(UTIL)
	$(CC) -g -o $@ $^ $(LDLIBS)
benchmark-blocked : benchmark.o dgemm-blocked.o $(UTIL)
	$(CC) -g -o $@ $^ $(LDLIBS)
benchmark-blas : benchmark.o dgemm-blas.o $(UTIL)
	$(CC) -g -o $@ $^ $(LDLIBS)
benchmark-avx : benchmark.o avx-dgemm-blocked.o $(UTIL)
	$(CC) -g -o $@ $^ $(LDLIBS)

%.o : %.c
	$(CC) -c -g $(CFLAGS) $<


.PHONY : clean
clean:
	rm -f $(targets) $(objects) $(UTIL) core
