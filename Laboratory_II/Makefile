CUDA_HOME=/usr/local/cuda-12.1
NVCC = ${CUDA_HOME}/bin/nvcc
CC = c++
NVCC_FLAG = -std=c++14
GPU_ARCH = -arch=sm_70 -m64
LIBS = -L${CUDA_HOME}/lib64
OPT = -std=c++14 -O3
#OPT = -O3 -g


# set main here
MAIN = lab2_ex1.cu

##################################################################################################################
BUILDDIR    := obj
TARGETDIR   := bin

all: $(TARGETDIR)/lab2_ex1

debug: OPT += -DDEBUG -g
debug: NVCC_FLAG += -G
debug: all

$(TARGETDIR)/lab2_ex1: ${MAIN}
	@mkdir -p $(@D)
	$(NVCC) $^ -o $@ $(GPU_ARCH) $(NVCC_FLAG) $(LIBS) $(OPT)


clean:
# 	rm $(BUILDDIR)/*.o $(TARGETDIR)/*
	rm -r $(TARGETDIR)
