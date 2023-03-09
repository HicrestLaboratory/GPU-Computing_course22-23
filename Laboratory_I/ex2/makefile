# create a makefile variable named CC for your C/C++ compiler (es gcc, c++, ... )

# create a makefile variable named OPT with your favorite C flags (at least with -std=c99 -O3)


BIN_FOLDER := bin
OBJ_FOLDER := obj
SRC_FOLDER := src
BATCH_OUT_FOLDER := outputs

MAIN    =  lab1_ex2.c
OBJECTS = $(OBJ_FOLDER)/lab1_ex2_lib.o

all: $(BIN_FOLDER)/lab1_ex2

#create the target 'bin/lab1_ex2' which needs 'lab1_ex2.c' and 'lab1_ex2_lib.o', generate the folder 'bin' and compile lab1_ex2.c using CC and OPT




$(OBJ_FOLDER)/lab1_ex2_lib.o: $(SRC_FOLDER)/lab1_ex2_lib.c
	@mkdir -p $(BIN_FOLDER) $(OBJ_FOLDER) $(BATCH_OUT_FOLDER)
	$(CC) -c $(SRC_FOLDER)/lab1_ex2_lib.c -o $@ $(OPT)


#create the target 'clean' which recursively remove the folders 'bin' and 'obj'


clean_batch_outputs:
	rm $(BATCH_OUT_FOLDER)/*
