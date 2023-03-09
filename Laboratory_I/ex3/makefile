CC = gcc
OPT = -std=c99 -O3

BIN_FOLDER := bin
BATCH_OUT_FOLDER := outputs

MAIN    =  lab1_ex3.c

all: $(BIN_FOLDER)/lab1_ex3

debug: OPT += -DDEBUG -g
debug: all

$(BIN_FOLDER)/lab1_ex3: $(MAIN)
	@mkdir -p $(BIN_FOLDER)
	@mkdir -p $(BATCH_OUT_FOLDER)
	$(CC) $^ -o $@ $(OPT)

clean:
	rm -r $(BIN_FOLDER)

clean_batch_outputs:
	rm $(BATCH_OUT_FOLDER)/*
