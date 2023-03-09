CC = c++
OPT = -std=c++14 -O3

# set main here
MAIN = lab1.c

##################################################################################################################
TARGETDIR   := bin

all: $(TARGETDIR)/lab1

debug: OPT += -DDEBUG -g
debug: all

$(TARGETDIR)/lab1: ${MAIN}
	@mkdir -p $(@D)
	$(CC) $^ -o $@ $(INCLUDE) $(OPT)

clean:
	rm -r $(TARGETDIR)/
