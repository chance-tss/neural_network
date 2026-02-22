NAME = my_torch_analyzer
GENERATOR = my_torch_generator

CC = g++
CFLAGS = -Wall -Wextra -Werror -std=c++20 -I./include

SRC_DIR = src
OBJ_DIR = obj
INC_DIR = include

# All source files
SRC_NN = $(wildcard $(SRC_DIR)/nn/*.cpp)
SRC_ANALYZER = $(wildcard $(SRC_DIR)/analyzer/*.cpp)

# Object files
OBJ_NN = $(SRC_NN:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJ_ANALYZER = $(SRC_ANALYZER:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Separate main objects
OBJ_MAIN = $(OBJ_DIR)/analyzer/main.o
OBJ_GENERATOR_MAIN = $(OBJ_DIR)/analyzer/generator_main.o

# Shared objects (everything except the two mains)
OBJ_SHARED = $(filter-out $(OBJ_MAIN) $(OBJ_GENERATOR_MAIN), $(OBJ_NN) $(OBJ_ANALYZER))

all: $(NAME) $(GENERATOR)

$(NAME): $(OBJ_SHARED) $(OBJ_MAIN)
	$(CC) $(OBJ_SHARED) $(OBJ_MAIN) -o $(NAME)

$(GENERATOR): $(OBJ_SHARED) $(OBJ_GENERATOR_MAIN)
	$(CC) $(OBJ_SHARED) $(OBJ_GENERATOR_MAIN) -o $(GENERATOR)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR)

fclean: clean
	rm -f $(NAME) $(GENERATOR)

re: fclean all

TEST_SRC = $(wildcard tests/*.cpp)
OBJ_NO_MAIN = $(filter-out $(OBJ_MAIN) $(OBJ_GENERATOR_MAIN), $(OBJ_NN) $(OBJ_ANALYZER))

tests: $(OBJ_NN) $(OBJ_ANALYZER)
	$(CC) $(CFLAGS) $(TEST_SRC) $(OBJ_NO_MAIN) -o run_tests
	./run_tests

.PHONY: all clean fclean re tests
