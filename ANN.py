'''
Sebastian Ballesteros 
This is an implementation of an Artificial Neural Netwoek trained using a
Genetic Algorithm to solve a diabetes classification problem.
'''

'''
Algorithm Definitions:
population_weights_matrix - array of weights representated by matrices, mainly for ANN operations
population_weights_vector - array of weights represented by vectos, mainly for GA operations
fitness - array of fitness values for each member of the population (each chromosome)
best_weights - set of weights with the highest fitness found
train_in_data - the 9 inputs of each sample represented in a matrix
train_out_data - the 2 outputs of each sample represented in a matrix
'''

############################## IMPORTS  ########################################
import pandas
import random
import math
import sys
import numpy as np
from matplotlib import pyplot as plt

########################## GLOBAL CONSTANTS ###################################
PATTERN_NUM = 768
TRAIN_DATA_PERCENTAGE = 0.75
INPUT_NUMBER = 9 #8 + bias
OUTPUT_NUMBER = 1
COLUMNS = INPUT_NUMBER + OUTPUT_NUMBER
HL_NUMBER = 1  #change if want more hidden layers
HL1_NEURONS = 10
OUTPUT_NEURONS = OUTPUT_NUMBER
TRAIN_ROWS = math.ceil(TRAIN_DATA_PERCENTAGE*PATTERN_NUM)
TEST_ROWS = PATTERN_NUM - TRAIN_ROWS
GET_VECTOR_CSV = True
POPULATION_SIZE = 50
if (GET_VECTOR_CSV == True): GENERATIONS = 10
else: GENERATIONS = 500
NUM_PARENTS_MATING = 20
CROSSOVER_POINT = int(((INPUT_NUMBER*HL1_NEURONS)+(HL1_NEURONS*OUTPUT_NUMBER))/2)
MUTATION_PERCENT = 50

########################### GLOBAL VARIABLES ##################################
population_weights_matrix = []
population_weights_vector = []
best_weights = []
best_weights_vector = []
predictions = []
fitness = []
accuracies = np.empty(shape=(GENERATIONS))
best_accuracy = 0
data_set = pandas.read_csv('diabetes.csv')
train_in_data = data_set.iloc[0:TRAIN_ROWS, 0:INPUT_NUMBER].values
train_out_data = data_set.iloc[0:TRAIN_ROWS, INPUT_NUMBER:COLUMNS].values
test_in_data = data_set.iloc[TRAIN_ROWS:PATTERN_NUM, 0:INPUT_NUMBER].values
test_out_data = data_set.iloc[TRAIN_ROWS:PATTERN_NUM, INPUT_NUMBER:COLUMNS].values

###########################  HELPER FUNCTIONS ##################################

# returns a value between 0 and 1, activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-1*x))

def hard_limiter(x):
    for (i,item) in enumerate(x):
        if(item>0):
            x[i]=1
        else:
            x[i]=0
    return x

def find_correct_predictions(predictions, real):
    num_correct = 0
    for i in range((len(predictions))):
        if(predictions[i][0]  == real[i][0]):
            num_correct += 1
    return num_correct


def matrix_to_vector(matrix_weights):
    vector_weights = []
    #loop through population (i.e population size)
    for chromosome in range(POPULATION_SIZE):
        current_chromosome = []
        #loop through the layers
        for layer in range(HL_NUMBER+1):
            #reshape the matrix in the layer to be 1D
            current_layer = np.reshape(matrix_weights[chromosome,layer], newshape=(matrix_weights[chromosome, layer].size))
            #use extend instead of append to create 1D vector for each chromosome
            current_chromosome.extend(current_layer)
        vector_weights.append(current_chromosome)
    return np.array(vector_weights)


def vector_to_matrix(vector_weights, matrix_weights):
    result_matrix_weights = []
    for chromosome in range(POPULATION_SIZE): #POP SIZE
        start = 0
        end = 0
        for layer in range(HL_NUMBER+1): #HLSIZE + 1
            end = end + matrix_weights[chromosome, layer].size #size of the first layer i.e. 9
            #decompose vector
            layer_vector = vector_weights[chromosome, start:end]
            matrix_layer_weights = np.reshape(layer_vector, newshape=(matrix_weights[chromosome, layer].shape))
            result_matrix_weights.append(matrix_layer_weights)
            start=end
    return np.reshape(result_matrix_weights, newshape=matrix_weights.shape)

def test_outputs(best_weights):
    global test_in_data
    global test_out_data
    # 576 x 2 array = 576 entries, each with a 1x2 array specifying the predicted output i.e [1 0]
    predictions = np.zeros(shape=(TEST_ROWS,2))
    for sample in range(TEST_ROWS):
        y = test_in_data[sample]
        for current_weight in best_weights:
            y = np.matmul(y, current_weight) #matrix multiplication
            y = hard_limiter(y)
        predicted_sample = hard_limiter(y)
        predictions[sample] = predicted_sample
    correct_predictions = find_correct_predictions(predictions, test_out_data)
    accuracy = (correct_predictions/TEST_ROWS)*100
    return accuracy


def select_parents(fitness):
    global population_weights_vector
    #2 parents of size 400
    parents = np.empty((NUM_PARENTS_MATING, population_weights_vector.shape[1]))
    for parent in range(NUM_PARENTS_MATING):
        max_fitness_index = np.where(fitness == np.max(fitness))
        max_fitness_index = max_fitness_index[0][0]
        parents[parent, :] = population_weights_vector[max_fitness_index, :]
        fitness[max_fitness_index] = -9999999
    return parents

##just for progress bar
def startProgress(title):
    global progress_x
    sys.stdout.write(title + ": [" + "-"*40 + "]" + chr(8)*41)
    sys.stdout.flush()
    progress_x = 0

def progress(x):
    global progress_x
    global best_accuracy
    x = int(x * 40 // 100)
    sys.stdout.write("#" * (x - progress_x))
    sys.stdout.flush()
    progress_x = x

def endProgress():
    sys.stdout.write("#" * (40 - progress_x) + "]\n")
    sys.stdout.flush()


###############################################################################

'''
0. ARTIFICIAL NEURAL NETWORK FUNCTIONS
    Calculate the fitness of each chromosome with the help of predict_outputs
    function, which applies the matrix multiplication of the train_in_data and
    the weights of each chromosome to predict the values and then find the
    correct predictions to assign a fitness value to each chromosome for futher
    reproduction.
'''


def calculate_fitness():
    global population_weights_matrix
    #array of 10 elements for each chromosome's fitness
    fitness = np.empty(shape=(POPULATION_SIZE))
    for chromosome in range(POPULATION_SIZE):
        #extract each chromosome's 3 arrays
        current_chromosome = population_weights_matrix[chromosome, :]
        fitness[chromosome] = predict_outputs(current_chromosome)
    return fitness

def predict_outputs(chromosome):
    global predictions
    global train_in_data
    global train_out_data
    # 576 x 2 array = 576 entries, each with a 1x2 array specifying the predicted output i.e [1 0]
    predictions = np.zeros(shape=(TRAIN_ROWS,2))
    for sample in range(TRAIN_ROWS):
        y = train_in_data[sample]
        for current_weight in chromosome:
            y = np.matmul(y, current_weight) #matrix multiplication
            y = hard_limiter(y)
        predicted_sample = hard_limiter(y)
        predictions[sample] = predicted_sample
    correct_predictions = find_correct_predictions(predictions, train_out_data)
    accuracy = (correct_predictions/TRAIN_ROWS)*100
    return accuracy

###############################################################################

'''
1. INITIAL POPULATION.
    Initialize the population weights matrix and popuation weights vector, which
    consist of 10 different instances of different weights, each representing a
    chromosome in the population. Initialize the weights of the chromosomes with
    a random number between -1 and 1.

'''

def populate():
    global population_weights_matrix
    global population_weights_vector
    global best_weights_vector
    initial_population_weights = []

    #one weight vector is a chromosome for each member of population
    for current_weight in range(0, POPULATION_SIZE):
        #from input --> HL1 (INPUTxH1)
        input_HL1_weights = np.random.uniform (low=-1, high=1, size=(INPUT_NUMBER, HL1_NEURONS))
        #from HL1 --> OUTPUT (H2xOUTPUT)
        HL1_output_weights = np.random.uniform(low=-1, high=1, size=(HL1_NEURONS, OUTPUT_NEURONS))

        # initial_population_weights[0] = 3 arrays of a 2D array
        initial_population_weights.append(np.array([input_HL1_weights,  HL1_output_weights]))

    #just if you want to take the best weights from the csv instead of training the network
    if(GET_VECTOR_CSV == True):
        weights_csv = pandas.read_csv('weights.csv')
        for (i) in range((INPUT_NUMBER*HL1_NEURONS)+(HL1_NEURONS*OUTPUT_NUMBER)):
            best_weights_vector.append(weights_csv['# weights'][i])

    #10 instances of different neural networks
    population_weights_matrix = np.array(initial_population_weights)
    #10 instances of a 1D vector of size INPUT*HL1 + HL1*HL2 + HL2*OUPUT (10 chromosomes)
    population_weights_vector = matrix_to_vector(population_weights_matrix)


################################################################################

'''
2. REPRODUCTION
    Fit the chromosomes to reproduce offspring according to
    their fitness values. So we are going to have a new population.
    In addition, we will calculate the parents through a function, which
    is going to pick a chromosome based on its fitness value. After producing
    offspring we are going to apply crossover and mutation for each child.
'''

def reproduce():
    #these variables will change in each generation
    global best_weights
    global accuracies
    global population_weights_matrix
    global population_weights_vector
    global best_accuracy
    global best_weights_vector

    if(GET_VECTOR_CSV == True):
        population_weights_vector[0] = best_weights_vector

    startProgress("Generations: ")
    for generation in range(GENERATIONS):
        progress( generation/GENERATIONS*100 )

        #we need matrices of weights to compute the fitness of each chromosome
        population_weights_matrix = vector_to_matrix(population_weights_vector, population_weights_matrix)

        #calculate fitness of each chromosome in population
        fitness = calculate_fitness()
        #store the accuray of current generation just for information's sake
        accuracies[generation] = np.max(fitness)

        #selecting the best parents in the population for mating
        parents = select_parents(fitness.copy())

        #offspring size is POPULATION SIZE - PARENT MATING x SIZE OF A CHROMOSOME
        offspring_size = (population_weights_vector.shape[0]-parents.shape[0], population_weights_vector.shape[1])

        #apply crossover
        offspring_crossover = crossover(parents, offspring_size)

        #apply mutation
        offspring_mutation = mutation(offspring_crossover)

        population_weights_vector[0:NUM_PARENTS_MATING, :] = parents
        population_weights_vector[NUM_PARENTS_MATING:, :] = offspring_mutation


    population_weights_matrix = vector_to_matrix(population_weights_vector, population_weights_matrix)
    #just take the first chromosome
    best_weights = population_weights_matrix[0, :]
    endProgress()

    best_accuracy  = predict_outputs(best_weights)
    best_weights_vector = population_weights_vector[0]
    print("Accuracy of the best solution is: ", best_accuracy)

###############################################################################

'''
3. CROSSOVER
    Take half the weights of the first parent and half the weights of the second
    parent to breed the offpsring.

'''

def crossover(parents, offspring_size):
    global CROSSOVER_POINT
    offspring = np.empty(offspring_size)
    # i in 8 (in this case)
    for i in range(offspring_size[0]):
        #index of the first parent to mate
        father_index = i%parents.shape[0]
        #index of second parent
        mother_index = (i+1)%parents.shape[0]
        #offspring will have first half from father and second half from mother
        offspring[i, 0:CROSSOVER_POINT] = parents[father_index, 0:CROSSOVER_POINT]
        offspring[i, CROSSOVER_POINT:] = parents[mother_index, CROSSOVER_POINT:]
    return offspring

###############################################################################

'''
4.MUTATION
    The implementation of mutation is really easy, just change random weights
    with a random number between -1 and 1.
'''

def mutation(offspring_crossover):
    number_mutations = (MUTATION_PERCENT * offspring_crossover.shape[1]) /100
    mutation_indices = np.array(random.sample(range(0, offspring_crossover.shape[1]), np.uint8(number_mutations)))
    #Change one gene in each offspring picked at random
    for i in range(offspring_crossover.shape[0]):
        random_num = np.random.uniform(-1.0, 1.0, 1)
        offspring_crossover[i, mutation_indices] = offspring_crossover[i, mutation_indices] + random_num
    return offspring_crossover

###############################################################################

'''
LEARN
    Train the ANN with the GA by applying its steps. Mainly: populate and reproduce.
    After the best weights have been found, store them in a csv for further analysis.
    To start the Genetic Algorithm we need to start with an initial POPULATION
    then we repeat the reproduction (along with crossover and mutation) of the
    population according to the number of cycles (which is given)

'''
def learn():
    global best_weights_vector
    populate()
    reproduce()
    np.savetxt("weights.csv", best_weights_vector, delimiter=",", header="weights")
    graph()

'''
TEST
    Test the ANN by analyzing its performance with unseen data using the best weights found.
'''
def test():
    global best_weights
    test_accuracy = test_outputs(best_weights)
    print("Accuracy for unseen data: ", test_accuracy)

'''
GRAPH
    Plot the accuracy reached with respect to the generations.

'''
def graph():
    global accuracies
    global population_weights_matrix
    plt.plot(accuracies, linewidth=2, color="black")
    plt.xlabel("Iteration", fontsize=10)
    plt.ylabel("Fitness", fontsize=10)
    plt.xticks(np.arange(0, GENERATIONS+1, 100), fontsize=10)
    plt.yticks(np.arange(0, 101, 5), fontsize=10)
    plt.show()

"""
* MAIN PROGRAM
    Train the ANN and then test its performance with unseen data.
"""

def main():
    learn()
    test()


if __name__ == "__main__":
    main()
