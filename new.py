import random

# Define a basic structure for Chromosome and Population
class Chromosome:
    def __init__(self, features, hyperparameters):
        self.genes = {
            'features': features,
            'hyperparameters': hyperparameters,
        }
        self.fitness = self.evaluate_fitness()

    def evaluate_fitness(self):
        # TO DO Decode
        # Placeholder for fitness evaluation function
        return random.uniform(0, 1)

def initialize_population(pop_size):
    # Initialize a population of chromosomes with random values
    population = []
    for _ in range(pop_size):
        # 5 (index) : ('^GSPC', 'Open')	('^GSPC', 'High')	('^GSPC', 'Low')	('^GSPC', 'Close')	('^GSPC', 'Volume')	
        # 31 (technical indicators) : Bollinger_Bands_Upper	Bollinger_Bands_Middle	Bollinger_Bands_Lower	DEMA	Midpoint	Midpoint_Price	T3_Moving_Average	ADX	Absolute_Price_Oscillator	Aroon_Up	Aroon_Down	Aroon_Oscillator	Balance_of_Power	CCI	Chande_Momentum_Oscillator	MACD	MACD_Signal	MACD_Histogram	Money_Flow_Index	Normalized_Average_True_Range	Chaikin_A/D_Line	Chaikin_A/D_Oscillator	Median_Price	Typical_Price	Weighted_Closing_Price	Hilbert_Dominant_Cycle_Phase	Hilbert_Phasor_Components_Inphase	Hilbert_Phasor_Components_Quadrature	Hilbert_SineWave	Hilbert_LeadSineWave	Hilbert_Trend_vs_Cycle_Mode	
        # 5( stock) : Open	High	Low	Close	Volume	
        # 9 (macro indicators) : M2	S&P CoreLogic Case-Shiller U.S. National Home Price Index	All-Transactions House Price Index for the United States	M1	Consumer Price Index for All Urban Consumers: All Items in U.S. City Average	Trade Balance: Goods and Services, Balance of Payments Basis	New Privately-Owned Housing Units Started: Total Units	Domestic Auto Production	New One Family Houses Sold

        features = [random.choice([0, 1]) for _ in range(50)]

        window_size = [random.choice([0, 1]) for _ in range(5)] # 0 to 31
        WaveKAN = [random.choice([0, 1])]
        NaiveFourierKAN = [random.choice([0, 1])]
        JacobiKAN = [random.choice([0, 1])]
        ChebyKAN = [random.choice([0, 1])]
        TaylorKAN = [random.choice([0, 1])]
        RBFKAN = [random.choice([0, 1])]
        KAN_layers = WaveKAN + NaiveFourierKAN + JacobiKAN + ChebyKAN + TaylorKAN + RBFKAN
        hyperparameters = window_size + KAN_layers

        population.append(Chromosome(features, hyperparameters))

    return population

def intra_chromosome_crossover(ch1, n_features, n_hyperparameters):
    print("Before Intra:\n", ch1.genes['features'])

    n = min(n_features, n_hyperparameters)

    features_filter = [1] * n + [0] * (n_features - n)
    random.shuffle(features_filter)

    selected_indices = [i for i, val in enumerate(features_filter) if val == 1]
    not_selected_index = [i for i in range(n)]
    
    # Swap the selected pairs
    for idx in selected_indices:
        swap_index = random.sample(not_selected_index, 1)[0]
        not_selected_index.remove(swap_index)
        ch1.genes['features'][idx], ch1.genes['hyperparameters'][swap_index] = ch1.genes['hyperparameters'][swap_index], ch1.genes['features'][idx]
    
    print("After Intra:\n", ch1.genes['features'])
    return ch1

def inter_chromosome_crossover(ch1, ch2, n_features, n_hyperparameters):
    print("Before Inter:\n", ch1.genes['features'])
    print("Before Inter:\n", ch2.genes['features'])

    features1 = ch1.genes['features']
    hyperparameters1 = ch1.genes['hyperparameters']
    
    features2 = ch2.genes['features']
    hyperparameters2 = ch2.genes['hyperparameters']
    
    crossover_point1 = random.randint(0, n_features - 1)
    crossover_point2 = random.randint(0, n_hyperparameters - 1)
    
    features1[crossover_point1:], features2[crossover_point1:] = features2[crossover_point1:], features1[crossover_point1:]
    hyperparameters1[crossover_point2:], hyperparameters2[crossover_point2:] = hyperparameters2[crossover_point2:], hyperparameters1[crossover_point2:]
    
    ch1.genes['features'] = features1
    ch1.genes['hyperparameters'] = hyperparameters1
    
    ch2.genes['features'] = features2
    ch2.genes['hyperparameters'] = hyperparameters2

    print("After Inter:\n", ch1.genes['features'])
    print("After Inter:\n", ch2.genes['features'])
    
    return ch1, ch2


def apply_mutation(chromosome, mutation_rate, n_features, n_hyperparameters):
    print("Before mut:\n", chromosome.genes['features'])

    # Mutate features
    chromosome.genes['features'] = [
        abs(gene - 1) if random.random() < mutation_rate else gene
        for gene in chromosome.genes['features']
    ]

    # Mutate hyperparameters
    chromosome.genes['hyperparameters'] = [
        abs(gene - 1) if random.random() < mutation_rate else gene
        for gene in chromosome.genes['hyperparameters']
    ]

    print("After mut:\n", chromosome.genes['features'])
    return chromosome

n_features = 50
n_hyperparameters = 11
population = initialize_population(2)  
ch1, ch2 = random.sample(population, 2)
ch1 = intra_chromosome_crossover(ch1, n_features, n_hyperparameters)
ch1, ch2 = inter_chromosome_crossover(ch1, ch2, n_features, n_hyperparameters)
mutation_rate = 0.1

of1 = apply_mutation(ch1, mutation_rate, n_features, n_hyperparameters)
of2 = apply_mutation(ch2, mutation_rate, n_features, n_hyperparameters)