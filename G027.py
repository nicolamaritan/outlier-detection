import math
import time
from pyspark import SparkContext, SparkConf
import sys
import os

def ExactOutliers(listOfPoints, D, M, K):
    # Each line of the kind "x,y\n" is converted into a tuple (x,y) of floats
    listOfPoints = [tuple(map(float, point.strip().split(','))) for point in listOfPoints]

    # [i, |B_s(p,D)|] for the i-th point in points
    # We start with 1 to count p itself, since in the following loop it will
    # not consider the case i=j (j starts from i+1).
    B_cardinality = [[i, 1] for i, _ in enumerate(listOfPoints)]

    # We compute only the upper triangle of the distance matrix for efficiency
    for i in range(0, len(listOfPoints)-1):
        for j in range(i+1, len(listOfPoints)):
            '''We use the function math.dist because we noticed that is faster than 
               computing the square of the distance.
               This is probably due to the inner implementation of math.dist'''
            if  math.dist(listOfPoints[i], listOfPoints[j]) <= D:
                B_cardinality[i][1] += 1
                B_cardinality[j][1] += 1

    print(f"Number of Outliers =", sum([1 for _, B in B_cardinality if B <= M]))
    
    # Sort by non-decreasing values of |B_s(p,D)|
    B_cardinality.sort(key = lambda x: x[1])

    for i in range(K):
        if B_cardinality[i][1] <= M:
            print(f"Point: {listOfPoints[B_cardinality[i][0]]}")

def floor_coordinates(point, D):
    capital_lambda = D / (2*math.sqrt(2))
    # Converts each key (x_p, y_p) into (floor(x_p/Lambda), floor(x_p/Lambda)).
    # Floor is implemented through casting from float to int.
    return [((math.floor(point[0] / capital_lambda), math.floor(point[1] / capital_lambda)), 1)]

def gather_pairs_partitions(pairs):
	pairs_dict = {}
	for p in pairs:
		coordinate, occurrence = p[0], p[1] # p[1] is always 1 from the previous round
		if coordinate not in pairs_dict.keys():
			pairs_dict[coordinate] = occurrence
		else:
			pairs_dict[coordinate] += occurrence
	return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def MRApproxOutliers(inputPoints, D, M, K):
    # -------------------- Step A --------------------
    pairs_RDD = (inputPoints.flatMap(lambda str: floor_coordinates(str, D))   # <-- MAP PHASE (R1)
           .mapPartitions(gather_pairs_partitions)                            # <-- REDUCE PHASE (R1)
           .groupByKey()                                                      # <-- SHUFFLE+GROUPING
           .mapValues(lambda vals: sum(vals))                                 # <-- REDUCE PHASE (R2)
           .cache()                         
           )

    # -------------------- Step B --------------------
    pairs_list = pairs_RDD.collect()

    cell_dict = {}
    for i in range(len(pairs_list)):
        current_cell = pairs_list[i][0]
        # Put the initial value of N3, N7 corresponding to the cell's size itself.
        cell_dict[current_cell] = [pairs_list[i][1], pairs_list[i][1]]

        # We consider only pairs (i,j) with j<i because of simmetry.
        for j in range(0, i):
            other_cell = pairs_list[j][0]
            
            # Checks if other_cell is within the square of size 3 with center current_cell.
            # If it is the case, it is also in the square of size 7.
            if(is_within_square(current_cell, other_cell, 3)):
                # If other_cell is in such square, then current_cell is within the square
                # of size 3 with center other_square. So we update N3 and N7 of both cells.
                # This works since we are considering only once the pairwise distance (wrt cells)
                # between two arbirary cells.
                cell_dict[current_cell][0] += pairs_list[j][1]
                cell_dict[other_cell][0] += pairs_list[i][1]
                cell_dict[current_cell][1] += pairs_list[j][1]
                cell_dict[other_cell][1] += pairs_list[i][1]

            # Checks if other_cell is within the square of size 7 with center current_cell.
            elif(is_within_square(current_cell, other_cell, 7)):
                cell_dict[current_cell][1] += pairs_list[j][1]
                cell_dict[other_cell][1] += pairs_list[i][1]

    outliers_count = 0
    uncertain_count = 0
    for pair in pairs_list:
        if cell_dict[pair[0]][1] <= M:
            outliers_count+=pair[1]
        elif(cell_dict[pair[0]][0] <= M):
            uncertain_count+=pair[1]

    print("Number of sure outliers =", outliers_count)
    print("Number of uncertain points =", uncertain_count)

    # We first swap size and cell, then sort by key, i.e. the size of the cell
    for size, cell in pairs_RDD.map(lambda pair: (pair[1], pair[0])).sortByKey().take(K):
        print(f"Cell: {cell}  Size = {size}")

def is_within_square(current, other, size):
    return (abs(other[0] - current[0]) <= size//2) and (abs(other[1] - current[1]) <= size//2)

def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 6, "Usage: python G027.py <file_name> <D> <M> <K> <L>"

    # SPARK SETUP
    conf = SparkConf().setAppName('Outlier Detection')
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read value of D
    D = sys.argv[2]
    assert is_float(D), "D must be a float"
    D = float(D)

    # 2. Read value of M
    M = sys.argv[3]
    assert M.isdigit(), "M must be an integer"
    M = int(M)

    # 3. Read value of K
    K = sys.argv[4]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 4. Read number of partitions
    L = sys.argv[5]
    assert L.isdigit(), "L must be an integer"
    L = int(L)

    # PRINT INPUT FILE, D, M, K, L
    print(f'{sys.argv[1]} D = {sys.argv[2]} M = {sys.argv[3]} K = {sys.argv[4]} L = {sys.argv[5]}')

    # 5. Read input file and subdivide it into L random partitions
    data_path = sys.argv[1]
    assert os.path.isfile(data_path), "File or folder not found"
    rawData = sc.textFile(data_path, minPartitions=L)

    # inputPoints is an RDD of pairs of float    
    inputPoints = rawData.flatMap(lambda point: [tuple(map(float, point.strip().split(',')))])
    inputPoints.repartition(numPartitions=L).cache()

    # total number of points
    num_Points = len(inputPoints.collect())
    print(f'Number of points = {num_Points}')

    MAX_POINTS_EXACT_OUTLIERS = 200000

    if num_Points <= MAX_POINTS_EXACT_OUTLIERS:
        # Open the file in read mode
        with open(data_path, 'r') as file:
            # Read all lines into a list
            listOfPoints = file.readlines()
        start = time.time()
        ExactOutliers(listOfPoints=listOfPoints, D=D, M=M, K=K)
        end = time.time()
        print(f'Running time of ExactOutliers = {round(1000*(end - start))} ms')

    start = time.time()
    MRApproxOutliers(inputPoints=inputPoints, D=D, M=M, K=K)
    end = time.time()
    print(f'Running time of MRApproxOutliers = {round(1000*(end - start))} ms')

if __name__ == "__main__":
	main()