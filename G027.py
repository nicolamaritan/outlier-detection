import math
from pyspark import SparkContext, SparkConf
import sys
import os

def ExactOutliers(points, D, M, K):
    # Each line of the kind "x,y\n" is converted into a tuple (x,y) of floats
    points = [tuple(map(float, point.strip().split(','))) for point in points]

    # [i, |B_s(p,D)|] for the i-th point in points
    B_cardinality = [[i, 0] for i, _ in enumerate(points)]

    # We compute only the upper triangle of the distance matrix for efficiency
    for i in range(0, len(points)-1):
        for j in range(i+1, len(points)):
            if  math.dist(points[i], points[j]) <= D:
                B_cardinality[i][1] += 1
                B_cardinality[j][1] += 1

    print(f"({D},{M})-outliers:", sum([1 for _, B in B_cardinality if B <= M]))
    
    # Sort by non-decreasing values of |B_s(p,D)|
    B_cardinality.sort(key = lambda x: x[1])

    for i in range(K):
        if B_cardinality[i][1] <= M:
            print(points[B_cardinality[i][0]])

def MRApproxOutliers(points, D, M, K):
    out = points.flatMap(lambda point: [(tuple(map(float, point.strip().split(','))), 1)])
    print(out.collect())
                    
def main():
    # CHECKING NUMBER OF CMD LINE PARAMETERS
    assert len(sys.argv) == 3, "Usage: python G027.py <K> <file_name>"

    # SPARK SETUP
    conf = SparkConf().setAppName('???')
    sc = SparkContext(conf=conf)

    # INPUT READING

    # 1. Read number of partitions
    K = sys.argv[1]
    assert K.isdigit(), "K must be an integer"
    K = int(K)

    # 2. Read input file and subdivide it into K random partitions
    data_path = sys.argv[2]
    assert os.path.isfile(data_path), "File or folder not found"
    points = sc.textFile(data_path, minPartitions=K).repartition(numPartitions=K).cache()

    # Open the file in read mode
    with open(data_path, 'r') as file:
        # Read all lines into a list
        points_file = file.readlines()
    ExactOutliers(points=points_file, D=2, M=3, K=15)
    MRApproxOutliers(points=points, D=2, M=3, K=15)

if __name__ == "__main__":
	main()