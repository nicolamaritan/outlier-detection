import math

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

                    
def main():
    # Open the file in read mode
    with open('TestN15-input.txt', 'r') as file:
        # Read all lines into a list
        points = file.readlines()
    ExactOutliers(points=points, D=2, M=3, K=15)

if __name__ == "__main__":
	main()