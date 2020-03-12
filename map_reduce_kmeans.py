from pyspark import SparkContext, SparkConf
from math import sin, cos, atan2, sqrt

def haversine(x1, x2):
        # Haversine Distance metric
        '''
                x1: (2,) array | list | tuple
                x2: (2,) array | list | tuple
        '''

        phi1 = x1[0]
        phi2 = x2[0]
        l1 = x1[1]
        l2 = x2[1]
        dphi = phi1 - phi2
        dl = l1 - l2
        R = 6371   # radius of earth (km)
        
        a = hav(dphi) + cos(phi1) * cos(phi2) * hav(dl)
        c = 2 * atan2(sqrt(a), sqrt(1-a))
        
        # haversine distance
        d = R * c
        
        return d

def hav(h):
        return (sin(h / 2))**2


def Estep(point, centroids):
        # E step of classic KMeans 
        '''
                point:          (2,) array | list | tuple
                centroids:      (K,2) array
        '''
        r = 0   # the cluster where point belongs to
        min_cluster_distance = float("+inf") 
        for i in range(len(centroids)):
                current_dist = haversine(point, centroids[i])
                if current_dist < min_cluster_distance:
                        min_cluster_distance = current_dist
                        r = i
        return r


# SparkConf and SparkContext
conf = SparkConf().setAppName('adv_db_kmeans')
sc = SparkContext(conf=conf)
path = "hdfs://master:9000/yellow_tripdata_1m.csv"

# Creating RDD
# read dataset and split using ',' as delimeter  
dataset = sc.textFile(path, 50).map( lambda line: line.split(',') )
# keep only the coordinates (x,y)
dataset = dataset.map( lambda line: (float(line[3]), float(line[4])) )
# filter zero coordinates (they do not correspond to real coordinates)
dataset = dataset.filter( lambda line: line[0] != 0 and line[1] != 0 )

# hyperparameters
MAX_ITER = 3

# Initialize centroids as the first five points of the dataset
centroids = dataset.take(5)

'''
        Map-Reduce KMeans,
        as it is described in "Pattern Recognition And Machine Learning, Bishop, 2006".
'''
it = 0
while it < MAX_ITER:
        # E step
        points_and_labels = dataset.map( 
                lambda point: (Estep(point, centroids), (point, 1)) 
        )
        
        # M step
        ## t:   (x,y), 1
        points_and_labels = points_and_labels.reduceByKey(
                lambda t1,t2: ( (t1[0][0] + t2[0][0], t1[0][1] + t2[0][1]), t1[1] + t2[1] )
        )
        ## cluster:     (x,y), sum(1)
        centroids = points_and_labels.mapValues(
                lambda cluster: (cluster[0][0] / cluster[1], cluster[0][1] / cluster[1])
        ).collectAsMap()
        
        # while step
        it += 1


print("KMeans results")
for i in range(len(centroids)):
        print(i, centroids[i])
