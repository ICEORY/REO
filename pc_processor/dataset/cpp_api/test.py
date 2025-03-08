import numpy as np
from build import occupancy_api
pointcloud = np.array([
    [0,1,2,3],
    [0,1,2,20],
    [0,1,2,3],
    [10,12,20,10],
    [10,12,20,30],
    [10,12,20,30],
]).astype(np.float32)
print(pointcloud.shape)
result = occupancy_api.pointcloudVoxelize(pointcloud, 0.2)
print(result)
