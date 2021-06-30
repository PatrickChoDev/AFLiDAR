import spconv


class VoxelGenerator:
    def __init__(
        self,
        voxel_size=[0.1, 0.1, 0.1],
        point_range=[-50, -50, -3, 50, 50, 1],
        max_num=30,
        max_voxel=40000,
    ):
        self.__voxel_generator = spconv.utils.VoxelGenerator(
            voxel_size=voxel_size,
            point_cloud_range=point_range,
            max_num_points=max_num,
            max_voxels=max_voxel,
        )

    def __call__(self, points):
        return self.__voxel_generator.generate(points)
