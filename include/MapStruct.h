#pragma once
#include <Eigen/Core>

#define BLOCK_SIZE 8
#define BLOCK_SIZE_3 512
#define BLOCK_SIZE_M1 7

namespace voxelization
{

struct Voxel
{
    short sdf;
    unsigned char wt;
};

struct HashEntry
{
    int ptr;
    int offset;
    Eigen::Vector3i pos;
};

struct RenderingBlock
{
    Eigen::Matrix<short, 2, 1> upper_left;
    Eigen::Matrix<short, 2, 1> lower_right;
    Eigen::Vector2f zrange;
};

struct MapStruct
{
    void release();
    bool empty();
    void reset();
    void create(int nEntry, int nBucket, int nVBlock, float voxelSize, float truncationDist);
    void getVisibleBlockCount(uint &hostData);
    void resetVisibleBlockCount();

    int nBucket;
    int nEntry;
    int nVBlock;
    float voxelSize;
    float truncationDist;

    int *heap;
    int *excessPtr;
    int *heapPtr;
    int *bucketMutex;
    Voxel *voxelBlock;
    HashEntry *hashTable;
    HashEntry *visibleTable;
    uint *visibleBlockNum;
};

} // namespace voxelization