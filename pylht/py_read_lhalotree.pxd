cimport numpy as np
from libc.stdint cimport uint32_t, uint64_t, int64_t, int32_t


cdef extern from "lhalotree.h":
    struct lhalotree:

        # merger tree pointers
        int Descendant
        int FirstProgenitor
        int NextProgenitor
        int FirstHaloInFOFgroup
        int NextHaloInFOFgroup

        # properties of halo
        int Len
        float M_Mean200
        float Mvir  # for Millennium, Mvir=M_Crit200
        float M_TopHat
        float Pos[3]
        float Vel[3]
        float VelDisp
        float Vmax
        float Spin[3]
        long long MostBoundID

        # original position in simulation tree files
        int SnapNum
        int FileNr
        int SubhaloIndex
        float SubHalfMass


cdef extern from "read_lhalotree.h":
    # Actual useful functions
    size_t read_single_lhalotree_from_stream(FILE *fp, struct lhalotree *tree,
                                             const int32_t nhalos)
    int pread_single_lhalotree_with_offset(int fd, struct lhalotree *tree,
                                           const int32_t nhalos, off_t offset)
    int read_file_headers_lhalotree(const char *filename, int32_t *ntrees,
                                    int32_t *totnhalos, int32_t **nhalos_per_tree)
    int32_t read_ntrees_lhalotree(const char *filename)
    struct lhalotree * read_entire_lhalotree(const char *filename, int *ntrees,
                                             int *totnhalos, int **nhalos_per_tree)
    struct lhalotree * read_single_lhalotree(const char *filename,
                                             const int32_t treenum)

    # Sorting an LHalotree output into a new order
    int sort_lhalotree_in_snapshot_and_fof_groups(struct lhalotree *tree,
                                                  const int64_t nhalos, int test)
    # And fixing lhalotree mergertree indices from a generic sort
    int fix_mergertree_index(struct lhalotree *tree, const int64_t nhalos,
                             const int32_t *index)
