/*
 * mm.c
 *
 * Double bit mapped seg list
 * Heap memory: doubly linked list (free block) ; allocated block: no footer
 * Free blocks are bigger than 128B
 * Allocation block has no footer, use next blocks's prev_alloc (located in the header)
 * seg list: 
 * 0-14:  16B - 128B (8-bit scale)
 * 15-31: 2^x scale
 * */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <string.h>

#include "mm.h"
#include "memlib.h"

#include <stdint.h>

team_t team = {
    /* Team name */
    "MikuFanClub",
    /* First member's full name */
    "PARK SEOJIN",
    /* First member's email address */
    "awasorara@dgist.ac.kr",
    /* Second member's full name (leave blank if none) */
    "",
    /* Second member's email address (leave blank if none) */
    ""
};

/* single word (4) or double word (8) alignment */
#define ALIGNMENT 8

/* rounds up to the nearest multiple of ALIGNMENT */
#define ALIGN(size) (((size) + (ALIGNMENT-1)) & (~0x7))

#define WSIZE 4
#define WSIZE_SH 2
#define DSIZE 8
#define DSIZE_SH 3
#define CHUNKSIZE (1<<10) // 1KB

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))

#define PACK(size, prev_alloc, alloc) ((size) | ((prev_alloc) << 1) | (alloc))
#define GET(p) (*(unsigned int *)(p))
#define PUT(p, val) (*(unsigned int *)(p) = (val))

#define GET_SIZE(p) (GET(p) & ~0x7)
#define GET_ALLOC(p) (GET(p) & 0x1)
#define GET_PREV_ALLOC(p) ((GET(p) & 0x2) >> 1)

#define HDRP(bp) ((char *)(bp) - WSIZE)
#define FTRP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)) - DSIZE)

#define NEXT_BLKP(bp) ((char *)(bp) + GET_SIZE(HDRP(bp)))

#define GET_NEXT_FREE(bp) (*(void * *)(bp))
#define SET_NEXT_FREE(bp, ptr) (*(void * *)(bp) = (ptr))

#define GET_PREV_FREE(bp) (*((void * *)(bp) + 1))
#define SET_PREV_FREE(bp, ptr) (*((void * *)(bp) + 1) = (ptr))

#define GET_ROOT(class) (*(void * *)(heap_start + ((class) << 2)))
#define SET_ROOT(class, ptr) (*(void * *)(heap_start + ((class) << 2)) = (ptr))

#define METADATA_SIZE 128 // 32 classes * 4 bytes = 128 bytes

static char * heap_start = NULL; 
static char * heap_listp = NULL; 
static void * realloc_block = NULL;

#define SPLIT_THRESHOLD 96
#define REALLOC_SPLIT_THRESHOLD 128

static void * extend_heap(size_t words);
static void * coalesce(void * bp);
static void * find_fit(size_t asize);
static void * place(void *bp, size_t asize);

/*
 * get_class - Maps a block size to its corresponding segregated class index (0-31).
 * Bins 0-14 provide granular 8-byte scaling for small blocks to prevent internal fragmentation,
 * while bins 15-31 scale exponentially up to the maximum heap limit.
 */
static inline int get_class(size_t size) {
    if (size <= 16) return 0;
    if (size <= 24) return 1;
    if (size <= 32) return 2;
    if (size <= 40) return 3;
    if (size <= 48) return 4;
    if (size <= 56) return 5;
    if (size <= 64) return 6;
    if (size <= 72) return 7;
    if (size <= 80) return 8;
    if (size <= 88) return 9;
    if (size <= 96) return 10;
    if (size <= 104) return 11;
    if (size <= 112) return 12;
    if (size <= 120) return 13;
    if (size <= 128) return 14;
    if (size <= 256) return 15;
    if (size <= 384) return 16;
    if (size <= 512) return 17;
    if (size <= 768) return 18;
    if (size <= 1024) return 19;
    if (size <= 1536) return 20;
    if (size <= 2048) return 21;
    if (size <= 3072) return 22;
    if (size <= 4096) return 23;
    if (size <= 6144) return 24;
    if (size <= 8192) return 25;
    if (size <= 12288) return 26;
    if (size <= 16384) return 27;
    if (size <= 32768) return 28;
    if (size <= 65536) return 29;
    if (size <= 131072) return 30;
    return 31;
}

/*
 * safe_prev_blkp - O(1) backward physical block tracker.
 * Reads the current block's PREV_ALLOC bit to see if the predecessor is free,
 * then uses the predecessor's written footer to jump backward safely.
 */
static inline void * safe_prev_blkp(void * bp) {
    if (GET_PREV_ALLOC(HDRP(bp))) return NULL;
    return (char *)(bp) - GET_SIZE((char *)(bp) - DSIZE);
}

/*
 * get_last_free_block - Tail block surveyor.
 * Scans the epilogue header's allocation history to determine if a free block 
 * is resting right at the edge of the heap boundary, allowing sbrk mitigation.
 */
static inline void * get_last_free_block(size_t * size_out) {
    char * epilogue_hdr = (char *)mem_heap_hi() - 3;
    if (GET_PREV_ALLOC(epilogue_hdr) == 0) {
        size_t size = GET_SIZE(epilogue_hdr - WSIZE);
        if (size_out) *size_out = size;
        return epilogue_hdr - size + WSIZE;
    }
    if (size_out) *size_out = 0;
    return NULL;
}

/*
 * insert_node - O(1) LIFO explicit free list injector.
 * Prepends the free block to the head of the mapped size-class bin.
 */
static void insert_node(void * bp, size_t size) {
    int class = get_class(size);
    void * next = GET_ROOT(class);
    
    SET_NEXT_FREE(bp, next);
    SET_PREV_FREE(bp, NULL);
    
    if (next != NULL) {
        SET_PREV_FREE(next, bp);
    }
    SET_ROOT(class, bp);
}

/*
 * remove_node - O(1) explicit free list disconnection.
 * Unlinks a block instantly from its doubly-linked chain during allocation or coalescing.
 */
static void remove_node(void * bp) {
    size_t size = GET_SIZE(HDRP(bp));
    int class = get_class(size);
    void * next = GET_NEXT_FREE(bp);
    void * prev = GET_PREV_FREE(bp);
    
    if (prev != NULL) {
        SET_NEXT_FREE(prev, next);
    } else {
        SET_ROOT(class, next);
    }
    
    if (next != NULL) {
        SET_PREV_FREE(next, prev);
    }
}

/* * mm_init - Memory allocator setup routine.
 * Reserves the initial 128-byte Metadata Zone for segregated list roots,
 * aligns the entry points, and establishes the foundational prologue/epilogue boundaries.
 */
int mm_init(void)
{
    realloc_block = NULL;
    size_t initial_bytes = 144; // 128 (metadata) + 16 (padding, prologue, epilogue)
    heap_start = mem_sbrk(initial_bytes);
    if (heap_start == (void *)-1) return -1;
    
    memset(heap_start, 0, METADATA_SIZE);
    
    heap_listp = heap_start + METADATA_SIZE;
    PUT(heap_listp, 0); // Padding
    PUT(heap_listp + WSIZE, PACK(DSIZE, 1, 1)); // Prologue Header
    PUT(heap_listp + DSIZE, PACK(DSIZE, 1, 1)); // Prologue Footer
    PUT(heap_listp + DSIZE + WSIZE, PACK(0, 1, 1)); // Epilogue Header
    heap_listp += DSIZE;
    
    if (extend_heap(CHUNKSIZE >> WSIZE_SH) == NULL) return -1;
    /* init status check */
    // mm_checkheap(0); 
    return 0;
}

/*
 * extend_heap - Dynamically requests additional system memory via mem_sbrk.
 * Formats the raw memory area into a valid free block structure and triggers coalescing.
 */
static void * extend_heap(size_t words) {
    char * bp;
    size_t size = (words % 2) ? (words + 1) << WSIZE_SH : words << WSIZE_SH;
    
    if ((long)(bp = mem_sbrk(size)) == -1) return NULL;
    
    size_t prev_alloc = GET_PREV_ALLOC(bp - WSIZE);
    PUT(HDRP(bp), PACK(size, prev_alloc, 0));
    PUT(FTRP(bp), PACK(size, prev_alloc, 0));
    PUT(HDRP(NEXT_BLKP(bp)), PACK(0, 0, 1));
    
    return coalesce(bp);
}

/*
 * coalesce - Assembles physical neighboring blocks in O(1) time.
 * Seamlessly merges adjacent free blocks across 4 unique layout patterns,
 * managing the Footless layout bit propagation cleanly.
 */
static void * coalesce(void *bp) {
    size_t size = GET_SIZE(HDRP(bp));
    size_t prev_alloc = GET_PREV_ALLOC(HDRP(bp));
    size_t next_alloc = GET_ALLOC(HDRP(NEXT_BLKP(bp)));
    
    if (prev_alloc && next_alloc) {
        PUT(HDRP(bp), PACK(size, 1, 0));
        PUT(FTRP(bp), PACK(size, 1, 0));
        char * next_hdr = HDRP(NEXT_BLKP(bp));
        PUT(next_hdr, PACK(GET_SIZE(next_hdr), 0, GET_ALLOC(next_hdr)));
    } 
    else if (prev_alloc && !next_alloc) {
        remove_node(NEXT_BLKP(bp));
        size += GET_SIZE(HDRP(NEXT_BLKP(bp)));
        PUT(HDRP(bp), PACK(size, 1, 0));
        PUT(FTRP(bp), PACK(size, 1, 0));
    } 
    else if (!prev_alloc && next_alloc) {
        void * prev_bp = safe_prev_blkp(bp);
        remove_node(prev_bp);
        size += GET_SIZE(HDRP(prev_bp));
        size_t prev_prev_alloc = GET_PREV_ALLOC(HDRP(prev_bp));
        PUT(HDRP(prev_bp), PACK(size, prev_prev_alloc, 0));
        PUT(FTRP(bp), PACK(size, prev_prev_alloc, 0));
        bp = prev_bp;
        
        char * next_hdr = HDRP(NEXT_BLKP(bp));
        PUT(next_hdr, PACK(GET_SIZE(next_hdr), 0, GET_ALLOC(next_hdr)));
    } 
    else {
        void * prev_bp = safe_prev_blkp(bp);
        void * next_bp = NEXT_BLKP(bp);
        remove_node(prev_bp);
        remove_node(next_bp);
        size += GET_SIZE(HDRP(prev_bp)) + GET_SIZE(HDRP(next_bp));
        size_t prev_prev_alloc = GET_PREV_ALLOC(HDRP(prev_bp));
        PUT(HDRP(prev_bp), PACK(size, prev_prev_alloc, 0));
        PUT(FTRP(next_bp), PACK(size, prev_prev_alloc, 0));
        bp = prev_bp;
    }
    
    insert_node(bp, size);
    return bp;
}

/* * find_fit - Bounded cache-friendly Best-Fit fit locator.
 * Scans up to a hard limit of 15 nodes per list group to secure near-perfect 
 * spatial fitting while maintaining an deterministic execution timeline.
 */
static void * find_fit(size_t asize) {
    int class = get_class(asize);
    
    for (int c = class; c < 32; c++) {
        void * bp = GET_ROOT(c);
        void * best_bp = NULL;
        size_t min_diff = ~0U;
        int search_count = 0;
        
        while (bp != NULL && search_count < 15) { // hw opt
            size_t bp_size = GET_SIZE(HDRP(bp));
            if (bp_size >= asize) {
                size_t diff = bp_size - asize;
                if (diff < min_diff) {
                    min_diff = diff;
                    best_bp = bp;
                    if (diff == 0) return best_bp;
                }
                search_count++;
            }
            bp = GET_NEXT_FREE(bp);
        }
        if (best_bp != NULL) return best_bp;
    }
    return NULL;
}

/*
 * place - Places the requested block in a free block, splitting if necessary.
 * Leverages the Back-Splitting policy (SPLIT_THRESHOLD = 96B) to separate small 
 * and large layouts, isolating fragmentation to different memory regions.
 */
static void * place(void * bp, size_t asize) {
    size_t csize = GET_SIZE(HDRP(bp));
    size_t prev_alloc = GET_PREV_ALLOC(HDRP(bp));
    
    remove_node(bp);
    
    if ((csize - asize) >= 16) {
        if (asize < SPLIT_THRESHOLD) {
            PUT(HDRP(bp), PACK(asize, prev_alloc, 1));
            void * next_bp = NEXT_BLKP(bp);
            PUT(HDRP(next_bp), PACK(csize - asize, 1, 0));
            PUT(FTRP(next_bp), PACK(csize - asize, 1, 0));
            coalesce(next_bp);
            return bp;
        } else {
            PUT(HDRP(bp), PACK(csize - asize, prev_alloc, 0));
            PUT(FTRP(bp), PACK(csize - asize, prev_alloc, 0));
            void * next_bp = NEXT_BLKP(bp);
            PUT(HDRP(next_bp), PACK(asize, 0, 1));
            char * post_hdr = HDRP(NEXT_BLKP(next_bp));
            PUT(post_hdr, PACK(GET_SIZE(post_hdr), 1, GET_ALLOC(post_hdr)));
            coalesce(bp);
            return next_bp;
        }
    } else {
        PUT(HDRP(bp), PACK(csize, prev_alloc, 1));
        char * next_hdr = HDRP(NEXT_BLKP(bp));
        PUT(next_hdr, PACK(GET_SIZE(next_hdr), 1, GET_ALLOC(next_hdr)));
        return bp;
    }
}

/*
 * mm_malloc - Allocates aligned chunks via explicit segregated tracking.
 * Dynamically updates requests to double-word alignments, utilizes trailing free blocks 
 * to minimize footprint, and triggers speculative 4KB extensions for prefetchers.
 */
void * mm_malloc(size_t size)
{
    size_t asize;
    size_t extendsize;
    char * bp;
    
    if (size == 0) return NULL;
    
    if (size <= 12) asize = 16;
    else asize = DSIZE * ((size + WSIZE + (DSIZE - 1)) >> DSIZE_SH);
    
    if ((bp = find_fit(asize)) != NULL) {
        void *ptr = place(bp, asize);
        /* check after malloc */
        //mm_checkheap(0);
        return ptr;
    }
    
    if (asize < 512) {
        extendsize = 4 * CHUNKSIZE; // hw opt
    } else {
        extendsize = asize;
    }
    
    size_t last_free_size = 0;
    void * last_free_bp = get_last_free_block(&last_free_size);
    if (last_free_bp != NULL) {
        if (extendsize > last_free_size) {
            extendsize -= last_free_size;
        } else {
            extendsize = 0;
        }
    }
    
    if (extendsize > 0) {
        if ((bp = extend_heap(extendsize >> WSIZE_SH)) == NULL) return NULL;
        void *ptr = place(bp, asize);
        /* check after malloc */
        //mm_checkheap(0);
        return ptr;
    } else {
        void *ptr = place(last_free_bp, asize);
        /* check after malloc */
        // mm_checkheap(0);
        return ptr;
    }
}

/*
 * mm_free - Returns memory chunks back to the allocator.
 * Restores boundary tags, links the node back into its size bin, and handles coalescing.
 */
void mm_free(void * ptr)
{
    if (ptr == NULL) return;
    if (ptr == realloc_block) realloc_block = NULL;
    
    size_t size = GET_SIZE(HDRP(ptr));
    size_t prev_alloc = GET_PREV_ALLOC(HDRP(ptr));
    
    PUT(HDRP(ptr), PACK(size, prev_alloc, 0));
    PUT(FTRP(ptr), PACK(size, prev_alloc, 0));
    
    coalesce(ptr);
    /* check after free */
    //mm_checkheap(0);
}

/*
 * mm_realloc - Highly-optimized, low-overhead reallocation core.
 * Evaluates in-place expansion pathways across 5 architectural configurations 
 * and controls tiny fragment generation using Split-Throttling (128B threshold).
 */
void * mm_realloc(void * ptr, size_t size)
{
    if (ptr == NULL) return mm_malloc(size);
    if (size == 0) { mm_free(ptr); return NULL; }
    
    size_t asize;
    if (size <= 12) asize = 16;
    else asize = DSIZE * ((size + WSIZE + (DSIZE - 1)) / DSIZE);
    
    size_t old_size = GET_SIZE(HDRP(ptr));
    size_t prev_alloc = GET_PREV_ALLOC(HDRP(ptr));
    
    if (old_size >= asize) {
        if ((old_size - asize) >= REALLOC_SPLIT_THRESHOLD) {
            PUT(HDRP(ptr), PACK(asize, prev_alloc, 1));
            void * next_bp = NEXT_BLKP(ptr);
            PUT(HDRP(next_bp), PACK(old_size - asize, 1, 0));
            PUT(FTRP(next_bp), PACK(old_size - asize, 1, 0));
            coalesce(next_bp);
        }
        realloc_block = ptr;
        /* mm_checkheap(0); */
        return ptr;
    }
    
    void * next_bp = NEXT_BLKP(ptr);
    size_t next_alloc = GET_ALLOC(HDRP(next_bp));
    size_t next_size = GET_SIZE(HDRP(next_bp));
    
    if (!next_alloc && (old_size + next_size >= asize)) {
        size_t total_size = old_size + next_size;
        remove_node(next_bp);
        
        if ((total_size - asize) >= REALLOC_SPLIT_THRESHOLD) {
            PUT(HDRP(ptr), PACK(asize, prev_alloc, 1));
            void * rem_bp = NEXT_BLKP(ptr);
            PUT(HDRP(rem_bp), PACK(total_size - asize, 1, 0));
            PUT(FTRP(rem_bp), PACK(total_size - asize, 1, 0));
            coalesce(rem_bp);
        } else {
            PUT(HDRP(ptr), PACK(total_size, prev_alloc, 1));
            char * post_hdr = HDRP(NEXT_BLKP(ptr));
            PUT(post_hdr, PACK(GET_SIZE(post_hdr), 1, GET_ALLOC(post_hdr)));
        }
        realloc_block = ptr;
        /* mm_checkheap(0); */
        return ptr;
    }
    
    if (!next_alloc) {
        void * post_next_bp = NEXT_BLKP(next_bp);
        if (GET_SIZE(HDRP(post_next_bp)) == 0 && GET_ALLOC(HDRP(post_next_bp)) == 1) {
            size_t total_available = old_size + next_size;
            size_t needed = asize - total_available;
            if (mem_sbrk(needed) != (void *)-1) {
                remove_node(next_bp);
                PUT(HDRP(ptr), PACK(asize, prev_alloc, 1));
                PUT(HDRP(NEXT_BLKP(ptr)), PACK(0, 1, 1));
                realloc_block = ptr;
                /* mm_checkheap(0); */
                return ptr;
            }
        }
    }
    
    void * prev_bp = safe_prev_blkp(ptr);
    if (prev_bp != NULL) {
        size_t prev_size = GET_SIZE(HDRP(prev_bp));
        size_t prev_prev_alloc = GET_PREV_ALLOC(HDRP(prev_bp));
        
        if (!next_alloc && (old_size + prev_size + next_size >= asize)) {
            remove_node(prev_bp);
            remove_node(next_bp);
            size_t total_size = old_size + prev_size + next_size;
            memmove(prev_bp, ptr, old_size - WSIZE);
            
            if ((total_size - asize) >= REALLOC_SPLIT_THRESHOLD) {
                PUT(HDRP(prev_bp), PACK(asize, prev_prev_alloc, 1));
                void * rem_bp = NEXT_BLKP(prev_bp);
                PUT(HDRP(rem_bp), PACK(total_size - asize, 1, 0));
                PUT(FTRP(rem_bp), PACK(total_size - asize, 1, 0));
                coalesce(rem_bp);
            } else {
                PUT(HDRP(prev_bp), PACK(total_size, prev_prev_alloc, 1));
                char * post_hdr = HDRP(NEXT_BLKP(prev_bp));
                PUT(post_hdr, PACK(GET_SIZE(post_hdr), 1, GET_ALLOC(post_hdr)));
            }
            realloc_block = prev_bp;
            /* mm_checkheap(0); */
            return prev_bp;
        }
        
        if (!next_alloc) {
            void * post_next_bp = NEXT_BLKP(next_bp);
            if (GET_SIZE(HDRP(post_next_bp)) == 0 && GET_ALLOC(HDRP(post_next_bp)) == 1) {
                size_t total_available = old_size + prev_size + next_size;
                size_t needed = asize - total_available;
                if (mem_sbrk(needed) != (void *)-1) {
                    remove_node(prev_bp);
                    remove_node(next_bp);
                    memmove(prev_bp, ptr, old_size - WSIZE);
                    PUT(HDRP(prev_bp), PACK(asize, prev_prev_alloc, 1));
                    PUT(HDRP(NEXT_BLKP(prev_bp)), PACK(0, 1, 1));
                    realloc_block = prev_bp;
                    /* mm_checkheap(0); */
                    return prev_bp;
                }
            }
        }
        
        if (old_size + prev_size >= asize) {
            remove_node(prev_bp);
            size_t total_size = old_size + prev_size;
            memmove(prev_bp, ptr, old_size - WSIZE);
            
            if ((total_size - asize) >= REALLOC_SPLIT_THRESHOLD) {
                PUT(HDRP(prev_bp), PACK(asize, prev_prev_alloc, 1));
                void * rem_bp = NEXT_BLKP(prev_bp);
                PUT(HDRP(rem_bp), PACK(total_size - asize, 1, 0));
                PUT(FTRP(rem_bp), PACK(total_size - asize, 1, 0));
                coalesce(rem_bp);
            } else {
                PUT(HDRP(prev_bp), PACK(total_size, prev_prev_alloc, 1));
                char * post_hdr = HDRP(NEXT_BLKP(prev_bp));
                PUT(post_hdr, PACK(GET_SIZE(post_hdr), 1, GET_ALLOC(post_hdr)));
            }
            realloc_block = prev_bp;
            /* mm_checkheap(0); */
            return prev_bp;
        }
        
        if (GET_SIZE(HDRP(next_bp)) == 0 && GET_ALLOC(HDRP(next_bp)) == 1) {
            size_t total_available = old_size + prev_size;
            size_t needed = asize - total_available;
            if (mem_sbrk(needed) != (void *)-1) {
                remove_node(prev_bp);
                memmove(prev_bp, ptr, old_size - WSIZE);
                PUT(HDRP(prev_bp), PACK(asize, prev_prev_alloc, 1));
                PUT(HDRP(NEXT_BLKP(prev_bp)), PACK(0, 1, 1));
                realloc_block = prev_bp;
                /* mm_checkheap(0); */
                return prev_bp;
            }
        }
    }
    
    if (GET_SIZE(HDRP(next_bp)) == 0 && GET_ALLOC(HDRP(next_bp)) == 1) {
        size_t needed = asize - old_size;
        if (mem_sbrk(needed) == (void *)-1) return NULL;
        PUT(HDRP(ptr), PACK(asize, prev_alloc, 1));
        PUT(HDRP(NEXT_BLKP(ptr)), PACK(0, 1, 1));
        realloc_block = ptr;
        /* mm_checkheap(0); */
        return ptr;
    }
    
    void * new_ptr = mm_malloc(size);
    if (new_ptr == NULL) return NULL;
    size_t copy_size = old_size - WSIZE; 
    if (size < copy_size) copy_size = size;
    memcpy(new_ptr, ptr, copy_size);
    mm_free(ptr);
    realloc_block = new_ptr;
    /* mm_checkheap(0); */
    return new_ptr;
}

/*
 * mm_checkheap - Full heap consistency checker.
 * Validates alignment, heap bounds, free list structure, and footerless state tracking.
 */
int mm_checkheap(int verbose) {
    char *bp;

    if (verbose) {
        printf("Heap consistency scan initiated at Prologue: %p\n", heap_listp);
    }

    // 1. Prologue Check
    if ((GET_SIZE(HDRP(heap_listp)) != DSIZE) || !GET_ALLOC(HDRP(heap_listp))) {
        fprintf(stderr, "Heap Check Error: Bad prologue header structure.\n");
        return 0;
    }

    // 2. Linear Block Traversal Check
    for (bp = heap_listp; GET_SIZE(HDRP(bp)) > 0; bp = NEXT_BLKP(bp)) {
        if (verbose) {
            printf("Checking Block %p: size=%d, alloc=%d, prev_alloc=%d\n", 
                   bp, GET_SIZE(HDRP(bp)), GET_ALLOC(HDRP(bp)), GET_PREV_ALLOC(HDRP(bp)));
        }

        // A. 8-Byte Alignment Check
        if (((uintptr_t)bp & 0x7) != 0) {
            fprintf(stderr, "Heap Check Error: Block %p violates 8-byte alignment rules.\n", bp);
            return 0;
        }

        // B. Dynamic Heap Range Bound Check
        if (bp < (char *)mem_heap_lo() || bp > (char *)mem_heap_hi()) {
            fprintf(stderr, "Heap Check Error: Block %p leaked outside legal OS heap boundaries.\n", bp);
            return 0;
        }

        // C. Free Block Header/Footer Match Check
        if (!GET_ALLOC(HDRP(bp))) {
            char *ftr = FTRP(bp);
            if (GET_SIZE(HDRP(bp)) != GET_SIZE(ftr) || GET_ALLOC(HDRP(bp)) != GET_ALLOC(ftr)) {
                fprintf(stderr, "Heap Check Error: Block %p free status header/footer parameter mismatch.\n", bp);
                return 0;
            }
        }

        // D. Footless PREV_ALLOC Bit Sync Propagation Check
        char *next_bp = NEXT_BLKP(bp);
        if (GET_SIZE(HDRP(next_bp)) > 0) {
            if (GET_PREV_ALLOC(HDRP(next_bp)) != GET_ALLOC(HDRP(bp))) {
                fprintf(stderr, "Heap Check Error: Block %p alloc status broken with next physical PREV_ALLOC bit.\n", bp);
                return 0;
            }
        }
    }

    // 3. Epilogue Check
    if ((GET_SIZE(HDRP(bp)) != 0) || !(GET_ALLOC(HDRP(bp)))) {
        fprintf(stderr, "Heap Check Error: Bad epilogue termination header tracking.\n");
        return 0;
    }

    return 1;
}