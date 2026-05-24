//
//  main.c // use as csim.c in the lab.
//  cacheSim
//
//  Created by Farkladin on 5/23/26.
//

#include "cachelab.h"
#include <stdio.h>
#include <getopt.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <stdint.h>

#ifdef __STDC_VERSION__
#if __STDC_VERSION__ < 202311L
#define nullptr NULL
#endif /* __STDC_VERSION__ >= 202311L */
#endif /* __STDC_VERSION__ */


typedef struct {
    char flag;
    int lruCount;
    void* pTag;
} Type;

void* allocationTable[524288];
static size_t allocationTableSize = 0;

void clearAllocation(void) {
    for (size_t i=0; i < allocationTableSize; ++i) {
        free(allocationTable[i]);
    }
}

void accessCache(Type** table, int s, int E, int b, uint64_t addr, int* hits, int* misses, int* evictions, int* lru_counter, int verbose) {
    uint64_t set_index = (addr >> b) & ((1ULL << s) - 1);
    uint64_t tag = addr >> (s + b);
    
    Type* set = table[set_index];
    
    // Check for hit
    for (int i = 0; i < E; i++) {
        if (set[i].flag && (uintptr_t)set[i].pTag == tag) {
            ++(*hits);
            set[i].lruCount = (*lru_counter)++;
            if (verbose) {
                printf(" hit");
            }
            return;
        }
    }
    
    // Miss
    ++(*misses);
    if (verbose) {
        printf(" miss");
    }
    
    // Find an empty line
    for (int i = 0; i < E; i++) {
        if (!set[i].flag) {
            set[i].flag = 1;
            set[i].pTag = (void*)(uintptr_t)tag;
            set[i].lruCount = (*lru_counter)++;
            return;
        }
    }
    
    // Eviction
    ++(*evictions);
    if (verbose) {
        printf(" eviction");
    }
    
    int min_idx = 0;
    int min_lru = set[0].lruCount;
    for (int i = 1; i < E; i++) {
        if (set[i].lruCount < min_lru) {
            min_lru = set[i].lruCount;
            min_idx = i;
        }
    }
    
    set[min_idx].pTag = (void*)(uintptr_t)tag;
    set[min_idx].lruCount = (*lru_counter)++;
}

int main(int argc, char* argv[]) {
    int s= -1, E= -1, b= -1, v= -1;
    char* t = nullptr;
    int opt;
    
    while ((opt = getopt(argc, argv, "s:E:b:t:hv")) != -1) {
        switch (opt) {
            case 's': {
                s = atoi(optarg);
                break;
            }
            case 'E': {
                E = atoi(optarg);
                break;
            }
            case 'b': {
                b = atoi(optarg);
                break;
            }
            case 't': {
                t = optarg;
                break;
            }
            case 'v': {
                v = 1;
                break;
            }
            case 'h': {
                printf("Usage: ./csim [-hv] -s <s> -E <E> -b <b> -t <tracefile>\n");
                return EXIT_SUCCESS;
            }
            default: {
                printf("Usage: ./csim [-hv] -s <s> -E <E> -b <b> -t <tracefile>\n");
                return EXIT_FAILURE;
            }
        }
    }
    
    if (!s || !E || !b || !t) {
        printf("Usage: ./csim [-hv] -s <s> -E <E> -b <b> -t <tracefile>\n");
        return EXIT_FAILURE;
    }
    
    int S = 1 << s;
    Type** table = malloc(sizeof(Type*) * S);
    allocationTable[allocationTableSize++] = table;
    
    for (int i=0; i < S; ++i) {
        table[i] = malloc(sizeof(Type) * E);
        allocationTable[allocationTableSize++] = table[i];
        
        Type* end = table[i] + E;
        for (Type* it = table[i]; it != end; ++it){
            it->flag = it->lruCount = 0;
            it->pTag = NULL;
        }
    }
    
    int hits = 0, misses = 0, evictions = 0;
    
    FILE* file = fopen(t, "r");
    if (!file) {
        printf("File does not exists.\n");
        return EXIT_FAILURE;
    }
    
    char buf[100];
    int lru_counter = 0;
    while (fgets(buf, sizeof(buf), file)) {
        if (buf[0] == ' ') {
            char op;
            uint64_t addr;
            int size;
            if (sscanf(buf, " %c %llx,%d", &op, &addr, &size) == 3) {
                if (v) {
                    printf("%c %llx,%d", op, addr, size);
                }
                if (op == 'L' || op == 'S') {
                    accessCache(table, s, E, b, addr, &hits, &misses, &evictions, &lru_counter, v);
                } else if (op == 'M') {
                    accessCache(table, s, E, b, addr, &hits, &misses, &evictions, &lru_counter, v);
                    accessCache(table, s, E, b, addr, &hits, &misses, &evictions, &lru_counter, v);
                }
                if (v) {
                    printf("\n");
                }
            }
        }
    }
    
    fclose(file);
    
    printSummary(hits, misses, evictions);
    
    clearAllocation();
    return 0;
}
