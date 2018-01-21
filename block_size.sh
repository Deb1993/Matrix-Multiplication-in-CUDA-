#!/bin/bash

cp dgemm-blocked.c temp.c
for b2 in {16..64..16}
do
    for ((b1=2; b1<=b2; b1+=2)); do
         cat temp.c | sed -e "s/#define BLOCK_SIZE .*/#define BLOCK_SIZE $b1/" > temp2.c
         cat temp2.c | sed -e "s/#define BLOCK_SIZE_2 .*/#define BLOCK_SIZE_2 $b2/" > dgemm-blocked.c
         make
         ./benchmark-blocked > "gflops_${b2}_${b1}"
    done
done

rm temp.c
rm temp2.c
