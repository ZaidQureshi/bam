
#echo "==============================================="
#echo "Running NVME GAP-urand with GPU 8 and Page Size 4096"
#../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 1 -p 4096 --gpu 8 --threads 64
#../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 2 -p 4096 --gpu 8 --threads 64
#../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 3 -p 4096 --gpu 8 --threads 64
#../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 4 -p 4096 --gpu 8 --threads 64

echo "==============================================="
echo "Running NVME uk-2007-05 with GPU 8 and Page Size 4096"
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 1 -p 4096 --gpu 8 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 2 -p 4096 --gpu 8 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 3 -p 4096 --gpu 8 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 9 --memalloc 6 --repeat 32  --n_ctrls 4 -p 4096 --gpu 8 --threads 64


#echo "==============================================="
#echo "Running EMOGI GAP-urand with GPU 0"
#../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 7 --memalloc 2 --repeat 32  --n_ctrls 1 -p 4096 --gpu 0 --threads 64

echo "==============================================="
echo "Running EMOGI uk-2007-05 with GPU0"
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 7 --memalloc 2 --repeat 32  --n_ctrls 1 -p 4096 --gpu 0 --threads 64
