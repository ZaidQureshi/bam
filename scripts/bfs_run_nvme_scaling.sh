echo "==============================================="
echo "Running GAP-urand with GPU 0 and Page Size 8192"
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 1 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 2 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 3 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 4 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 5 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 6 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 7 -p 8192 --gpu 0 --threads 64


echo "==============================================="
echo "Running GAP-urand with GPU 5 and Page Size 8192"
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 1 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 2 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 3 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 4 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 5 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 6 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 7 -p 8192 --gpu 5 --threads 64


echo "==============================================="
echo "Running uk-2007-05 with GPU 0 and Page Size 8192"
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 1 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 2 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 3 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 4 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 5 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 6 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 7 -p 8192 --gpu 0 --threads 64


echo "==============================================="
echo "Running uk-2007-05 with GPU 5 and Page Size 8192"
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 1 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 2 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 3 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 4 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 5 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 6 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-bfs-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6 --repeat 32  --n_ctrls 7 -p 8192 --gpu 5 --threads 64
