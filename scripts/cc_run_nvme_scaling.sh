echo "==============================================="
echo "Running GAP-urand with GPU 0 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 1 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 2 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 0 --threads 64


echo "==============================================="
echo "Running GAP-urand with GPU 5 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 1 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 2 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 5 --threads 64


echo "==============================================="
echo "Running GAP-kron with GPU 0 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel      -l $((1024*1024*1024*0))  --impl_type 3 --memalloc 6  --n_ctrls 1 -p 8192 --gpu 0 --threads 64
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel      -l $((1024*1024*1024*0))  --impl_type 3 --memalloc 6  --n_ctrls 2 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 0 --threads 64


echo "==============================================="
echo "Running GAP-kron with GPU 5 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel      -l $((1024*1024*1024*0))  --impl_type 3 --memalloc 6  --n_ctrls 1 -p 8192 --gpu 5 --threads 64
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel      -l $((1024*1024*1024*0))  --impl_type 3 --memalloc 6  --n_ctrls 2 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 5 --threads 64

echo "==============================================="
echo "Running com-Friendster with GPU 0 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 3  --memalloc 6  --n_ctrls 1 -p 8192 --gpu 0 --threads 64 
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 3  --memalloc 6  --n_ctrls 2 -p 8192 --gpu 0 --threads 64 
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 0 --threads 64


echo "==============================================="
echo "Running com-Friendster with GPU 5 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 3  --memalloc 6  --n_ctrls 1 -p 8192 --gpu 5 --threads 64 
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 3  --memalloc 6  --n_ctrls 2 -p 8192 --gpu 5 --threads 64 
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 5 --threads 64

echo "==============================================="
echo "Running MOILERE with GPU 0 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 3  --memalloc 6  --n_ctrls 1 -p 8192 --gpu 0 --threads 64  ## -t $((1024*1024*2)) --queue_depth 8192 --num_queues 128  
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 3  --memalloc 6  --n_ctrls 2 -p 8192 --gpu 0 --threads 64  ## -t $((1024*1024*2)) --queue_depth 8192 --num_queues 128  
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 0 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 0 --threads 64


echo "==============================================="
echo "Running MOILERE with GPU 5 and Page Size 8192"
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 3  --memalloc 6  --n_ctrls 1 -p 8192 --gpu 5 --threads 64  ## -t $((1024*1024*2)) --queue_depth 8192 --num_queues 128  
../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 3  --memalloc 6  --n_ctrls 2 -p 8192 --gpu 5 --threads 64  ## -t $((1024*1024*2)) --queue_depth 8192 --num_queues 128  
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 3 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 4 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 5 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 6 -p 8192 --gpu 5 --threads 64
#../build/bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320))  --impl_type 3 --memalloc 6  --n_ctrls 7 -p 8192 --gpu 5 --threads 64
