make benchmarks
echo "Runnning 512B page size"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 1 -p 512  ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 1 -p 512  ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 512  ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 512  ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 512  ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 512  ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 4KB page size"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 1 -p 4096 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 1 -p 4096 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 4096 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 4096 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 4096 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 4096 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 8KB page size"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 1 -p 8192 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 1 -p 8192 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 8192 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 8192 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 8192 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p 8192 ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 32KB page size"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 1 -p $((32*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 1 -p $((32*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p $((32*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p $((32*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p $((32*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 1 -p $((32*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 4KB page size - 2 ctrl"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 2 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 2 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 2 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 2 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 2 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 2 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 4KB page size - 4 ctrl"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 4 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 4 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 4 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 4 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 4 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 4 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 4KB page size - 6 ctrl"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 6 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 6 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 6 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 6 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 6 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 6 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

echo "Runnning 4KB page size - 8 ctrl"
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel        -l $((1024*1024*1024*0))   --impl_type 4  --memalloc 6  --n_ctrls 8 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel       -l $((1024*1024*1024*64))  --impl_type 4  --memalloc 6  --n_ctrls 8 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel  -l $((1024*1024*1024*160)) --impl_type 4  --memalloc 6  --n_ctrls 8 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel    -l $((1024*1024*1024*224)) --impl_type 4  --memalloc 6  --n_ctrls 8 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel      -l $((1024*1024*1024*320)) --impl_type 4  --memalloc 6  --n_ctrls 8 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  
#sudo ./bin/nvm-cc-bench -f /nvme0/graphs/EMOGI/sk-2005.bel         -l $((1024*1024*1024*384)) --impl_type 4  --memalloc 6  --n_ctrls 8 -p $((4*1024)) ## -t $((1024*1024*2)) --queue_depth 4096 --num_queues 128  

