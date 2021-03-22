make benchmarks -j
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel.dst -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*0)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/GAP-kron.bel.val -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*32)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel.dst -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*64)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/GAP-urand.bel.val -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*128)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel.dst -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*160)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/com-Friendster.bel.val -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*192)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel.dst -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*224)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/MOLIERE_2016.bel.val -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*288)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel.dst -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*320)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/uk-2007-05.bel.val -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*352)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/sk-2005.bel.dst -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*384)) -o 1
sudo ./bin/nvm-readwrite-bench -f /nvme0/graphs/EMOGI/sk-2005.bel.val -p $((1024*1024*2)) -t $((1024*1024*2)) -b 128 -i 16  --queue_depth 4096 --num_queues 128 -l $((1024*1024*1024*416)) -o 1

