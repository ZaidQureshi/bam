for a in $(dmesg|grep "nvme.*pci"|awk '{print $7}'); 
do 
	echo $a
	echo -n $a > /sys/bus/pci/devices/$a/driver/unbind; 
done 
