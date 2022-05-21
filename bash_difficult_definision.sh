#!/bin/bash

declare -A 'host_1'
declare -A 'host_2'

host_1['hostname']='192.xxx.xxx.xx1'
host_1['surface']='ITb1'
host_1['role']='BFF1'
host_1['vm']="VM1_1234 VM2_2345 VM3_3456"

host_2['hostname']='192.xxx.xxx.xx2'
host_2['surface']='ITb2'
host_2['role']='BFF2'
host_2['vm']="VM1_1111 VM2_2222 VM3_3333"

#######################################################
echo ${host_1['hostname']}
echo ${host_1['surface']}
echo ${host_1['role']}
for i in ${host_1['vm']}; 
do 
  #echo ${i} 
  vm=$(echo ${i} | cut -d '_' -f 1)
  IP=$(echo ${i} | cut -d '_' -f 2)
  echo ${vm}"->"${IP}
done


echo ${host_2['hostname']}
echo ${host_2['surface']}
echo ${host_2['role']}
for i in ${host_2['vm']}; 
do 
  #echo ${i} 
  vm=$(echo ${i} | cut -d '_' -f 1)
  IP=$(echo ${i} | cut -d '_' -f 2)
  echo ${vm}"->"${IP}
done

