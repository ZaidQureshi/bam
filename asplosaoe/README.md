# ASPLOS AOE 

BaM work is submitted for Artifact Available and Functional ACM Badges. 
Reproducible Badge requires access to propretiary codebase and large datasets that may not feasible to complete during the AoE review process. 
As BaM is requires hardware and software customization across the stack, for AOE we provide remote access to our mini-prototype machine. 
As such it is important to understand not all results presented in the BaM paper will be reproducible due to differences in system capabilities. 
However, we have strived hard to enable to show the functional validation of the prototype system in this README.md

The AoE evaulation is performed based on individual figures provided in the paper. 
Primarily, we want to establish following key things:

1) BaM codebase is functional and supports the claims described in the paper
2) BaM codebase works with single SSD (upto 2 SSD can be tested on this system) 
3) BaM codebase works across different types of SSDs (Samsung 980 pro and Intel Optane SSDs) are provided.
4) BaM codebase can saturate peak throughput of underlying storage system (if single Intel Optane SSD, we can hit 5M IOPs for 512B access, 1.5M for 4KB)
5) BaM codebase can work with different applications - microbenchmarks and graph applications are provided. 


This README.md provides necessarily command line arguments required to establish above mentioned goals. 
The README.md is structured such that we go over each Figures in the paper individually, describe if they are runnable in this mini-prototype system and what to expect to see as output from the experiments. 
If  you run into any troubles during the AoE process, please reach out over comments in HotCRP page. 

