# Self-defined greedy decoding
Time taken for golden greedy decoding without KV cache:  5.7610249519348145
Time taken for customized greedy decoding:  3.4918675422668457
Time taken for customized greedy decoding without KV cache:  4.3078858852386475 


# Memory results and time taken:
Memory results:config: baseline
baseline: 494.95 MB
final: 495.79 MBpeak: 499.89 MBrepeat: 10 timesavg time: 1.70 savg token num: 1.0config: kv_cachebaseline: 536.52 MBfinal: 541.07 MB
peak: 579.15 MB
repeat: 10 times
avg time: 1.89 s
avg token num: 1.0

config: fp16
baseline: 174.99 MB
final: 176.86 MB
peak: 645.67 MB
repeat: 10 times
avg time: 3.49 s
avg token num: 1.0

config: int8
baseline: 310.69 MB
final: 312.56 MB
peak: 645.67 MB
repeat: 10 times
avg time: 3.47 s
avg token num: 1.0