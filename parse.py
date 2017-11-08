import os
import pdb

files = os.listdir(os.curdir)

for f in files:
    with open(f,'r') as txt:
        if not ('.txt' in f): continue
        lines = txt.readlines()

        max_ctaid = [0,0,0]
        max_tid = [0,0,0]
        max_cta = 0
        num_kernel = 0

        for line in lines:

            if 'gpu_tot_ipc' in line:
                gpu_tot_ipc = line
                num_kernel = num_kernel + 1

            if 'gpu_tot_sim_insn' in line:
                gpu_tot_insn = line

            if 'ctaid=(' in line:
                ctaid_start = line.index('ctaid')+7
                ctaid_end = line.index(') t')
                ctaid = [int(i) for i in line[ctaid_start:ctaid_end].split(',')]

                tid_start = line.index('tid')+5
                tid_end = len(line)-2
                tid = [int(i) for i in line[tid_start:tid_end].split(',')]

                for i in range(3):
                    if (ctaid[i]>max_ctaid[i]):
                        max_ctaid[i] = ctaid[i]
                    if (tid[i]>max_tid[i]):
                        max_tid[i] = tid[i]

            if 'CTA #' in line:
                start_num_cta = line.index('#')+1
                end_num_cta = line.index(' (')
                num_cta = int(line[start_num_cta:end_num_cta])
                if (num_cta>max_cta): max_cta = num_cta

            if 'L1D_total_cache_accesses' in line:
                D_cache_access = line
            if 'L1D_total_cache_misses' in line:
                D_cache_miss = line


            if 'L2_total_cache_miss_rate' in line:
                L2_cache_miss_rate = line


        num_D_cache_access = int(D_cache_access[D_cache_access.index('= ')+2:])
        num_D_cache_miss = int(D_cache_miss[D_cache_miss.index('= ')+2:])

        total_L1_hit_rate = 1. - (num_D_cache_miss/float(num_D_cache_access))
        total_L2_hit_rate = 1. - float(L2_cache_miss_rate[L2_cache_miss_rate.index('= ')+2:])

        print('------------Stat of {}-------------'.format(f))
        print(gpu_tot_ipc[:-1])
        print(gpu_tot_insn[:-1])
        print('number of kernel executions : {}'.format(num_kernel))
        print('ctaid : [{},{},{}]'.format(max_ctaid[0],max_ctaid[1],max_ctaid[2]))
        print('tid : [{},{},{}]'.format(max_tid[0],max_tid[1],max_tid[2]))
        print('max concurrent cta : {}'.format(max_cta+1))
        print('L1 hit rate : {}'.format(total_L1_hit_rate))
        print('L2_hit_rate : {}'.format(total_L2_hit_rate))
        print('')
