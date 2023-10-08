import os
# dirname='/Users/tongxueqing/Downloads/HipMRI_study_complete_release_v1/semantic_MRs_anon'
# for i in os.listdir(dirname):
#     if int(i.split('_')[1])<=33:
#         with open('train_list.txt','a') as f:
#             f.write(i+'\n')
#     else:
#         with open('test_list.txt','a') as f:
#             f.write(i+'\n')
with open('train_list.txt','r') as f:
    y=f.readlines()
    print(y)
    print(len(y))
    print(y[0].strip())
