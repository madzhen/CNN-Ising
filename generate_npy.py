import numpy as np




Train_Set1=np.load("T_S1.npy")
Train_Set2=np.load("T_S2.npy")
Train_Set3=np.load("T_S3.npy")
Train_Set4=np.load("T_S4.npy")

Train_Set=np.concatenate((Train_Set1,Train_Set2,Train_Set3,Train_Set4))

#Train_Set=Train_Set.reshape(100*100000,256)

#Train_Set=np.load("Train_Set.npy")

Train_Set_Extend_label=np.zeros((100,100000,356),dtype=np.int8)

for i in range(100):
    print(i)
    a = np.zeros(100)
    a[i] = 1
    for j in range(100000):
        Train_Set_Extend_label[i][j][0:256]=Train_Set[i][j]
        Train_Set_Extend_label[i][j][256:356]=a
#np.save("Train_Set_Extend_label.npy",Train_Set_Extend_label)

Train_Set_Extend_label=Train_Set_Extend_label.reshape(100*100000,356)

np.random.shuffle(Train_Set_Extend_label)
np.random.shuffle(Train_Set_Extend_label)
np.random.shuffle(Train_Set_Extend_label)


Train_Set_out=np.zeros((100*100000,256),dtype=np.int8)

label_Set_out=np.zeros((100*100000,100),dtype=np.int8)
#Train_Set_all=np.load("Train_Set_shuffle.npy")


for i in range(100*100000):
    if(i%100000)==0:
        print(i/100000)
    Train_Set_out[i]=Train_Set_Extend_label[i][0:256]
    label_Set_out[i]=Train_Set_Extend_label[i][256:356]


np.save("Train_Set_final_100tho.npy",Train_Set_out)
np.save("label_Set_final_100tho.npy",label_Set_out)

