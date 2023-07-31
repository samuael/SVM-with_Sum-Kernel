import os
from svm_sum import sumMismatchBasedSVMKernel
from rbf_ridge_regression import rbfKernel

print("\n DNA sequence classification with Kernel Methods \n\n")
print("Group Members: \nSamuael Adnew \t sadnew@aimsammi.org\nJean Robin RAHERISAMBATRA \t jrraherisambatra@aimsammi.org\n")
def run():
    value = input("To proceed with \n1. SVM with Sum-Mismatch Kernel enter \033[1mreturn\033[0m\n2. Ridge Regression Model with RBF Word2Vec \033[1m 'r' and return\033[0m")
    if value == "r":
        rbfKernel()
    else:
        value = input("Remove and generate a new k-mers and mismatch? \033[1m (y/n) \033[0m")
        if value == "y":
            try:
                files = ["0_5_1", "0_8_1", "0_10_1", "0_12_2", "0_13_2", "0_15_3", "1_5_1", "1_8_1", "1_10_1", "1_12_2", "1_13_2", "1_15_3", "2_5_1", "2_8_1", "2_10_1", "2_12_2", "2_13_2", "2_15_3",]
                for a in files:
                    os.remove(f"generated_kmer_neighbours/{a}.p")
            except:
                pass
        sumMismatchBasedSVMKernel()
run()