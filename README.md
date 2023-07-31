# SVM model with Sum-Mismatch kernel

This implementation uses an Support Vector Machine(SVM) with Sum-Mismatch kernel to perform classification on DNA sequence data.

### Kernel SVM with Sum-Mismatch kernal

African Institute of Machine Intelligence(AMMI)
Academic year: 2022/23

Group Members:
1. Jean Robin RAHERISAMBATRA jrraherisambatra@aimsammi.org
2. Samuael Adnew Birhane     sadnew@aimsammi.org

Dependencies required: **numpy**, **scipy**, **pandas**, **pickle**, **tqdm**, **cvxpy**, **cvxopt**, **gensim**
1. **numpy** for array related computations.

2. **scipy** for Matrix computation.

3. **pandas** for data loading and saving

4. **pickle** for loading and saving byte map data into and from ```.p``` files.

5. **tqdm** for iteration, feedback, and progress report during computation.

6. **cvxpy** and **cvxopt** for convex optimization problems.

7. **gensim** for Word2Vec embedding creation.

----
### How to run

To generate the prediction outputs, run 
```python main.py```.then press ```Enter```

----
While running the ```main.py``` file, notice that the RBF implementation is added just to show structure. But to see how we attain the Kaggle competition prediction accuracy, you can use the SVM implementation.

``` Default == SVM```