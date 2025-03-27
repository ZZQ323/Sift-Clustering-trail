# Sift-Clustering-trail

## [Datasets](https://github.com/greatzh/Image-Forgery-Datasets-List)


## TEST results
### HAC


### DBSCAN


### BRICH

使用参数：`Birch(n_clusters = 4, threshold = 0.3, branching factor = 30)`

```bash
Copy-Move Forgery Detection performance:
TPR = 94.55%
FPR = 6.36%
```


使用参数：`Birch(n_clusters = 2, threshold = 0.3, branching factor = 30)`

```bash
Copy-Move Forgery Detection performance:
TPR = 74.55%
FPR = 6.36%
Computational time: 00:07:32.5
```

使用参数：`Birch(n_clusters=5, threshold = 0.3, branching factor = 30)`
```bash
Copy-Move Forgery Detection performance:
TPR = 95.45%
FPR = 10.00%
Computational time: 00:06:51.2
```


使用参数：`Birch(n_clusters=7, threshold = 0.1, branching factor = 30)`
```bash
Copy-Move Forgery Detection performance:
TPR = 97.27%
FPR = 9.09%
Computational time: 00:07:18.8
```


使用参数：`Birch(n_clusters=7, threshold = 0.1, branching factor = 10)`
```bash
Copy-Move Forgery Detection performance:
TPR = 97.27%
FPR = 10.91%
Computational time: 00:06:47.8
```

使用参数：`Birch(n_clusters=7, threshold = 0.1, branching factor = 50)`
```bash
Copy-Move Forgery Detection performance:
TPR = 97.27%
FPR = 11.82%
Computational time: 00:08:27.1
```

使用参数：`Birch(n_clusters=9, threshold = 0.1, branching factor = 30)`
```bash
Copy-Move Forgery Detection performance:
TPR = 95.45%
FPR = 11.82%
Computational time: 00:07:21.3
```


使用参数：`Birch(n_clusters=8, threshold = 0.1, branching factor = 35)`
```bash
Copy-Move Forgery Detection performance:
TPR = 97.27%
FPR = 11.82%
Computational time: 00:07:03.9
```


使用参数：`Birch(n_clusters=8, threshold = 0.1, branching factor = 20)`
```bash
Copy-Move Forgery Detection performance:
TPR = 97.27%
FPR = 10.91%
```

使用参数：`Birch(n_clusters=8, threshold = 0.1, branching factor = 10)`
```bash
Copy-Move Forgery Detection performance:
TPR = 97.27%
FPR = 10.91%
```


默认参数：

```py
threshold:FLoat 0.5,
branching factor:Int 50,
n_clusters:int | None = 3,
```

```bash
Copy-Move Forgery Detection performance:
TPR = 91.82%
FPR = 5.45%
Computational time: 00:07:34.4
```
