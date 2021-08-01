# Linear Regression

### Main Concept:
- A linear model makes a prediction by simply computing a weighted sum of the input features, plus a constant called the bias term (also called the intercept term)

![2021-08-01 07_43_21-Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_ Concepts, To](https://user-images.githubusercontent.com/44503223/127771299-41aa6965-6a88-436a-a6a3-39e5152fd554.png)

- The SVD approach used by Scikit-Learn’s Linear Regression class is about O(n^2). If we double the number of features, we multiply the computation time by roughly 4. 
- Both the Normal Equation and the SVD approach get very slow when the number of features grows large (e.g., 100,000). In this case, Gradient Descent might be preferred.

### Procedures:

#### 1. Simulate Some Random Data

```Python
import numpy as np
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
```

![download (1)](https://user-images.githubusercontent.com/44503223/127771210-ee8c87ad-934e-48c4-b333-b293879dd9fd.png)


#### 2. Perform Linear Regression using Scikit-Learn

```Python
lin_reg = LinearRegression()
lin_reg.fit(X, y)
lin_reg.intercept_, lin_reg.coef_
```

#### 3. Visualize the Results

![download](https://user-images.githubusercontent.com/44503223/127771201-9cd7b143-9e9d-4e53-8de3-a170f7f7e4b1.png)


## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io).

## Reference

This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
- StatQuest: https://statquest.org
