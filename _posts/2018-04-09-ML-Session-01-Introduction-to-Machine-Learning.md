---
title: "ML Session 01: Introduction to Machine Learning"
categories:
  - sessions
description: "Session presentation on April 9th, 2018 "
header:
  teaser: https://cdn-images-1.medium.com/max/1024/1*ZkZS46p7Lbw-PDBtPMfEEw.jpeg
tags:
  - announcement
  - session
  - content
toc: true
toc_label: "Sessions"
toc_icon: "gear"
---

## What is Machine Learning?

* **Arthur Samuel (1959)** - Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.  


* **Tom Mitchell (1998)** - Well-posed Learning Problem:(_The Classic Definition of ML_)   
A computer program is said to learn from experience E with respect to some task T and some performance measure P,   
if its performance on T, as measured by P, improves with experience E.

## The essence of mahine learning
* A pattern exists.
* We cannot pin it down mathematially. (not straightforward, or we want to reverse-engineer)
* We have data on it.

![alt]({{"/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/sess_workflow.png"}})

Credits: Learning from Data, Caltech

# Some Terminology
* **Features**  
  – The number of features or distinct traits that can be used to describe  
each item in a quantitative manner.
* **Samples**  
  – A sample is an item to process (e.g. classify). It can be a document, a picture, a sound, a video, a row in database or CSV file, or whatever you can describe with a fixed set of quantitative traits.
* **Feature vector**  
  – is an n-dimensional vector of numerical features that represent some
object.

*  **Feature extraction**  
  – Preparation of feature vector  
  – transforms the data in the high-dimensional space to a space of
  fewer dimensions.  
* **Training/Evolution set**
  – Set of data to discover potentially predictive relationships.

## The Basic premise of learning

* Using a set of observations to uncover an underlying process
*broad premise =⇒ many variations*
*  Supervised Learning
* Unsupervised Learning
* Reinforcement Learning

## Supervised learning

Supervised learning regroups different techniques which all share the same principles:

* The training dataset contains inputs data ( your predictors ) and the value you want to predict (which can be numeric or not).  

* The model will use the training data to learn a link between the input and the outputs. Underlying idea is that the training data can be generalized and that the model can be used on new data with some accuracy.

## Unsupervised learning

On the other hand, unsupervised learning does not use output data (at least output data that are different from the input). Unsupervised algorithms can be split into different categories:

* Clustering algorithm, such as K-means, hierarchical clustering or mixture models. These algorithms try to discriminate and separate the observations in different groups.

* Dimensionality reduction algorithms (which are mostly unsupervised) such as PCA, ICA or autoencoder. These algorithms find the best representation of the data with fewer dimensions.

Anomaly detections to find outliers in the data, i.e. observations which do not follow the data set patterns.

Most of the time unsupervised learning algorithms are used to pre-process the data, during the exploratory analysis or to pre-train supervised learning algorithms.



![Andrew_Course](https://leonardoaraujosantos.gitbooks.io/artificial-inteligence/content/Images/supervised_unsupervised.png)
Credits: _Andrew NG's Coursera Run ML_

## Reinforcement learning

Reinforcement learning algorithms try to find the best ways to earn the greatest reward. Rewards can be winning a game, earning more money or beating other opponents.

They present state-of-art results on very human tasks, for instance, [this paper](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf) from UofT shows how a computer can beat human in old-school Atari video game, which more famously culminated in the Alpha-Go series.

## ML Techniques

_A Broad division_
* **classification:** predict class from observations

* **clustering:** group observations into
“meaningful” groups

* **regression (prediction):** predict value from observations

# Linear and Polynomial Regression
_The Art of Fitting_


The Single Variable Case: Given samples from populations of variables x and y (data points in blue), can we fit a line like the one shown in red?
 ![alt text](https://upload.wikimedia.org/wikipedia/commons/3/3a/Linear_regression.svg)

The fitted line can be represented as $$h_\theta(x)=\theta_0+\theta_1 x$$  

How do we decide the line? Our job is to find parameters $$\theta_0$$ and $$\theta_1$$.  
We simply use a loss function that blows up if we get a wrong fit. And, we choose those parameters which minimize the value of loss function.

One way is simply taking the difference between the value of y and $$h_\theta (x)$$ at each x.  
![alt text](https://upload.wikimedia.org/wikipedia/commons/b/b0/Linear_least_squares_example2.svg)

But, wait a minute!!! We have got negative values for points lying below the fitted line. This means that all points lying below the line give a very low loss function.  

A simple workaround is that we simply square each term before taking the sum.

Usually instead of the sum, the mean is taken (_Doesn't matter though, Why?)._

This loss is known is the mean square loss. The process of choosing those parameters $$\theta_0$$ and $$\theta_1$$ which minimize this loss function, is known as linear least squares regression.

Why linear? Because the parameters represent a linear relationship between the predicted value of y (modellled by $$h_\theta (x)$$).


Let's now get a bit mathematical. The loss function $$J(\theta)$$ for n samples of x and y can be represented as:

$$J(\theta)=\dfrac{1}{n}\sum_{i=1}^{{\displaystyle n}}(y^{(i)}-h_{\theta}(x^{(i)}))^{2}$$, where $$x^{(i)}$$ and $$y^{(i)}$$ represent each individual samples.

It turns out that, those parameters $$\theta_0$$ and $$\theta_1$$ which minimize the above defined loss function can be found out as:  
$$\dfrac{\partial J(\theta)}{\partial\theta_{0}}=\sum_{i=1}^{n}\dfrac{2}{n}(h_{\theta}(x^{(i)})-y^{(i)})\dfrac{\partial h_{\theta}(x^{(i)})}{\partial\theta_{0}}$$  
$$\dfrac{\partial J(\theta)}{\partial\theta_{0}}=\sum_{i=1}^{n}\dfrac{2}{n}(h_{\theta}(x^{(i)})-y^{(i)})$$

Similarly, $$\dfrac{\partial J(\theta)}{\partial\theta_{1}}=\sum_{i=1}^{n}\dfrac{2}{n}(h_{\theta}(x^{(i)})-y^{(i)})\dfrac{\partial h_{\theta}(x^{(i)})}{\partial\theta_{1}}$$  

$$\dfrac{\partial J(\theta)}{\partial\theta_{1}}=\sum_{i=1}^{n}\dfrac{2}{n}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}$$


Setting the partial derivatives as 0, we have,  
$$\sum_{i=1}^{n}\dfrac{2}{n}(h_{\theta}(x^{(i)})-y^{(i)})=0$$  
$$\sum_{i=1}^{n}(\theta_0+\theta_1x^{(i)}-y^{(i)})=0$$ $$\rightarrow1$$   


$$\sum_{i=1}^{n}\dfrac{2}{n}(h_{\theta}(x^{(i)})-y^{(i)})x^{(i)}=0$$  
$$\sum_{i=1}^{n}(\theta_0+\theta_1x^{(i)}-y^{(i)})x^{(i)}=0$$ $$\rightarrow2$$

Solving these 2 equations, we can see that the solution can represented in the form of matrices shown below,

$$\Theta=(X^TX)^{-1}X^TY$$ where
$$X=\begin{bmatrix}1 & x^{(1)}\\
1 & x^{(2)}\\
. & .\\
. & .\\
. & .\\
1 & x^{(n)}
\end{bmatrix}
, Y=\begin{bmatrix}y^{(1)}\\
y^{(2)}\\
.\\
.\\
.\\
y^{(n)}
\end{bmatrix}
 , \Theta=\begin{bmatrix}\theta_{0}\\
\theta_{1}
\end{bmatrix}$$



Multiple Variable Case: Now, we can extend this to higher dimensional x. Until now, we have seen x to be a scalar, we can extend x to be a vector.

**NOTATION:** $$x^{(i)}=[x^{(i)}_1, x^{(i)}_2 ... x^{(i)}_j,...x^{(i)}_m]$$

The approximating function can now be represented as   
$$$$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...\theta_mx_m$$$$

The least squares equation still holds except for changes in the matrices,
$$\Theta=(X^TX)^{-1}X^TY$$
$$X=\begin{bmatrix}1 & x^{(1)}_1 & x^{(1)}_2 & . & . & x^{(1)}_m\\
1 & x^{(2)}_1 & x^{(2)}_2 & . & . & x^{(2)}_{m}\\
. & . & . & . & . & .\\
. & . & . & . & . & .\\
. & . & . & . & . & .\\
1 & x^{(n)}_1 & x^{(n)}_2 & . & . & x^{(n)}_m
\end{bmatrix}
 , Y=\begin{bmatrix}y_{1}\\
y_{2}\\
.\\
.\\
.\\
y_{n}
\end{bmatrix}
 , \Theta=\begin{bmatrix}\theta_{0}\\
\theta_{1}\\
\theta_{2}\\
.\\
.\\
\theta_{m}
\end{bmatrix}$$


## Thats a lot of Calculus, lets see what Linear Algebra has to say !

Lets assume a matrix $$\boldsymbol{A}$$ of size $$N*N$$, a column vector $$\boldsymbol{x}$$ of size $$N*1$$.

$$
\begin{equation}
y = Ax
\end{equation}
$$

Let's assume $$A$$ is a $$3*3$$ Matrix with its entries as follows,

$$
\begin{equation}
A =
\begin{bmatrix}
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3 \\
c_1 & c_2 & c_3  
\end{bmatrix}
\end{equation}$$

and $$x$$ to be a column vector ($$3*1$$)
$$ \begin{equation}
x =
\begin{bmatrix}
x_1  \\
x_2  \\
x_3   
\end{bmatrix}
\end{equation}
$$



Now lets multiply them and see the resulting output vector,
$$
\begin{equation}
y = Ax =
\begin{bmatrix}
a_1 & a_2 & a_3 \\
b_1 & b_2 & b_3 \\
c_1 & c_2 & c_3  
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix} =
\begin{bmatrix}
a_1x_1 + a_2x_2 + a_3x_3 \\
b_1x_1 + b_2x_2 + b_3x_3 \\
c_1x_1 + c_2x_2 + c_3x_3
\end{bmatrix} =
\begin{bmatrix}
  x_1
  \begin{pmatrix}
  a_1 \\
  b_1 \\
  c_1
  \end{pmatrix} +
  x_2
  \begin{pmatrix}
  a_2 \\
  b_2 \\
  c_2
  \end{pmatrix} +
  x_3
  \begin{pmatrix}
  a_3 \\
  b_3 \\
  c_3
  \end{pmatrix}
\end{bmatrix}
\end{equation}
$$
So multiplying a matrix with a column vector returns a vector which is a linear combination of its column vectors (as shown above).

* $$C(A) = $$ It denotes the column space of the matrix $$A$$.
* Column space is the span of all the column vectors of Matrix $$A$$.

For example,

Lets take,
$$
\begin{equation}
\boldsymbol{A} =
\begin{bmatrix}
1 & 2 & 3 \\
2 & 3 & 2 \\
3 & 1 & 1
\end{bmatrix}
\end{equation}
$$
$$
\begin{equation}
y = Ax =
\begin{bmatrix}
1 & 2 & 3 \\
2 & 3 & 2 \\
3 & 1 & 1  
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2 \\
x_3
\end{bmatrix} =
\begin{bmatrix}
1x_1 + 2x_2 + 3x_3 \\
2x_1 + 3x_2 + 2x_3 \\
3x_1 + 1x_2 + x_3
\end{bmatrix} =
\begin{bmatrix}
  x_1
  \begin{pmatrix}
  1 \\
  2 \\
  3
  \end{pmatrix} +
  x_2
  \begin{pmatrix}
  2 \\
  3 \\
  1
  \end{pmatrix} +
  x_3
  \begin{pmatrix}
  3 \\
  2 \\
  1
  \end{pmatrix}
\end{bmatrix}
\end{equation}
$$

### What does this mean??

This matrix multiplication $$\boldsymbol{Ax}$$ cannot represent the entire 3-D plane as the previous matrix, rather what can it represent ??

Since its a linear combination of 2 $$ 3-D$$ vectors, it actually represents a plane in $$3-D$$ space passing through origin if the complete equarion is $$\boldsymbol{Ax} = 0$$

Note the normal to the plane these guys represent in this case is,
$$
\begin{equation}
N =
\begin{pmatrix}
1 \\
1 \\
1
\end{pmatrix}
\times
\begin{pmatrix}
1 \\
2 \\
3
\end{pmatrix}
\end{equation}
$$

## Why did we do all that???

You guys know that equations can be represented in $$Matrix\ form$$.

Lets take an example of 3 simultaneous equations,

$$\begin{equation} 3x + 5y + z = 1\\ 7x – 2y + 4z = 5\\ -6x + 3y + 2z = 10 \end{equation}$$

Whats its matrix representation ??
$$
\begin{equation}
\begin{bmatrix}
3 & 5 & 1 \\
7 & -2 & 4 \\
-6 & 3 & 2
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix} =
\begin{bmatrix}
1 \\
5 \\
10
\end{bmatrix} =
\begin{bmatrix}
  x
  \begin{pmatrix}
  3 \\
  7 \\
  -6
  \end{pmatrix} +
  y
  \begin{pmatrix}
  5 \\
  -2 \\
  3
  \end{pmatrix} +
  z
  \begin{pmatrix}
  1 \\
  4 \\
  2
  \end{pmatrix}
\end{bmatrix}
\end{equation}
$$
Another interpretation for these equations is that whether linear combination of the columns of this matrix can produce the constant vector in the right hand side.

* If it can **solution** exists.
* If the span as seen earlier, doenst cover this constant vector $$\implies$$ solution for the set of equations **Does Not Exist**.

In Linear Regression problems we have many data points, but less number of independent variables. *Its like I need to find the value of 3 variables $$(x, y, z)$$ but to find that I have 10 equations.*

### Lets formalize the Regression problem once again.
$$
\begin{equation}
h_{\theta}(x) = \theta_0 + \theta_1x
\end{equation}
$$
We need to find $$\theta_0\ and\ \theta_1$$ from the dataset which has $$N$$ samples.
$$
\begin{equation}
X=\begin{bmatrix}1 & x^{(1)}\\
1 & x^{(2)}\\
. & .\\
. & .\\
. & .\\
1 & x^{(n)}
\end{bmatrix}
, Y=\begin{bmatrix}y^{(1)}\\
y^{(2)}\\
.\\
.\\
.\\
y^{(n)}
\end{bmatrix}
 , \Theta=\begin{bmatrix}\theta_{0}\\
\theta_{1}
\end{bmatrix}
\end{equation}
$$

We have more equations than unknowns (independent variables). Writing in matrix form, the linear combination of the columns of the $$X$$ neednot (and mostly won't) span matrix $$Y$$ in its column space.

So we must find the nearest possible solution to these set of equations to hold. Its clear that these equations are **inconsistent**.

The equation we have to solve to get the parameters is
$$
\begin{equation}
X_{n \times 2}\Theta_{2 \times 1} = Y_{n \times 1}
\end{equation}
$$
![projection_matrix](http://homepages.dcc.ufmg.br/~vgs/images/blog/lls/projection.jpg)

Following Observations from the image above:

* The vector $$b$$ in the image corresponds to $$Y$$ in the equation.
* The plane corresponds the column space of $$X$$. Any vector in that plane can be represented as a linear combination of the columns of the matrix $$X$$.
* Since the $$Y$$ doesnt lie in the column space of $$X$$, solution cannot exist. So we find the projection of $$Y$$ onto the column space of $$X$$, we get the closest possible solution.


Lets define the error vector as the vector that needs to be added to the projection to get back the original vector ($$Y$$).
$$
\begin{equation}
b = Projection + e \\
Projection\ \epsilon\ C(A) \implies Projection\ = Ax \\
Where\ x\ is\ any\ vector.
\end{equation}
$$
Its clear that $$e$$ is $$\bot$$ to $$All\ Columns\ of\ A_{m \times n}\ \implies Dot\ Product\ is\ 0\ with\ all\ columns$$



Let the columns be $$a_1, a_2, ......... a_n$$
$$
\begin{equation}
a_1^T(b - Ax) = 0 \\
a_2^T(b - Ax) = 0 \\
a_3^T(b - Ax) = 0 \\
\vdots \\
a_n^T(b - Ax) = 0
\end{equation}
$$

This can be written as,

$$
\begin{equation}
\begin{bmatrix}
(a_1^T)_{1\times m} \\
(a_2^T)_{1\times m} \\
(a_3^T)_{1\times m} \\
\vdots \\
(a_n^T)_{1\times m}
\end{bmatrix}_{n \times m}
(b - Ax)_{m \times 1} = 0 \implies A^T(b - Ax) = 0
\end{equation}
$$_

And hence we get the equation,
$$
\begin{equation}
A^Tb - A^TAx =   0\\
A^TAx = A^Tb \\
\end{equation}
$$

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
# Create Polynomial Features
def poly_create(x,n_poly):
  x_=np.zeros((np.size(x),n_poly))
  for i in range(n_poly):
      x_[:,i]=np.power(x,i+1)
  return x_

#sample x and return it along with y
def sampler(k,n_feat,sample_size,params,Poly=False,eps=0):
  noise=eps*np.random.normal(0,100*k,sample_size)
  if not Poly:
    x=np.random.normal(loc=0,scale=k,size=(sample_size,n_feat))
    y=np.matmul(np.transpose(params[1:]),np.transpose(x))+params[0]+noise
  if Poly:
    x=np.random.normal(loc=0,scale=k,size=sample_size)
    x_=poly_create(x,n_feat)
    y=np.matmul(np.transpose(params[1:]),np.transpose(x_))+params[0]+noise
  return x,y

def lstq_fitter(X,y):
  XTX=np.matmul(np.transpose(X),X)
  XTy=np.matmul(np.transpose(X),y)
  params=np.matmul(np.linalg.inv(XTX),XTy)
  return params
```

## Linear Regression
Let's now implement a linear regression model on our own.


```python
n_feat=1
train_size=50
validate_size=5
sample_size=train_size+validate_size
x,y=sampler(5,n_feat,train_size,np.array([1,2]),eps=5e-3)
x_train=x[:train_size] #train
x_validate=x[-validate_size:]
y_train=y[:train_size]
y_validate=y[-validate_size:]
```


```python
X_t=np.ones((train_size,n_feat+1))
X_t[:,1:]=x_train
X_val=np.ones((validate_size,n_feat+1))
X_val[:,1:]=x_validate
params=lstq_fitter(X_t,y_train)
plt.plot(x_train,y_train,'bo')
plt.plot(x_validate,y_validate,'go')
plt.plot(x_train,np.matmul(params,np.transpose(X_t)),'r')
plt.plot(x_validate,np.matmul(params,np.transpose(X_val)),'yo')
plt.legend(['train pts','validate pts','fitted line','prediction over validation set'])
plt.show()
```


![png]({{"/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_38_0.jpg"}})


### Polynomial Regression

In polynomial regression, we use higher order powers of x to fit the model. The idea is very simple. The hypothesis function in this case is:
 $$ h_\theta(x)=\theta_0+\theta_1 x+\theta_2x^2+...+\theta_nx^n $$ for a polynomial of order n.

The multivariate least squares equation remains the same with the only change being, instead of features $$x^{(i)}_j$$ we have $$(x^{(i)})^j$$ ie. the vector variable x is now the vector of multiple powers of x.


The loss function changes as : $$ J(\theta)=\dfrac{1}{n}\sum_{i=1}^{{\displaystyle n}}(y^{(i)}-h_{\theta}(x^{(i)}))^{2}$$
The least square equation remains the same except for what the matrices represent:
$$\Theta=(X^TX)^{-1}X^TY$$
$$
X=\begin{bmatrix}1 & x^{(1)} & (x^{(1)})^{2} & . & . & (x^{(1)})^{m}\\
1 & x^{(2)} & (x^{(2)})^{2} & . & . & (x^{(2)})^{m}\\
. & . & . & . & . & .\\
. & . & . & . & . & .\\
. & . & . & . & . & .\\
1 & x^{(n)} & (x^{(n)})^{3} & . & . & (x^{(n)})^{m}
\end{bmatrix}
 ,

Y=\begin{bmatrix}y_{1}\\
y_{2}\\
.\\
.\\
.\\
y_{n}
\end{bmatrix}
 ,

\Theta=\begin{bmatrix}\theta_{0}\\
\theta_{1}\\
\theta_{2}\\
.\\
.\\
\theta_{m}
\end{bmatrix}$$


### Let's implement a model on our own.



```python
poly_order=2
test_size=501 #just to create a linspace array for visualising fitted function
train_size=100
validate_size=20
sample_size=train_size+validate_size
x,y=sampler(5,poly_order,sample_size,np.array([1,2,1]),eps=1e-2,Poly=True)
x_train=x[:train_size]
x_validate=x[-validate_size:]
y_train=y[:train_size]
y_validate=y[-validate_size:]
```

## Just About Right Fit


```python
poly_fit=2 # order of polynomial to be fitted
X_train=np.ones((train_size,poly_fit+1))
X_train[:,1:]=poly_create(x_train,poly_fit)
params=lstq_fitter(X_train,y_train)
X_val=np.ones((validate_size,poly_fit+1))
X_val[:,1:]=poly_create(x_validate,poly_fit)
x_test=poly_create(np.linspace(-10,10,test_size),poly_fit)
X_test=np.ones((test_size,poly_fit+1))
X_test[:,1:]=x_test
y_test=np.matmul(params,np.transpose(X_test))
plt.plot(x_train,y_train,'bo')
plt.plot(x_validate,y_validate,'go')
plt.plot(x_test[:,0],y_test,'r')
plt.plot(x_validate,np.matmul(params,np.transpose(X_val)),'yo')
plt.legend(['train pts','validate pts','fitted line','prediction over validation set'])
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_44_0.png)


## Model With High Bias


```python
poly_fit=1 # order of polynomial to be fitted
X_train=np.ones((train_size,poly_fit+1))
X_train[:,1:]=poly_create(x_train,poly_fit)
params=lstq_fitter(X_train,y_train)
X_val=np.ones((validate_size,poly_fit+1))
X_val[:,1:]=poly_create(x_validate,poly_fit)
x_test=poly_create(np.linspace(-10,10,test_size),poly_fit)
X_test=np.ones((test_size,poly_fit+1))
X_test[:,1:]=x_test
y_test=np.matmul(params,np.transpose(X_test))
plt.plot(x_train,y_train,'bo')
plt.plot(x_validate,y_validate,'go')
plt.plot(x_test[:,0],y_test,'r')
plt.plot(x_validate,np.matmul(params,np.transpose(X_val)),'yo')
plt.legend(['train pts','validate pts','fitted line','prediction over validation set'])
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_46_0.png)


## Model With High Variance


```python
poly_fit=8 # order of polynomial to be fitted
y_train[99]=1000
X_train=np.ones((train_size,poly_fit+1))
X_train[:,1:]=poly_create(x_train,poly_fit)
params=lstq_fitter(X_train,y_train)
X_val=np.ones((validate_size,poly_fit+1))
X_val[:,1:]=poly_create(x_validate,poly_fit)
x_test=poly_create(np.linspace(-10,10,test_size),poly_fit)
X_test=np.ones((test_size,poly_fit+1))
X_test[:,1:]=x_test
y_test=np.matmul(params,np.transpose(X_test))
plt.plot(x_train,y_train,'bo')
plt.plot(x_validate,y_validate,'go')
plt.plot(x_test[:,0],y_test,'r')
plt.plot(x_validate,np.matmul(params,np.transpose(X_val)),'yo')
plt.legend(['train pts','validate pts','fitted line','prediction over validation set'])
plt.ylim([-100,200])
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_48_0.png)


## Bias and Variance
![alt text](https://www.learnopencv.com/wp-content/uploads/2017/02/Bias-Variance-Tradeoff-In-Machine-Learning-1.png)

# Logistic Regression
In statistics, logistic regression, or logit regression, or logit model[1] is a regression model where the dependent variable (DV) is categorical.  
Logistic Regression is used for doing classification using regression.

A particular case of the discriminative approach (fit $$$$ p (y|{x}) $$$$ directly, where x is the feature vector, y is the prediction) to building a probabilistic classifier, logisitic regression (despite the name being a misnomer), assumes parameters to be linear.

The steps involved in any machine learning model:
1. Train data  
2. Test data  
3. Model architecture  
4. Cost function.  

Model architecture is decided based on the nature of the data by dong some of the preprocessing techniques like boxplot, scatterplot etc.  
The cost function is the objective function that has to be optimized(here it is to be minimized) or the log_likelihood has to be maximised


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))
```


```python
plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4,)
```




    <matplotlib.collections.PathCollection at 0x1191f2780>




![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_52_1.png)



```python
def sigmoid(scores):
    return 1 / (1 + np.exp(-scores))
```

# Maximizing the Likelihood
To maximize the likelihood, I need a way to compute the likelihood and the gradient of the likelihood. Fortunately, the likelihood (for binary classification) can be reduced to a fairly intuitive form by switching to the log-likelihood. We're able to do this without affecting the weights parameter estimation because log transformation are monotonic.

For anyone interested in the derivations of the functions I'm using, check out Section 4.4.1 of Hastie, Tibsharani, and Friedman's Elements of Statistical Learning.

## Calculating the Log-Likelihood

The mean log-likelihood can be viewed as as sum over all the training data. Mathematically,

$$$$\begin{equation}
mll = \frac{1}{N} \sum_{i=1}^{N}y_{i}\beta ^{T}x_{i} - log(1+e^{\beta^{T}x_{i}})
\end{equation}$$$$

where $$y$$ is the target class, $$x_{i}$$ represents an individual data point, and $$\beta$$ is the weights vector.

I can easily turn that into a function and take advantage of matrix algebra.

## Calculating the Gradient

Now I need an equation for the gradient of the log-likelihood. By taking the derivative of the equation above and reformulating in matrix form, the gradient becomes:

$$$$\begin{equation}
\bigtriangledown ll = X^{T}(Y - Predictions)
\end{equation}$$$$


```python
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))

    weights = np.zeros(features.shape[1])

    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions

        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            mean_log_likelihood = np.mean( target*scores - np.log(1 + np.exp(scores)) )
            print(mean_log_likelihood)


    return weights
```


```python
weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)
```

    -0.6931471805599453
    -0.01487079575635989
    -0.01429651811212252
    -0.014154537992689521
    -0.014106034850122885



```python
print(weights)
```

    [-13.58690551  -4.8809644    7.99812915]


# Comparing to Sk-Learn's LogisticRegression
The obtain_weights can be compared with the weights from sk-learn's logistic regression function. They should be the same if I did everything correctly. Since sk-learn's `LogisticRegression` automatically regularizes (which I didn't do), I set `C=1e15` to essentially turn off regularization.


```python
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(fit_intercept=True, C = 1e15)
clf.fit(simulated_separableish_features, simulated_labels)
```




    LogisticRegression(C=1000000000000000.0, class_weight=None, dual=False,
              fit_intercept=True, intercept_scaling=1, max_iter=100,
              multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
              solver='liblinear', tol=0.0001, verbose=0, warm_start=False)




```python
print(clf.intercept_, clf.coef_)
print(weights)
```

    [-13.99400797] [[-5.02712572  8.23286799]]
    [-13.58690551  -4.8809644    7.99812915]


# Accuracy
To get the accuracy, I just need to use the final weights to get the logits for the dataset (`final_scores`). Then I can use `sigmoid` to get the final predictions and round them to the nearest integer (0 or 1) to get the predicted class.


```python
final_scores = np.dot(np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                 simulated_separableish_features)), weights)
preds = np.round(sigmoid(final_scores))

print ('Accuracy from our program:',format((preds == simulated_labels).sum().astype(float) / len(preds)))
print ('Accuracy from sk-learn:',format(clf.score(simulated_separableish_features, simulated_labels)))
```

    Accuracy from our program: 0.9948
    Accuracy from sk-learn: 0.9948


Plotting the points where the predictions were wrong


```python
plt.figure(figsize = (12, 8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = preds == simulated_labels - 1, alpha = .8, s = 50)
```




    <matplotlib.collections.PathCollection at 0x10c6e1f28>




![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_66_1.png)


# Decision Boundary
Decision Boundary is the curve on which the probability of getting predicted as class 1 or class 0 is equal to 0.5.
The context of the above mentioned log_likelihood maximization is it the curve where the hypothesis is zero,i.e
\begin{equation}
y - \beta ^{T}x = 0
\end{equation}
Plotting the decision boundary


```python
px1 = np.linspace(-5, 5)
px2 = -(weights[1]/weights[2])*px1 - weights[0]/weights[2]
plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4,)
plt.plot(px1, px2, 'k-')
```




    [<matplotlib.lines.Line2D at 0x10c7c5080>]




![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_68_1.png)


# K Means Clustering
* Step 1 - Pick K random points as cluster centers called centroids.
* Step 2 - Assign each xi to nearest cluster by calculating its distance to each centroid.
* Step 3 - Find new cluster center by taking the average of the assigned points.
* Step 4 - Repeat Step 2 and 3 until none of the cluster assignments change.


## Step 1
 Let’s assume these are c1,c2,…,ck,and we can say that;

                      C=c1,c2,…,ck

C is the set of all centroids.

## Step 2
In this step we assign each input value to closest center. This is done by calculating Euclidean(L2) distance between the point and the each centroid.
\begin{equation}
   argminc_i∈C dist(ci,x)2
\end{equation}
Where dist(.) is the Euclidean distance

## Step 3
In this step, we find the new centroid by taking the average of all the points assigned to that cluster.
\begin{equation}
c_i=\frac{1}{|S_i|} \sum_{x_i \epsilon S_i}x_i
\end{equation}
Si
is the set of all points assigned to the ith cluster.


```python
%matplotlib inline
from copy import deepcopy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

```


```python
X,y = make_blobs(n_samples=3000, n_features=2, centers=3)
plt.scatter(X[:,0],X[:,1], c='black', s=7)

```




    <matplotlib.collections.PathCollection at 0x1181f5048>




![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_74_1.png)



```python
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
```


```python
# Number of clusters
k = 3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X), size=k)
# Y coordinates of random centroids
C_y = np.random.randint(0, np.max(X), size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
```

    [[1. 0.]
     [0. 1.]
     [7. 1.]]



```python
plt.scatter(X[:,0], X[:,1], c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')
```




    <matplotlib.collections.PathCollection at 0x118508048>




![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_77_1.png)



```python
# To store the value of centroids when it updates
threshold = 2

C_old = np.zeros(C.shape)

# Cluster Lables(0, 1, 2)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
print(error)
# Loop will run till the error becomes zero
while error >= threshold:

    # Assigning each value to its closest cluster\
    print(error)
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)
```

    7.211102550927978
    7.211102550927978


    /Users/Ankivarun/anaconda3/envs/tf_python3/lib/python3.6/site-packages/numpy/core/fromnumeric.py:2957: RuntimeWarning: Mean of empty slice.
      out=out, **kwargs)
    /Users/Ankivarun/anaconda3/envs/tf_python3/lib/python3.6/site-packages/numpy/core/_methods.py:80: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)



```python
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
```


    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-31-a524602c10b3> in <module>()
          3 for i in range(k):
          4         points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
    ----> 5         ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
          6 ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')


    IndexError: too many indices for array



![png](Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_79_1.png)


## Scikit Learn K means Clustering


```python
from sklearn.cluster import KMeans

# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_
```


```python
# Comparing with scikit-learn centroids
print(C) # From Scratch
print(centroids) # From sci-kit learn
```

    [[-6.945285  -9.455215 ]
     [-1.5425268  1.984167 ]
     [       nan        nan]]
    [[-7.99409852 -8.61303692]
     [ 0.35592072  4.59554469]
     [-6.35614693 -9.86715012]]


## Optimal Value of K:
The value of k is tough to guess in the case of higher dimensional data.Therefore , SSE(sum of the squares of the errors) and K are plotted.
After a significantly higher value of K, decrease in error is small. At higher vvalues of K, the amount of computation taken for every updation is more.The marked value of K =3 is the optimal value of the above given data.![](https://i.imgur.com/k3o6NxK.jpg = 20x20)

# K-Nearest Neighbour Classification.

Very Intuitive Idea.

Say you are given a set of points in N-dimensional space. We'll take the case of 2-D for convenience (each point is represented by a vector of dimension 2). Each of this point is also assigned a class.

Now, a new point is given whose class has to be identified.  
We find the k nearest neighbouring points (from the given samples of points) based on L2 norm.(_Infact the norm to choose is another hyper-parameter)


We assign a method of voting for these k neighbours (each of them vote for their class) to determine the class of the new point. The vote of each neighbour may or may not carry the same weight.  
The new point is assigned the class which has the maximum vote.


```python
# Rishi is it? yeah
# Guys get a room
# Use the comments feature
from sklearn import datasets
from sklearn import neighbors
from matplotlib.colors import ListedColormap
```


```python
cmap_space=ListedColormap([[1,0.5,1],[1,0.5,0.5],[0.5,1,0.5],[0.5,0.5,1]])
cmap_train=ListedColormap([[1,0,1],[1,0,0],[0,1,0],[0,0,1]])
X,y=datasets.make_blobs(n_samples=200,n_features=2,random_state=2,centers=4)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_train)
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_87_0.png)


### Let's set the weights parameter to 'distance'.

This will weigh the votes of each neighbour by the inverse of its distance from the given point.

In other words, closer neighbours influence more of the point's class.


```python
KNN_classifier=neighbors.KNeighborsClassifier(n_neighbors=5,weights='distance') #We'll weight the votes based on the nearness of each neighbour
KNN_classifier.fit(X,y)
KNN_classifier.kneighbors_graph()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
pred=KNN_classifier.predict(np.c_[xx.ravel(),yy.ravel()])
pred=pred.reshape(xx.shape)
plt.pcolormesh(xx,yy,pred,cmap=cmap_space)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_train)
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_89_0.png)


In the above plot we see the division of the entire space into 4 regions. The regions denote the class that would be predicted based on our model.

Now let's give uniform weight to all the k neighbours' votes.


```python
KNN_classifier=neighbors.KNeighborsClassifier(n_neighbors=5,weights='uniform') #We'll weight the votes based on the nearness of each neighbour
KNN_classifier.fit(X,y)
KNN_classifier.kneighbors_graph()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
pred=KNN_classifier.predict(np.c_[xx.ravel(),yy.ravel()])
pred=pred.reshape(xx.shape)
plt.pcolormesh(xx,yy,pred,cmap=cmap_space)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_train)
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_91_0.png)


### Now let's increase k to 25


```python
KNN_classifier=neighbors.KNeighborsClassifier(n_neighbors=25,weights='uniform') #We'll weight the votes based on the nearness of each neighbour
KNN_classifier.fit(X,y)
KNN_classifier.kneighbors_graph()
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
pred=KNN_classifier.predict(np.c_[xx.ravel(),yy.ravel()])
pred=pred.reshape(xx.shape)
plt.pcolormesh(xx,yy,pred,cmap=cmap_space)
plt.scatter(X[:,0],X[:,1],c=y,cmap=cmap_train)
plt.show()
```


![png](/assets/images/posts/2018-05-09-Introduction-to-Machine-Learning/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_files/Linear_Regression%2C_Logistic_Regression%2C_KNN%2C_K_Means_Clustering_93_0.png)


# References

Courses:

1. **CS229 (Stanford):** _An immensly popular course (and MOOC), this covers nearly all aspects of ML._

2. **NPTEL ML:** _Covers a broad spectrum of topics, insti's ML course run._

3. **CS231n (Stanford):** _A good start to Deep vision._

A few text references:

1. **PRML, Christopher Bishop:** _A gentle primer on ML_

2. **Probabilistic Approach to ML,Kevin Murphy:** _Not for the faint hearted. 1300 pages, 28 chapters. Oh and the author conceeds this too XD ._

3. **Deep Learning Book, Ian Goodfellow, Yoshua Benjio:** _The gold standard for DL_

An exhuastive list including libraries and frameworks may be found on our blog, [here](https://iitmcvg.github.io/tutorials/getting_started/).
