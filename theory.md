# Теоретическая часть — ЛР 5

**Студент:** [Имя]  
**Student ID:** 22  
**Вариант Q3:** 1 (Sigmoid + Tanh)  
**Архитектура:** 784 → 256 → 128 → 10  

---

## Q1. Прямой проход

### 1.1

$$z^{(1)}=a^{(0)}W^{(1)}+b^{(1)}$$
$$a^{(1)}=\sigma_{\text{sigmoid}}(z^{(1)})=\frac{1}{1+e^{-z^{(1)}}}$$

$$z^{(2)}=a^{(1)}W^{(2)}+b^{(2)}$$
$$a^{(2)}=\sigma_{\text{tanh}}(z^{(2)})=\frac{e^{z^{(2)}}-e^{-z^{(2)}}}{e^{z^{(2)}}+e^{-z^{(2)}}}$$

$$z^{(3)}=a^{(2)}W^{(3)}+b^{(3)}$$
$$\hat{y}_{ik}=\frac{\exp(z_{ik}^{(3)})}{\sum_{j=1}^{10}\exp(z_{ij}^{(3)})}$$

Размерности: $a^{(0)}\in\mathbb{R}^{N\times784}$, $W^{(1)}\in\mathbb{R}^{784\times256}$, $b^{(1)}\in\mathbb{R}^{256}$, $a^{(1)}\in\mathbb{R}^{N\times256}$, $W^{(2)}\in\mathbb{R}^{256\times128}$, $b^{(2)}\in\mathbb{R}^{128}$, $a^{(2)}\in\mathbb{R}^{N\times128}$, $W^{(3)}\in\mathbb{R}^{128\times10}$, $b^{(3)}\in\mathbb{R}^{10}$, $\hat{Y}\in\mathbb{R}^{N\times10}$.

### 1.2

$$\mathscr{L}(\hat{y}_i,y_i)=-\sum_{k=1}^{K}y_{ik}\log\hat{y}_{ik}$$
$$Q(\boldsymbol{\theta},X^{\ell})=\frac{1}{\ell}\sum_{i=1}^{\ell}\mathscr{L}(\hat{y}_i,y_i)$$

### 1.3

$$\hat{y}_k=\frac{\exp(z_k^{(3)})}{\sum_{j=1}^{K}\exp(z_j^{(3)})}$$

$$\hat{y}_k=\frac{\exp(z_k^{(3)}-m)}{\sum_{j=1}^{K}\exp(z_j^{(3)}-m)},\quad m=\max_j z_j^{(3)}$$

---

## Q2. Обратный проход

### 2.1

$$\frac{\partial Q}{\partial Z^{(3)}}=\frac{1}{N}(\hat{Y}-Y)$$

### 2.2

$$\frac{\partial Q}{\partial W^{(3)}}=(a^{(2)})^{\top}\cdot\frac{\partial Q}{\partial Z^{(3)}},\quad \frac{\partial Q}{\partial b^{(3)}}=\sum_{i=1}^{N}\frac{\partial Q}{\partial Z_i^{(3)}}$$

### 2.3

$$\delta^{(l)}=\left(\delta^{(l+1)}\cdot(W^{(l+1)})^{\top}\right)\odot\sigma_l'(z^{(l)})$$

### 2.4

$$\delta^{(2)}=\left(\delta^{(3)}\cdot(W^{(3)})^{\top}\right)\odot(1-(a^{(2)})^2)$$
$$\delta^{(1)}=\left(\delta^{(2)}\cdot(W^{(2)})^{\top}\right)\odot(a^{(1)}\odot(1-a^{(1)}))$$

---

## Q3. Функции активации

### 3.1

$$\sigma_{\text{sigm}}(z)=\frac{1}{1+e^{-z}},\quad \sigma_{\text{sigm}}'(z)=\sigma_{\text{sigm}}(z)(1-\sigma_{\text{sigm}}(z))$$

$$\sigma_{\text{tanh}}(z)=\frac{e^{z}-e^{-z}}{e^{z}+e^{-z}},\quad \sigma_{\text{tanh}}'(z)=1-\tanh^2(z)$$

### 3.2

Sigmoid: $(0,1)$, насыщение при $z\to\pm\infty$, $\max|\sigma'|=0.25$

Tanh: $(-1,1)$, насыщение при $z\to\pm\infty$, $\max|\sigma'|=1$

### 3.3

$$\|\delta^{(1)}\|\le0.25\cdot\|W^{(2)}\|\cdot\|W^{(3)}\|\cdot\|\delta^{(3)}\|$$
