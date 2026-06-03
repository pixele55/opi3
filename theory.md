# Теоретическая часть — ЛР 5

**Студент:** [Ваше Имя]  
**Student ID:** 22  
**Вариант Q3:** 1 (Sigmoid + Tanh)  
**Архитектура:** 784 → 256 → 128 → 10  

---

## Q1. Прямой проход

### 1.1 Прямой проход многослойного перцептрона

Формулы прямого прохода в матричной форме:
$$z^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}, \quad a^{(l)} = \sigma_l(z^{(l)})$$

Раскроем их последовательно для слоёв $l = 1, 2, 3$ при обработке батча размера $N$:

**Слой 1 (скрытый):**
$$z^{(1)} = a^{(0)} W^{(1)} + b^{(1)}$$
$$a^{(1)} = \sigma_1(z^{(1)})$$

*Размерности:*
- $a^{(0)} \in \mathbb{R}^{N \times 784}$
- $W^{(1)} \in \mathbb{R}^{784 \times 256}$
- $b^{(1)} \in \mathbb{R}^{256}$
- $z^{(1)}, a^{(1)} \in \mathbb{R}^{N \times 256}$

**Слой 2 (скрытый):**
$$z^{(2)} = a^{(1)} W^{(2)} + b^{(2)}$$
$$a^{(2)} = \sigma_2(z^{(2)})$$

*Размерности:*
- $W^{(2)} \in \mathbb{R}^{256 \times 128}$
- $b^{(2)} \in \mathbb{R}^{128}$
- $z^{(2)}, a^{(2)} \in \mathbb{R}^{N \times 128}$

**Слой 3 (выходной):**
$$z^{(3)} = a^{(2)} W^{(3)} + b^{(3)}$$
$$\hat{y}_{i,k} = \frac{\exp(z_{i,k}^{(3)})}{\sum_{j=1}^{10} \exp(z_{i,j}^{(3)})}$$

*Размерности:*
- $W^{(3)} \in \mathbb{R}^{128 \times 10}$
- $b^{(3)} \in \mathbb{R}^{10}$
- $z^{(3)} \in \mathbb{R}^{N \times 10}$
- $\hat{Y} \in \mathbb{R}^{N \times 10}$

### 1.2 Функция потерь cross-entropy

Функция потерь для одного примера:
$$\mathscr{L}(\hat{y}_i, y_i) = -\sum_{k=1}^{K} y_{i,k} \log \hat{y}_{i,k}$$

Функционал качества для всей выборки:
$$Q(\boldsymbol{\theta}, X^\ell) = \frac{1}{\ell}\sum_{i=1}^{\ell} \mathscr{L}(\hat{y}_i, y_i)$$

### 1.3 Численная стабильность softmax

Оригинальная формула softmax:
$$\hat{y}_k = \frac{\exp(z_k^{(3)})}{\sum_{j=1}^{K} \exp(z_j^{(3)})}$$

Численно стабильная формула:
$$\hat{y}_k = \frac{\exp(z_k^{(3)} - m)}{\sum_{j=1}^{K} \exp(z_j^{(3)} - m)}, \quad m = \max_j z_j^{(3)}$$

Доказательство эквивалентности:
$$\frac{\exp(z_k - m)}{\sum_j \exp(z_j - m)} = \frac{\exp(z_k) \cdot \exp(-m)}{\sum_j (\exp(z_j) \cdot \exp(-m))} = \frac{\exp(z_k)}{\sum_j \exp(z_j)}$$

---

## Q2. Обратный проход

### 2.1 Градиент по логитам выходного слоя

Для одного примера:
$$\frac{\partial \mathscr{L}}{\partial z_k^{(3)}} = \hat{y}_k - y_k$$

Для батча:
$$\frac{\partial Q}{\partial Z^{(3)}} = \frac{1}{N}(\hat{Y} - Y)$$

### 2.2 Градиенты весов и смещений выходного слоя

$$\frac{\partial Q}{\partial W^{(3)}} = (a^{(2)})^\top \cdot \frac{\partial Q}{\partial Z^{(3)}}$$

$$\frac{\partial Q}{\partial b^{(3)}} = \sum_{i=1}^{N} \frac{\partial Q}{\partial Z_i^{(3)}}$$

### 2.3 Общая формула обратного распространения

$$\delta^{(l)} = \left(\delta^{(l+1)} \cdot (W^{(l+1)})^\top\right) \odot \sigma'_l(z^{(l)})$$

### 2.4 Вывод $\delta$ для слоёв 1 и 2

**Для слоя 2:**
$$\delta^{(2)} = \left(\delta^{(3)} \cdot (W^{(3)})^\top\right) \odot \sigma'_2(z^{(2)})$$

**Для слоя 1:**
$$\delta^{(1)} = \left(\delta^{(2)} \cdot (W^{(2)})^\top\right) \odot \sigma'_1(z^{(1)})$$

---

## Q3. Функции активации Sigmoid и Tanh

### 3.1 Определения и производные

**Sigmoid:**
$$\sigma_{\text{sigm}}(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma'_{\text{sigm}}(z) = \sigma_{\text{sigm}}(z) \cdot (1 - \sigma_{\text{sigm}}(z))$$

**Tanh:**
$$\sigma_{\text{tanh}}(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
$$\sigma'_{\text{tanh}}(z) = 1 - \tanh^2(z)$$

### 3.2 Свойства функций активации

| Свойство | Sigmoid | Tanh |
|----------|---------|------|
| Область значений | $(0, 1)$ | $(-1, 1)$ |
| Зоны насыщения | $z \to \pm\infty$ | $z \to \pm\infty$ |
| $\max|\sigma'(z)|$ | $0.25$ | $1$ |

### 3.3 Затухание градиентов

$$\delta^{(1)} = \left(\left(\delta^{(3)} \cdot (W^{(3)})^\top\right) \odot \sigma'_{\text{Tanh}}(z^{(2)})\right) \cdot (W^{(2)})^\top \odot \sigma'_{\text{Sigmoid}}(z^{(1)})$$

Оценка нормы:
$$\|\delta^{(1)}\| \le 0.25 \cdot \|W^{(2)}\| \cdot \|W^{(3)}\| \cdot \|\delta^{(3)}\|$$
