# Теоретическая часть — ЛР 5
**Студент:** [Ваше Имя]  
**Student ID:** 22  
**Вариант Q3:** 1 (Sigmoid + Tanh)  
**Архитектура:** 784 → 256 → 128 → 10  

---

## Q1. Прямой проход

### 1.1 Формулы прямого прохода для батча

Пусть у нас есть батч из $N$ примеров. Обозначим:

- $X = a^{(0)} \in \mathbb{R}^{N \times n_0}$ — входная матрица батча ($n_0 = 784$)
- $W^{(1)} \in \mathbb{R}^{n_0 \times n_1}$ — веса первого слоя ($n_1 = 256$)
- $b^{(1)} \in \mathbb{R}^{n_1}$ — смещения первого слоя
- $W^{(2)} \in \mathbb{R}^{n_1 \times n_2}$ — веса второго слоя ($n_2 = 128$)
- $b^{(2)} \in \mathbb{R}^{n_2}$
- $W^{(3)} \in \mathbb{R}^{n_2 \times K}$ — веса выходного слоя ($K = 10$)
- $b^{(3)} \in \mathbb{R}^{K}$

**Прямой проход:**

1. **Скрытый слой 1 (Sigmoid):**
   $$Z^{(1)} = a^{(0)} W^{(1)} + b^{(1)} \quad \text{(размерность: } N \times n_1\text{)}$$
   $$a^{(1)} = \sigma_{\text{sigmoid}}(Z^{(1)}) = \frac{1}{1 + \exp(-Z^{(1)})}$$

2. **Скрытый слой 2 (Tanh):**
   $$Z^{(2)} = a^{(1)} W^{(2)} + b^{(2)} \quad \text{(размерность: } N \times n_2\text{)}$$
   $$a^{(2)} = \sigma_{\text{tanh}}(Z^{(2)}) = \frac{e^{Z^{(2)}} - e^{-Z^{(2)}}}{e^{Z^{(2)}} + e^{-Z^{(2)}}}$$

3. **Выходной слой (Softmax):**
   $$Z^{(3)} = a^{(2)} W^{(3)} + b^{(3)} \quad \text{(размерность: } N \times K\text{)}$$
   $$\hat{Y} = \text{softmax}(Z^{(3)}), \quad \hat{y}_{ik} = \frac{\exp(z^{(3)}_{ik})}{\sum_{j=1}^{K} \exp(z^{(3)}_{ij})}$$

### 1.2 Функция потерь Cross-Entropy

Для одного примера $(\mathbf{x}_i, y_i)$ с one-hot кодированием:

$$\mathscr{L}(\hat{y}_i, y_i) = -\sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$$

**Роль one-hot кодирования:** Сумма содержит $K-1$ нулевых слагаемых (где $y_{ik}=0$) и одно ненулевое (где $y_{ik}=1$). Лосс упрощается до $-\log \hat{y}_{i, \text{true}}$.

**Стремление к нулю:** При правильной классификации $\hat{y}_{i, \text{true}} \to 1$, тогда $\log(1) = 0$ и $\mathscr{L} \to 0$.

Функционал качества:
$$Q(\boldsymbol{\theta}, X^\ell) = \frac{1}{\ell}\sum_{i=1}^{\ell} \mathscr{L}(\hat{y}_i, y_i)$$

### 1.3 Численная стабильность Softmax

**Проблема:** При больших $z_k$ значение $\exp(z_k)$ вызывает переполнение.

**Стабильная формула** (вычитаем максимум $M = \max_j z_j$):
$$\hat{y}_k = \frac{\exp(z_k - M)}{\sum_{j=1}^{K} \exp(z_j - M)}$$

**Доказательство эквивалентности:**
$$\hat{y}_k = \frac{\exp(z_k)/\exp(M)}{\sum_{j}\exp(z_j)/\exp(M)} = \frac{\exp(z_k)/\exp(M)}{\left(\sum_{j}\exp(z_j)\right)/\exp(M)} = \frac{\exp(z_k)}{\sum_{j}\exp(z_j)}$$

---

## Q2. Обратный проход

### 2.1 Градиент Softmax + Cross-Entropy

Средняя ошибка по батчу: $Q = \frac{1}{N} \sum_{i=1}^{N} \mathscr{L}_i$, где $\mathscr{L}_i = -\sum_{k=1}^{K} y_{ik} \log \hat{y}_{ik}$.

Производная softmax: $\frac{\partial \hat{y}_{im}}{\partial z^{(3)}_{ij}} = \hat{y}_{im}(\delta_{mj} - \hat{y}_{ij})$

Вывод для одного примера:
$$\frac{\partial \mathscr{L}_i}{\partial z^{(3)}_{ij}} = -\sum_{m=1}^{K} \frac{y_{im}}{\hat{y}_{im}} \cdot \frac{\partial \hat{y}_{im}}{\partial z^{(3)}_{ij}} = -\sum_{m} y_{im}(\delta_{mj} - \hat{y}_{ij}) = \hat{y}_{ij} - y_{ij}$$

Усредняя по батчу:
$$\frac{\partial Q}{\partial z^{(3)}_{ij}} = \frac{1}{N}(\hat{y}_{ij} - y_{ij})$$

В матричной форме:
$$\boxed{\frac{\partial Q}{\partial Z^{(3)}} = \frac{1}{N}(\hat{Y} - Y)}$$

### 2.2 Градиенты выходного слоя

Для весов:
$$\frac{\partial Q}{\partial W^{(3)}_{pq}} = \sum_{i=1}^{N} \frac{\partial Q}{\partial Z^{(3)}_{iq}} \cdot a^{(2)}_{ip}$$
$$\boxed{\frac{\partial Q}{\partial W^{(3)}} = (a^{(2)})^{\top} \cdot \frac{\partial Q}{\partial Z^{(3)}}}$$

Для смещений:
$$\boxed{\frac{\partial Q}{\partial b^{(3)}} = \sum_{i=1}^{N} \left(\frac{\partial Q}{\partial Z^{(3)}}\right)_{i}}$$

### 2.3 Общая формула обратного распространения

Определим $\delta^{(l)} = \frac{\partial Q}{\partial Z^{(l)}}$. Тогда:
$$\delta^{(l)} = \left( \delta^{(l+1)} \cdot (W^{(l+1)})^{\top} \right) \odot \sigma'_l(Z^{(l)})$$

**Содержательный смысл:**
- $(W^{(l+1)})^{\top}$ транспонирует веса, чтобы "передать" ошибку обратно через слой
- $\sigma'_l(Z^{(l)})$ показывает, как изменение преактивации влияет на пост-активацию

### 2.4 Вывод дельт и градиентов

**Слой 2 (Tanh):**
$$\delta^{(2)} = \left( \delta^{(3)} \cdot (W^{(3)})^{\top} \right) \odot \sigma'_{\text{tanh}}(Z^{(2)})$$
$$\sigma'_{\text{tanh}}(z) = 1 - \tanh^2(z) = 1 - (a^{(2)})^2$$

**Слой 1 (Sigmoid):**
$$\delta^{(1)} = \left( \delta^{(2)} \cdot (W^{(2)})^{\top} \right) \odot \sigma'_{\text{sigmoid}}(Z^{(1)})$$
$$\sigma'_{\text{sigmoid}}(z) = \sigma_{\text{sigmoid}}(z)(1 - \sigma_{\text{sigmoid}}(z)) = a^{(1)} \odot (1 - a^{(1)})$$

**Градиенты:**
$$\frac{\partial Q}{\partial W^{(2)}} = (a^{(1)})^{\top} \cdot \delta^{(2)}, \quad \frac{\partial Q}{\partial b^{(2)}} = \sum_{i=1}^{N} \delta^{(2)}_i$$
$$\frac{\partial Q}{\partial W^{(1)}} = (a^{(0)})^{\top} \cdot \delta^{(1)}, \quad \frac{\partial Q}{\partial b^{(1)}} = \sum_{i=1}^{N} \delta^{(1)}_i$$

---

## Q3. Функции активации и затухание градиентов

### 3.1 Определения и производные

**Sigmoid:**
$$\sigma_{\text{sigm}}(z) = \frac{1}{1 + e^{-z}}$$
$$\sigma'_{\text{sigm}}(z) = \sigma_{\text{sigm}}(z) \cdot (1 - \sigma_{\text{sigm}}(z))$$

**Tanh:**
$$\sigma_{\text{tanh}}(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$
$$\sigma'_{\text{tanh}}(z) = 1 - \tanh^2(z)$$

*Доказательство для Tanh:*
$$\sigma'_{\text{tanh}}(z) = \frac{4}{(e^{z} + e^{-z})^2} = 1 - \left(\frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}\right)^2 = 1 - \tanh^2(z)$$

### 3.2 Свойства функций активации

| Свойство | Sigmoid | Tanh |
|----------|---------|------|
| Область значений | $(0, 1)$ | $(-1, 1)$ |
| Зоны насыщения | $z \to \pm\infty$ | $z \to \pm\infty$ |
| Производная ~ 0 | $|z| \approx 3$ и более | $|z| \approx 2$ и более |
| $\max|\sigma'(z)|$ | $0.25$ при $z=0$ | $1$ при $z=0$ |

**Почему Tanh предпочтительнее:**
1. **Нулевое среднее** — выход центрирован около 0, что ускоряет сходимость
2. **Больший максимальный градиент** (1 против 0.25) — меньше затухание

### 3.3 Затухание градиентов

Выражение для $\delta^{(1)}$ через $\delta^{(3)}$:
$$\delta^{(1)} = \left( \left( \delta^{(3)} \cdot (W^{(3)})^{\top} \right) \odot \sigma'_{\text{Tanh}}(z^{(2)}) \right) \cdot (W^{(2)})^{\top} \odot \sigma'_{\text{Sigmoid}}(z^{(1)})$$

Оценка нормы градиента:
$$\|\delta^{(1)}\| \le \|\delta^{(3)}\| \cdot \|W^{(3)}\| \cdot 1 \cdot \|W^{(2)}\| \cdot 0.25$$

**Вывод:** Множитель $0.25$ от Sigmoid приводит к экспоненциальному затуханию градиента с глубиной сети. Для $L$ слоёв Sigmoid градиент уменьшается как $0.25^L$.

**Практический вывод:** Sigmoid непригодна для глубоких сетей. Tanh работает лучше, но также может насыщаться. Современные решения (ReLU, LeakyReLU) избегают этой проблемы.
