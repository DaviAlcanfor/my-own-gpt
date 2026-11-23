# 2. Softmax

* **Referência:** [How to Build a GPT from Scratch — Green Code Codes](http://www.youtube.com/watch?v=3LWHYlmyqZ0)
* **Timestamps:** [[09:30](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=570)] – [[15:47](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=947)]
* **Implementação:** [`layers/softmax.py`](../layers/softmax.py)

---

## Definição

A camada **Softmax** converte o vetor de pontuações brutas da última camada linear (**logits**, $\mathbf{z} \in \mathbb{R}^V$) em uma distribuição de probabilidade válida, onde todos os valores estão em $[0, 1]$ e somam $1$ [[09:49](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=589), [10:16](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=616)]:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}, \quad \text{com } \sum_{i=1}^{V} \text{Softmax}(z_i) = 1$$

## Instabilidade Numérica e Invariância por Deslocamento (*Shift Invariance*)

Quando um logit é muito grande (ex.: $z_i = 275$), calcular $e^{275}$ causa *overflow* (`inf` / `NaN`) em ponto flutuante [[14:15](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=855), [14:29](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=869)]. Como a Softmax é invariante à subtração de uma constante $c$, subtrai-se $m = \max(\mathbf{z})$ antes de exponenciar [[15:09](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=909), [15:26](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=926)]:

$$\frac{e^{z_i - m}}{\sum_{j=1}^{V} e^{z_j - m}} = \frac{e^{-m} e^{z_i}}{e^{-m} \sum_{j=1}^{V} e^{z_j}} = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

Isso garante que o maior expoente seja $e^{0} = 1$, eliminando o *overflow* [[15:47](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=947)].

---

## Implementação

```python
class Softmax(GPTLayer):
    def __init__(self):
        self.softmax_out = None

    def forward(self, inputs):
        max_logit = np.max(inputs, axis=-1, keepdims=True)
        softmax_out = np.exp(inputs - max_logit) / np.sum(np.exp(inputs - max_logit), axis=-1, keepdims=True)
        self.softmax_out = softmax_out
        return softmax_out
```

A normalização é aplicada no último eixo (`axis=-1`), ou seja, sobre o vocabulário $V$.
