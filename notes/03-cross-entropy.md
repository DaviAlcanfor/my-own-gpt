# 3. Cross-Entropy Loss (com Softmax)

* **Referência:** [How to Build a GPT from Scratch — Green Code Codes](http://www.youtube.com/watch?v=3LWHYlmyqZ0)
* **Timestamps:** [[17:10](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1030)] – [[24:14](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1454)]
* **Implementação:** [`layers/cross_entropy.py`](../layers/cross_entropy.py)
* **Pré-requisito:** [02-softmax.md](02-softmax.md)

---

## Definição

A entropia cruzada quantifica o erro entre a distribuição alvo *one-hot* ($\mathbf{y}$) e a probabilidade prevista pelo modelo ($\hat{\mathbf{y}} = \text{Softmax}(\mathbf{z})$) [[17:10](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1030), [17:34](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1054)]:

$$\mathcal{L} = -\sum_{i=1}^{V} y_i \ln(\hat{y}_i)$$

Como $y_k = 1$ apenas para o índice do token correto $k$ e $y_i = 0$ para todos os demais ($i \neq k$), a soma colapsa para [[20:56](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1256), [21:09](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1269)]:

$$\mathcal{L} = -\ln(\hat{y}_k)$$

## Formulação Estável Combinada (Log-Sum-Exp)

Aplicar $\ln$ diretamente sobre $\hat{y}_k$ é instável: se $\hat{y}_k$ arredondar para zero, obtém-se $\ln(0) = -\infty$ [[21:17](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1277), [22:05](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1325)]. Substituindo a definição da Softmax deslocada diretamente no logaritmo [[22:23](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=1343)]:

$$\mathcal{L} = -\ln\left(\frac{e^{z_k - m}}{\sum_{j=1}^{V} e^{z_j - m}}\right) = -\left(z_k - m - \ln\sum_{j=1}^{V} e^{z_j - m}\right)$$

Equivalentemente, definindo $\text{LSE}(\mathbf{z}) = m + \ln\sum_{j} e^{z_j - m}$:

$$\mathcal{L} = \text{LSE}(\mathbf{z}) - z_k$$

Para um batch com $N = B \cdot T$ posições, a loss final é a média:

$$\mathcal{L}_{\text{batch}} = \frac{1}{N}\sum_{n=1}^{N} \left(\text{LSE}(\mathbf{z}^{(n)}) - z^{(n)}_{k_n}\right)$$

---

## Implementação

```python
class CrossEntropyWithSoftmax(GPTLayer):
    def forward(self, logits, labels):
        assert len(logits.shape) == 3
        B, T, V = logits.shape

        logits_flat = np.reshape(logits, (B * T, V))
        labels_flat = np.reshape(labels, (B * T))

        max_float = np.max(logits_flat, axis=-1, keepdims=True)
        diff = logits_flat - max_float

        lse = max_float + np.log(np.sum(np.exp(diff), axis=-1, keepdims=True))
        log_softmax = np.expand_dims(logits_flat[np.arange(B * T), labels_flat], axis=-1) - lse

        ...
        return -np.sum(log_softmax) / logits_flat.shape[0]
```

Os logits `(B, T, V)` e os rótulos `(B, T)` são achatados para `(B·T, V)` e `(B·T,)`, de modo que cada posição da sequência é tratada como uma amostra de classificação independente sobre o vocabulário.
