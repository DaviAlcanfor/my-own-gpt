# 1. Embedding Layer

* **Referência:** [How to Build a GPT from Scratch — Green Code Codes](http://www.youtube.com/watch?v=3LWHYlmyqZ0)
* **Timestamps:** [[04:04](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=244)] – [[09:25](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=565)]
* **Implementação:** [`layers/embedding.py`](../layers/embedding.py)

---

## Tokenização e Embedding Lookup

Como redes neurais operam exclusivamente sobre tensores numéricos, o texto bruto passa por dois estágios iniciais [[04:04](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=244)]:

1. **Tokenização:** mapeia palavras ou subpalavras para IDs inteiros $\mathbf{t} = [t_1, t_2, \dots, t_T]$ [[04:16](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=256)].
2. **Embedding Lookup:** converte cada ID discreto em um vetor denso contínuo de dimensão $d_{\text{model}}$ (ex.: $d_{\text{model}} = 768$ no GPT-2; $d_{\text{model}} = 4$ nos exemplos didáticos da aula) [[04:48](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=288), [07:54](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=474)].

Formalmente, a camada é uma matriz de pesos $\mathbf{E} \in \mathbb{R}^{V \times d_{\text{model}}}$, e o forward é uma indexação por linha:

$$\mathbf{x}_i = \mathbf{E}[t_i]$$

## Ilustração da Matriz de Embeddings ($\mathbf{X} \in \mathbb{R}^{5 \times 4}$)

Para a frase `"I love green code videos"` ($T = 5$ tokens, $d = 4$):

$$\mathbf{X} = \begin{bmatrix} x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\ x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\ x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\ x_{4,1} & x_{4,2} & x_{4,3} & x_{4,4} \\ x_{5,1} & x_{5,2} & x_{5,3} & x_{5,4} \end{bmatrix} \begin{array}{l} \leftarrow \text{"I"} \\ \leftarrow \text{"love"} \\ \leftarrow \text{"green"} \\ \leftarrow \text{"code"} \\ \leftarrow \text{"videos"} \end{array}$$

## Propriedades Geométricas no Espaço de Embeddings

Durante o treinamento, vetores semanticamente próximos agrupam-se no espaço vetorial e preservam relações lineares [[05:47](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=347), [07:09](https://www.youtube.com/watch?v=3LWHYlmyqZ0&t=429)]:

$$\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}$$

---

## Implementação

```python
class EmbeddingLayer(GPTLayer):
    def __init__(self, vocab_size, embedding_dim):
        self.embeddings = np.random.randn(vocab_size, embedding_dim) * 0.02
        self.inputs = None

    def forward(self, inputs):
        self.inputs = inputs
        return self.embeddings[inputs]
```

Para uma entrada de IDs com shape `(B, T)`, a saída tem shape `(B, T, embedding_dim)`.
