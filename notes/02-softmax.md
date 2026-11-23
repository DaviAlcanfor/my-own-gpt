# Softmax

No fim da rede sai um número pra cada palavra do vocabulário, os logits. Eles podem ser qualquer coisa: negativo, enorme, tanto faz. A softmax pega esses números e transforma em probabilidades, ou seja, tudo fica entre 0 e 1 e a soma dá 1.

A ideia é simples: passa cada número pela exponencial (que deixa tudo positivo e faz o maior se destacar mais ainda) e divide pela soma.

## Fórmula

$$
\text{Softmax}(z_i) = \frac{e^{z_i}}{\displaystyle\sum_{j=1}^{V} e^{z_j}}
$$

## Exemplo com 3 palavras

$$
\mathbf{z} = \begin{bmatrix} 2.0 \\ 1.0 \\ 0.1 \end{bmatrix}
\;\xrightarrow{\;e^{z}\;}\;
\begin{bmatrix} 7.39 \\ 2.72 \\ 1.11 \end{bmatrix}
\;\xrightarrow{\;\div\, 11.22\;}\;
\begin{bmatrix} 0.659 \\ 0.242 \\ 0.099 \end{bmatrix}
\qquad
0.659 + 0.242 + 0.099 = 1
$$

Visualmente:

```text
  logits                     probabilidades

  2.0  ████████████████      0.659  ██████████████████████████
  1.0  ████████              0.242  ██████████
  0.1  █                     0.099  ████
```

Repara que a diferença de 2.0 pra 1.0 era "só 1", mas na probabilidade a primeira ficou quase 3x maior. A exponencial exagera as diferenças. É por isso que o modelo consegue ficar bem confiante numa palavra só.

## O problema do número grande

Se um logit for 1000, $e^{1000}$ estoura o float e vira `inf`. Aí `inf / inf` dá `NaN` e acabou.

O truque: subtrair o maior valor de todos antes de exponenciar. Isso não muda o resultado, porque o mesmo fator aparece em cima e embaixo e cancela:

$$
\frac{e^{z_i - m}}{\sum_j e^{z_j - m}}
= \frac{e^{-m}\, e^{z_i}}{e^{-m} \sum_j e^{z_j}}
= \frac{e^{z_i}}{\sum_j e^{z_j}}
\qquad m = \max(\mathbf{z})
$$

Com números:

$$
\begin{bmatrix} 1000 \\ 999 \\ 998 \end{bmatrix}
\;\xrightarrow{\;-\,1000\;}\;
\begin{bmatrix} 0 \\ -1 \\ -2 \end{bmatrix}
\;\xrightarrow{\;e^{z}\;}\;
\begin{bmatrix} 1.000 \\ 0.368 \\ 0.135 \end{bmatrix}
\;\xrightarrow{\;\div\, 1.503\;}\;
\begin{bmatrix} 0.665 \\ 0.245 \\ 0.090 \end{bmatrix}
$$

Depois da subtração o maior expoente sempre é $e^0 = 1$, então nunca estoura. E os outros são no máximo 1, então também não explodem.

## Com batch

A softmax roda separada em cada linha, sempre no último eixo (o do vocabulário). Cada token tem a sua própria distribuição:

```text
  logits (T, V)                       probs (T, V)

  token 1 │ 2.0  1.0  0.1 │           │ 0.66 0.24 0.10 │  soma 1
  token 2 │ 0.5  3.0  0.2 │  ──────▶  │ 0.07 0.88 0.05 │  soma 1
  token 3 │ 1.0  1.0  1.0 │           │ 0.33 0.33 0.33 │  soma 1
                    ▲
             axis = -1 (normaliza nessa direção)
```
