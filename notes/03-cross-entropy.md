# Cross-entropy (com softmax)

Essa é a nota que o modelo recebe por cada palpite. Ele chuta uma distribuição de probabilidade pra próxima palavra, a gente olha quanto de probabilidade ele deu pra palavra certa, e pune de acordo. Deu muita probabilidade pra certa, loss perto de zero. Deu pouca, loss alta.

## Exemplo

Contexto `"I love"`, a próxima palavra certa é `"green"`. O modelo cospe logits, a softmax vira probabilidade:

$$
\begin{array}{l|c|c|c}
 & \text{logit } z & \text{prob } \hat{y} & \text{alvo } y \\ \hline
\text{I}      & 0.5 & 0.059 & 0 \\
\text{love}   & 1.0 & 0.097 & 0 \\
\text{green}  & 3.0 & \mathbf{0.720} & \mathbf{1} \\
\text{code}   & 0.2 & 0.044 & 0 \\
\text{videos} & 0.8 & 0.080 & 0
\end{array}
$$

O alvo $y$ é one-hot: 1 na palavra certa, 0 no resto.

## Fórmula

A definição geral soma sobre o vocabulário inteiro:

$$
\mathcal{L} = -\sum_{i=1}^{V} y_i \ln(\hat{y}_i)
$$

Só que $y$ é zero em todo lugar menos na palavra certa $k$. Então quase todos os termos somem:

$$
\mathcal{L}
= -\big(\,\underbrace{0 \cdot \ln 0.059}_{0} + \underbrace{0 \cdot \ln 0.097}_{0} + \underbrace{1 \cdot \ln 0.720}_{\text{só esse sobra}} + \underbrace{0 \cdot \ln 0.044}_{0} + \underbrace{0 \cdot \ln 0.080}_{0}\,\big)
$$

$$
\mathcal{L} = -\ln(\hat{y}_k) = -\ln(0.720) \approx 0.329
$$

No fim, a loss é só "menos o log da probabilidade que você deu pra resposta certa".

## Como a loss reage

$$
\begin{array}{c|c|l}
\hat{y}_k & -\ln(\hat{y}_k) & \\ \hline
0.99 & 0.010 & \text{quase perfeito} \\
0.72 & 0.329 & \text{nosso exemplo} \\
0.50 & 0.693 & \\
0.10 & 2.303 & \\
0.01 & 4.605 & \text{errou feio}
\end{array}
$$

```text
  loss
   5 │█
     │█
   4 │█
     │ █
   3 │  █
     │   █
   2 │    █
     │      █
   1 │         ██
     │              ████
   0 │                   ██████████████
     └────────────────────────────────── prob da palavra certa
     0                                 1
```

Errar com confiança custa muito mais do que errar em dúvida. Quando $\hat{y}_k \to 0$, a loss vai pro infinito.

## O problema do ln(0)

Se eu calcular a softmax primeiro e depois o log, pode acontecer de $\hat{y}_k$ arredondar pra 0 no float, e $\ln(0) = -\infty$. Pra evitar, junta as duas coisas numa conta só e nunca calcula a probabilidade de fato.

Abrindo o log da softmax (já com o truque de subtrair o máximo $m$):

$$
-\ln\left(\frac{e^{z_k - m}}{\sum_j e^{z_j - m}}\right)
= -\Big(\, (z_k - m) - \ln \sum_j e^{z_j - m} \,\Big)
$$

Juntando os termos com $m$, aparece o log-sum-exp (LSE):

$$
\mathcal{L} = \underbrace{m + \ln \sum_j e^{z_j - m}}_{\text{LSE}(\mathbf{z})} \;-\; z_k
$$

Não tem divisão e não tem log de número pequeno. O log é de uma soma que vale pelo menos 1 (porque um dos termos é $e^0$).

Conferindo com o exemplo ($m = 3.0$):

$$
\mathbf{z} - m = \begin{bmatrix} -2.5 \\ -2.0 \\ 0.0 \\ -2.8 \\ -2.2 \end{bmatrix}
\;\xrightarrow{\;e^{z}\;}\;
\begin{bmatrix} 0.082 \\ 0.135 \\ 1.000 \\ 0.061 \\ 0.111 \end{bmatrix}
\;\xrightarrow{\;\Sigma\;}\; 1.389
$$

$$
\mathcal{L} = 3.0 + \ln(1.389) - 3.0 = 0.329
$$

Mesmo resultado de antes.

## Com batch

O modelo prevê uma palavra pra cada posição de cada frase. A implementação achata tudo e trata cada posição como uma pergunta independente:

```text
  logits (B, T, V)                     logits (B·T, V)        alvo (B·T)

  frase 1 ┌ pos 1 │ ... V ... │        │ ... V ... │            k₁
          └ pos 2 │ ... V ... │  ───▶  │ ... V ... │            k₂
  frase 2 ┌ pos 1 │ ... V ... │        │ ... V ... │            k₃
          └ pos 2 │ ... V ... │        │ ... V ... │            k₄
```

E a loss final é a média de todas:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{B \cdot T} \sum_{n=1}^{B \cdot T} \Big(\text{LSE}(\mathbf{z}^{(n)}) - z^{(n)}_{k_n}\Big)
$$
