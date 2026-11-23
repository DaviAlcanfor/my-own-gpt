# Embedding

A rede não entende texto, só número. Então o primeiro passo é trocar cada palavra (ou pedaço de palavra) por um ID, e depois trocar cada ID por uma lista de números que a rede consegue ajustar durante o treino. Essa lista é o embedding.

O embedding é só uma tabela. Cada linha é uma palavra do vocabulário, cada coluna é uma "característica" que o modelo aprende sozinho. Ninguém diz o que a coluna 3 significa, ela vai ganhando sentido com o treino.

## A tabela

Um vocabulário de brinquedo com 6 palavras ($V = 6$) e vetores de tamanho 4 ($d = 4$):

$$
\mathbf{E} =
\begin{array}{r|l}
\text{0: I}      & \begin{bmatrix} \phantom{-}0.12 & -0.40 & \phantom{-}0.33 & \phantom{-}0.05 \end{bmatrix} \\
\text{1: love}   & \begin{bmatrix} -0.21 & \phantom{-}0.18 & \phantom{-}0.07 & \phantom{-}0.44 \end{bmatrix} \\
\text{2: green}  & \begin{bmatrix} \phantom{-}0.30 & \phantom{-}0.02 & -0.15 & \phantom{-}0.27 \end{bmatrix} \\
\text{3: code}   & \begin{bmatrix} \phantom{-}0.09 & \phantom{-}0.35 & \phantom{-}0.41 & -0.12 \end{bmatrix} \\
\text{4: videos} & \begin{bmatrix} -0.05 & \phantom{-}0.22 & \phantom{-}0.19 & \phantom{-}0.31 \end{bmatrix} \\
\text{5: cat}    & \begin{bmatrix} \phantom{-}0.48 & -0.11 & \phantom{-}0.02 & -0.36 \end{bmatrix}
\end{array}
\qquad
\mathbf{E} \in \mathbb{R}^{V \times d} = \mathbb{R}^{6 \times 4}
$$

No começo esses números são aleatórios e pequenos. O treino que vai mexendo neles.

## O forward é só pegar a linha

Pra frase `"love code"` os IDs são `[1, 3]`. A camada pega a linha 1 e a linha 3:

$$
\mathbf{x}_i = \mathbf{E}[t_i]
$$

$$
\underbrace{[\,1,\ 3\,]}_{\text{IDs}}
\;\xrightarrow{\text{lookup}}\;
\begin{bmatrix}
-0.21 & \phantom{-}0.18 & \phantom{-}0.07 & \phantom{-}0.44 \\
\phantom{-}0.09 & \phantom{-}0.35 & \phantom{-}0.41 & -0.12
\end{bmatrix}
\begin{array}{l} \leftarrow \text{love} \\ \leftarrow \text{code} \end{array}
$$

Não tem multiplicação nenhuma acontecendo, é indexação.

## Por que isso é a mesma coisa que multiplicar por one-hot

Às vezes o embedding aparece escrito como uma multiplicação de matriz. É equivalente. Se eu escrever "love" como um vetor one-hot (tudo zero, 1 na posição dele) e multiplicar pela tabela, os zeros apagam todas as linhas menos a dele:

$$
\underbrace{\begin{bmatrix} 0 & 1 & 0 & 0 & 0 & 0 \end{bmatrix}}_{\text{one-hot de "love"}}
\cdot
\begin{bmatrix}
\cdots & \text{I} & \cdots \\
\cdots & \text{love} & \cdots \\
\cdots & \text{green} & \cdots \\
\cdots & \text{code} & \cdots \\
\cdots & \text{videos} & \cdots \\
\cdots & \text{cat} & \cdots
\end{bmatrix}
=
\begin{bmatrix} \cdots & \text{love} & \cdots \end{bmatrix}
$$

Na prática ninguém faz isso porque é desperdício: multiplicar por um monte de zero pra no fim pegar uma linha.

## Frase inteira

Para `"I love green code videos"` ($T = 5$ tokens) sai uma matriz $5 \times 4$, uma linha por token:

$$
\mathbf{X} =
\begin{bmatrix}
x_{1,1} & x_{1,2} & x_{1,3} & x_{1,4} \\
x_{2,1} & x_{2,2} & x_{2,3} & x_{2,4} \\
x_{3,1} & x_{3,2} & x_{3,3} & x_{3,4} \\
x_{4,1} & x_{4,2} & x_{4,3} & x_{4,4} \\
x_{5,1} & x_{5,2} & x_{5,3} & x_{5,4}
\end{bmatrix}
\begin{array}{l}
\leftarrow \text{I} \\
\leftarrow \text{love} \\
\leftarrow \text{green} \\
\leftarrow \text{code} \\
\leftarrow \text{videos}
\end{array}
$$

Com batch, ganha mais uma dimensão na frente:

```text
   IDs (B, T)                    Embeddings (B, T, d)

  ┌───┬───┬───┐                  ┌───────────────┐
  │ 0 │ 1 │ 2 │   ── lookup ──▶  │ ■ ■ ■ ■       │ ← frase 1, cada token virou 4 números
  ├───┼───┼───┤                  │ ■ ■ ■ ■       │
  │ 3 │ 4 │ 5 │                  │ ■ ■ ■ ■       │
  └───┴───┴───┘                  ├───────────────┤
   B=2, T=3                      │ ■ ■ ■ ■       │ ← frase 2
                                 │ ■ ■ ■ ■       │
                                 │ ■ ■ ■ ■       │
                                 └───────────────┘
```

## O que acontece depois de treinar

Palavras parecidas acabam com vetores parecidos, e algumas relações viram direções no espaço. O exemplo clássico:

$$
\vec{v}_{\text{king}} - \vec{v}_{\text{man}} + \vec{v}_{\text{woman}} \approx \vec{v}_{\text{queen}}
$$

Ou seja, "tirar homem e colocar mulher" é mais ou menos o mesmo deslocamento não importa de qual palavra você parte. Isso não é programado, aparece sozinho.
