# CS224N
## 스탠포드 2021

### 1강 word vectors

- GPT-3 에 대한 소개
- WordNet : contain lists of synonym sets and hypernyms
  - only correct in some context (일부 문장에서만 맞고 주로 이상하거나 다름)
  - missing new meanings of words (현대적 용어 이해 못함)
  - 단어 유사성을 가지고 진행

- localist representation => one-hot vector (각각의 단어가 다른 vector에 위치)
  - 그렇기에 seattle motel 과 seattle hotel 을 유사하게 연관 짓고 싶지만 orthogonal 하기에 no similariy
  - solution : learn to encode similarity in vectors themselves

- Distributional semantics (분포 의미론 활용) : frequently appear close (주변에 많이 나온 것 이용하기)
  - type, token 두 종류의 의미

#### Word2vec

word vectors are also called `word embeddings` or `word representations`

idea
  - large corpus of text (말뭉치, 텍스트 몸)
  - fixed vocabulary by vector
  - use the similarity of word vectors for c and o to calculate the probability
  - keep adjusting the word vectors to maximize tis porbabilty
    - 유사성을 이용해 확률을 계산해서 최대화를 유지하게끔

  - center word (중심단어)가 문맥에서 나타날 확률

![image](https://user-images.githubusercontent.com/108413432/209459824-18af8ece-fd67-4df1-a9fb-d43c55baab8c.png)
  - each word as the center word => product of each word => window around of probability of prdicting
  - 로그를 취하면 sum으로 바뀌니까 minimize objective function을 이용해서 정확도 계산 가능
  - Vw (w는 center word) , Uw (w는 context word) 통해 확률 계산

![image](https://user-images.githubusercontent.com/108413432/209459934-0008fd6d-ad60-497f-b088-a4949b0284cd.png)
  - 단어 내적 크면 more similar
  - exp => nomalize (확률 계산 위해)
  - softmax ftn 이라고 불림 (0~1 값 변환 됨)
  - minimize negative log

- Gradient 구하기 (화이트보드로 설명)
  - softmax 식에서 log 취하고 Vc(센터 워드)로 미분해줌
    - 분자의 경우 exp과 없어지기에 단순해짐
    - 분모는 chain rule 이용하기 (속미분 겉미분 느낌)

- Gensim 이용해 Python 사용
  - most_similar() 이용해서 유사 값들을 보여줄 수 있음
  - most_similart(positive 얻고자 하는 값, negative 제외하고자 하는 값) => 새로운 결과 (유사성 이용해서)
  - 
