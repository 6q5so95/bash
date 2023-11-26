# Pythonの可読性を上げていくためのツールや手法
https://zenn.dev/shimakaze_soft/scraps/4b02e4662e6d1f

## 循環的複雑度 (Cyclomatic complexity) -> CC
- lizerd
- pip install lizard
- lizard ./complexy.py

---

## 認知的複雑度 (Cognitive Complexity)
- flake8-cognitive-complexity
- pip install flake8-cognitive-complexity
- .flake8ファイルを作成
- flake8 --enable-extensions cognitive_complexity ./complexy.py
```
[flake8]
enable-extensions = cognitive_complexity
cognitive-complexity-threshold = 1
```

---



