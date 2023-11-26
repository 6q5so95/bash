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
```
[flake8]
enable-extensions = cognitive_complexity
cognitive-complexity-threshold = 1
```
- flake8 --enable-extensions cognitive_complexity ./complexy.py

---

## 保守容易性指数(Maintainability Index)
- radonによる測定
- pip install radon
- radon mi -s ./complexy.py

## 凝集度 (Cohesion)
- pip install cohesion
- cohesion --files ./coh.py

---

## flake8プラグイン
```
flake8-bugbear
flake8-builtins
flake8-eradicate
pep8-naming
flake8-pytest-style
flake8-isort
flake8-quotes
flake8-print
flake8-annotations
flake8-mypy
flake8-docstrings
pycodestyle
pyflakes
```

- pip install flake8-bugbear flake8-builtins flake8-eradicate pep8-naming flake8-pytest-style flake8-isort flake8-quotes flake8-print flake8-annotations flake8-mypy flake8-docstrings pycodestyle pyflakes










