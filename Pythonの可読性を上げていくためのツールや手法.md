# Pythonの可読性を上げていくためのツールや手法
https://zenn.dev/shimakaze_soft/scraps/4b02e4662e6d1f

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

##.flake8ファイルを作成
```
[flake8]
# E501: line too long (82 > 79 characters)
# W503: line break before binary operator
# F401: module imported but unused
# D400 First line should end with a period
# D200 One-line docstring should fit on one line with quotes
ignore = E501,W503,F401,D400,D200

# Source directories
exclude =
    .git,
    __pycache__,
    build,
    dist

# McCabe complexity threshold
max-complexity = 10

# cognitive_complexity
enable-extensions = cognitive_complexity,G
cognitive-complexity-threshold = 1

#max-line-length
max-line-length = 120

# flake8-eradicate
eradicate-ignore-annotations = True
eradicate-ignore-warnings = True

# flake8-isort
#known_third_party = your_third_party_library

# flake8-quotes
inline-quotes = double
multiline-quotes = double

## flake8-pytest-style
pytest-fixture-no-parentheses = true

## flake8-annotations
annotations-complexity = 2
annotations-multiline-no-trailing-comma = true

## flake8-mypy
#plugins = mypy
#mypy-config = mypy.ini
#
#
```

## 循環的複雑度 (Cyclomatic complexity) -> CC
- lizerd
- pip install lizard
- lizard ./complexy.py

---

## 認知的複雑度 (Cognitive Complexity)
- flake8-cognitive-complexity
- pip install flake8-cognitive-complexity

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












