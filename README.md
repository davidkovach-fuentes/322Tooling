# Testing

**Entire Suite**: .venv/bin/python -m pytest LB/test.py -q

**Individual Tests**: .venv/bin/python -m pytest LB/test py -k testXX

### Example

**.venv/bin/python -m pytest LB/test.py -k test1**

This will run all test files that match the substring `````test1`````

So: test1, test10, test11, test12, ..., test100, ..., test115

