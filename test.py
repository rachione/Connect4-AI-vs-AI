import re

with open('test.txt', 'r') as f:
    data = f.read()

match = re.findall('([OX.\r\n]*).*$', data)
print(match)