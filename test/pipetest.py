import json
data = input()
result = json.loads(data)
print(f'Python got data and parsed {result["hoge"]}\n')