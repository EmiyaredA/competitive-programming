import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

data = sys.stdin.read().strip().split()
if not data:
    sys.exit(0)

n = int(data[0])
nums = list(map(int, data[1:1+n]))
print(sum([_ for _ in nums if _%2==0]))
