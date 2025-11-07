import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf8')

data = sys.stdin.read().strip().split()
if not data:
    sys.exit(0)

nums = list(map(int, data))
pos_cnt = 0
pos_sum = 0 
neg_cnt = 0
for num in nums:
    if num < 0:
        neg_cnt += 1
    elif num > 0:
        pos_cnt += 1
        pos_sum += num
        
print(neg_cnt)
print(f"{(round(pos_sum/pos_cnt, 2)):.2f}")
    
