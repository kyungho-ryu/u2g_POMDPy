from collections import deque

backward_traj = deque()

backward_traj.append(1)
backward_traj.append(2)
backward_traj.append(3)
print(backward_traj.popleft())
print(backward_traj.popleft())
print(backward_traj.popleft())
print(backward_traj.popleft())
print(backward_traj)